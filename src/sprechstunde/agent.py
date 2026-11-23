"""The agent loop: tools, verification guardrail and escalation.

One episode = one user question. Each turn the model emits one JSON action (schema-
constrained). Tool results are fed back until it answers or a limit is hit.

Evidence is cited by sentence id ("S12") rather than copied: search results are
numbered, and claims point at a number. That removes copy errors and cuts the tokens a
small CPU model must generate per answer — the main latency cost of a voice turn.

Guardrails
* only the four tools below exist; at most ``max_steps`` model calls per episode
* side effects are verified *before* they happen: a reminder is created only if its
  date is in the cited sentence, and only after the user confirms
* the answer is spoken aloud, so every date and amount in it must be backed by a claim
  whose value belegt finds in the cited sentence. Otherwise the model gets one chance to
  correct itself, then the question goes to a larger model, and if that fails too the
  agent says it cannot confirm the answer instead of guessing.
* a grounded value can still answer a different question (the Finanzamt deadline for a
  question about the dog tax), so the letters cited must also be *about* the question's
  topics (``relevance``: an NLI check, see ``sprechstunde.relevance``).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from belegt import Claim, FuzzyMatcher, ValueType, Verifier
from belegt.values import find_amounts, find_dates, parse_amount, parse_date

from sprechstunde.llm import ChatBackend
from sprechstunde.mailbox import Mailbox
from sprechstunde.relevance import TopicCheck, topics

FALLBACK = "Das kann ich in Ihren Briefen leider nicht sicher bestätigen."

SYSTEM = """You are "Sprechstunde", a voice assistant that helps people with their official \
German letters. You may only state facts that are written in the user's letters. Use the \
tools to find them. Sentences in tool results have ids like S12.

Reply with exactly one JSON action per turn:
- {"tool": "search_letters", "query": "<keywords>"}: find sentences in the letters
- {"tool": "read_letter", "letter_id": "<id>"}: read one whole letter
- {"tool": "add_reminder", "title": "<short title>", "date": "YYYY-MM-DD", "source": "<id of the \
sentence with the date>"}: create a reminder
- {"tool": "answer", "say": "<answer>", "claims": [...]}: finish

If the user asks to be reminded ("Erinnere mich ..."), you MUST call add_reminder before you \
answer.

Rules for "answer":
- "say" is spoken aloud: one or two short German sentences. Write dates like 15.03.2026 and \
amounts like 93,00 Euro. Call the letters "Briefe".
- Every date or amount in "say" must be listed in "claims" as {"type": "date" or "amount", \
"value": "<the value>", "source": "<sentence id>"}.
- If the letters do not contain the answer, say so and give no claims. Never guess."""

_CLAIM = {
    "type": "object",
    "properties": {
        "type": {"enum": ["date", "amount", "id"]},
        "value": {"type": "string"},
        "source": {"type": "string"},
    },
    "required": ["type", "value", "source"],
    "additionalProperties": False,
}


def _action(tool: str, **props: Any) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"tool": {"const": tool}, **props},
        "required": ["tool", *props],
        "additionalProperties": False,
    }


ACTION_SCHEMA: dict[str, Any] = {
    "anyOf": [
        _action("search_letters", query={"type": "string"}),
        _action("read_letter", letter_id={"type": "string"}),
        _action(
            "add_reminder",
            title={"type": "string"},
            date={"type": "string"},
            source={"type": "string"},
        ),  # fmt: skip
        _action(
            "answer",
            say={"type": "string"},
            claims={"type": "array", "items": _CLAIM, "maxItems": 4},
        ),  # fmt: skip
    ]
}


@dataclass
class Step:
    model: str
    action: dict[str, Any]
    observation: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float


@dataclass
class Reminder:
    title: str
    date: str
    source: str
    letter_id: str
    sentence: str


@dataclass
class Result:
    question: str
    say: str
    claims: list[dict[str, Any]] = field(default_factory=list)
    reminders: list[Reminder] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    verified: bool = False  # the final answer passed verification (always False if verify=False)
    escalated: bool = False
    fallback: bool = False  # the agent gave up and said it cannot confirm
    problems: list[str] = field(default_factory=list)  # why the last answer was rejected
    primary_reminders: int = 0  # reminders created by the small model before escalation

    @property
    def llm_seconds(self) -> float:
        return sum(s.seconds for s in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Agent:
    def __init__(
        self,
        backend: ChatBackend,
        mailbox: Mailbox,
        *,
        verify: bool = True,
        escalate_to: ChatBackend | None = None,
        max_steps: int = 6,
        retries: int = 1,
        confirm: Callable[[Reminder], bool] = lambda _r: True,
        relevance: TopicCheck | None = None,
    ) -> None:
        self.backend = backend
        self.mailbox = mailbox
        self.verify = verify
        self.escalate_to = escalate_to
        self.max_steps = max_steps
        self.retries = retries
        self.confirm = confirm
        self.relevance = relevance
        self._topics: list[str] = []
        self.verifier = Verifier(FuzzyMatcher(85))
        self.reminders: list[Reminder] = []
        self._wants_reminder = False
        self._scope: set[str] = set()

    # ------------------------------------------------------------------ public

    def ask(self, question: str) -> Result:
        result = self._episode(self.backend, question)
        if not self.verify or result.verified:
            return result
        if self.escalate_to is not None:
            second = self._episode(self.escalate_to, question)
            second.primary_reminders = len(result.reminders)
            second.steps = result.steps + second.steps
            second.reminders = result.reminders + second.reminders
            second.escalated = True
            if second.verified:
                return second
            result = second
        result.say, result.claims, result.fallback = FALLBACK, [], True
        return result

    # ------------------------------------------------------------------ episode

    def _episode(self, backend: ChatBackend, question: str) -> Result:
        res = Result(question=question, say="")
        letters = "\n".join(f"- {lid}: {lt.sender}" for lid, lt in self.mailbox.letters.items())
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Letters:\n{letters}\n\nQuestion: {question} /no_think"},
        ]
        retries = self.retries
        seen: set[str] = set()
        self._wants_reminder = bool(re.search(r"\berinner|\bremind", question, re.IGNORECASE))
        self._scope = self.mailbox.mentioned(question)
        self._topics = topics(question)
        for _ in range(self.max_steps):
            out = backend.complete(messages, ACTION_SCHEMA)
            try:
                action = json.loads(out.text)
            except json.JSONDecodeError:
                action = {"tool": "invalid", "raw": out.text[:200]}
            key = json.dumps(action, sort_keys=True)
            if key in seen and action.get("tool") in ("search_letters", "read_letter"):
                done, observation = (
                    False,
                    (
                        "You already did exactly this and have the result above. Answer now, or "
                        "say that the letters do not contain the answer."
                    ),
                )
            else:
                done, observation = self._act(action, res, retries)
            seen.add(key)
            res.steps.append(Step(backend.name, action, observation, out.prompt_tokens,
                                  out.completion_tokens, out.seconds))  # fmt: skip
            if done:
                return res
            if action.get("tool") == "answer":
                retries -= 1
            messages.append({"role": "assistant", "content": out.text})
            messages.append({"role": "user", "content": observation})
        res.problems.append("step limit reached")
        return res

    def _act(self, a: dict[str, Any], res: Result, retries: int) -> tuple[bool, str]:
        tool = a.get("tool")
        if tool == "search_letters":
            hits = self.mailbox.search(str(a.get("query", "")))
            if not hits:
                return False, "search_letters: no matching sentences."
            lines = [f'{h.sid} [{h.letter_id}] "{h.sentence}"' for h in hits]
            return False, "search_letters results:\n" + "\n".join(lines)
        if tool == "read_letter":
            lid = str(a.get("letter_id", "")).strip()
            if self.mailbox.get(lid) is None and (found := self.mailbox.sentence(lid)):
                lid = found[0]  # small models pass a sentence id ("S12"); open its letter
            if self.mailbox.get(lid) is None:
                return False, f"read_letter: unknown id. Known: {', '.join(self.mailbox.letters)}"
            lines = [f'{sid} "{s}"' for sid, s in self.mailbox.numbered(lid)]
            return False, f"Letter {lid}:\n" + "\n".join(lines)
        if tool == "add_reminder":
            return False, self._add_reminder(a, res)
        if tool == "answer":
            res.say = str(a.get("say", ""))
            res.claims = list(a.get("claims", []))
            if not self.verify:
                return True, ""
            res.problems = self.check_answer(res.say, res.claims)
            if self._wants_reminder and not res.reminders:
                res.problems.append(
                    "the user asked for a reminder but none was created: call add_reminder with "
                    "the date and the id of the sentence that contains it"
                )
            if not res.problems:
                res.verified = True
                return True, ""
            if retries <= 0:
                return True, ""
            return False, (
                "Your answer was NOT spoken because it could not be verified:\n- "
                + "\n- ".join(res.problems)
                + "\nCheck the sentences and answer again. If the letters do not contain the "
                "answer, say so without claims."
            )
        return False, "Invalid action. Use search_letters, read_letter, add_reminder or answer."

    # ------------------------------------------------------------------ guardrails

    def grounded(self, kind: ValueType, value: str, source: str) -> bool:
        """Is ``value`` in sentence ``source`` (checked by belegt, OCR-tolerant)?"""
        found = self.mailbox.sentence(source)
        if found is None:
            return False
        letter_id, sentence = found
        claim = Claim("claim", value, sentence, kind)
        return self.verifier.verify(self.mailbox.letters[letter_id].text, claim).supported

    def _add_reminder(self, a: dict[str, Any], res: Result) -> str:
        date, source = str(a.get("date", "")), str(a.get("source", ""))
        if parse_date(date) is None:
            return "add_reminder REJECTED: date must be YYYY-MM-DD."
        if self.verify and not self.grounded(ValueType.DATE, date, source):
            return (f"add_reminder REJECTED: {date} is not in sentence {source}. Find the "
                    "sentence that contains the date first.")  # fmt: skip
        if self.verify and not self._in_scope(source):
            return f"add_reminder REJECTED: {self._scope_problem(source)}."
        if self.verify and (off := self._off_topic([source])):
            return f"add_reminder REJECTED: {off[0]}."
        letter_id, sentence = self.mailbox.sentence(source) or ("", "")
        r = Reminder(str(a.get("title", "")), date, source, letter_id, sentence)
        if not self.confirm(r):
            return "add_reminder: the user declined."
        self.reminders.append(r)
        res.reminders.append(r)
        return f"add_reminder OK: reminder '{r.title}' on {r.date} created. Now answer the user."

    def check_answer(self, say: str, claims: list[dict[str, Any]]) -> list[str]:
        """Problems that block an answer from being spoken (empty list = OK).

        Only what the user hears is checked: every date and amount in ``say`` must be backed
        by a claim whose value belegt finds in the cited sentence. Claims for values that
        are not spoken are ignored, so a stray extra claim cannot block a correct answer.
        If the question names a letter ("beim Jobcenter"), evidence must come from it, and
        with ``relevance`` the cited letters must be about the question's topics."""
        problems = []
        spoken_dates, spoken_amounts = set(find_dates(say)), set(find_amounts(say))
        dates, amounts = set(), set()
        cited = []
        for c in claims:
            try:
                kind, value, source = ValueType(c["type"]), str(c["value"]), str(c["source"])
            except (KeyError, ValueError):
                continue
            if kind is ValueType.DATE:
                parsed, spoken, verified = parse_date(value), spoken_dates, dates
            elif kind is ValueType.AMOUNT:
                parsed, spoken, verified = parse_amount(value), spoken_amounts, amounts
            else:
                continue
            if parsed is None or parsed not in spoken:
                continue
            if not self.grounded(kind, value, source):
                problems.append(f"{value!r} does not appear in sentence {source}")
            elif not self._in_scope(source):
                problems.append(self._scope_problem(source))
            else:
                verified.add(parsed)
                cited.append(source)
        problems += self._off_topic(cited)
        for d in sorted(spoken_dates - dates):
            problems.append(f"the date {d:%d.%m.%Y} in your answer has no verified claim")
        for m in sorted(spoken_amounts - amounts):
            problems.append(f"the amount {m} in your answer has no verified claim")
        return list(dict.fromkeys(problems))

    def _off_topic(self, sources: list[str]) -> list[str]:
        """Per cited letter: its sender, subject and cited sentences must be about every
        topic of the question."""
        if self.relevance is None or not self._topics:
            return []
        by_letter: dict[str, list[str]] = {}
        for s in sources:
            if found := self.mailbox.sentence(s):
                by_letter.setdefault(found[0], []).append(found[1])
        problems = []
        for lid, sentences in by_letter.items():
            lt = self.mailbox.letters[lid]
            evidence = " ".join([lt.sender, lt.subject, *dict.fromkeys(sentences)])
            if missing := self.relevance(evidence, self._topics):
                problems.append(
                    f"letter {lid} is not about {', '.join(missing)}. If no letter answers "
                    "the question, say so without any date or amount"
                )
        return problems

    def _in_scope(self, source: str) -> bool:
        found = self.mailbox.sentence(source)
        return not self._scope or (found is not None and found[0] in self._scope)

    def _scope_problem(self, source: str) -> str:
        found = self.mailbox.sentence(source)
        named = ", ".join(sorted(self._scope))
        return (f"sentence {source} is from letter {found[0] if found else '?'}, but the "
                f"question is about letter {named}")  # fmt: skip
