"""Agent logic with scripted model responses: no model download needed."""

import pytest

from sprechstunde.agent import FALLBACK, Agent
from sprechstunde.llm import ScriptedBackend
from sprechstunde.mailbox import Letter, Mailbox

JOBCENTER = """Jobcenter Karlsruhe
Kaiserstraße 12 · 76131 Karlsruhe

Karlsruhe, 02.03.2026

Kundennummer: 123D456789

Betreff: Einladung zu einem Gespräch

Wir laden Sie zu einem persönlichen Gespräch am 16.03.2026 um 10:00 Uhr ein.

Bitte reichen Sie die fehlenden Kontoauszüge der letzten drei Monate bis zum 27.03.2026 ein."""

AMT = """Stadt – Ausländerbehörde Karlsruhe

Karlsruhe, 01.03.2026

Die Bearbeitungsgebühr beträgt 93,00 €."""


@pytest.fixture
def mailbox() -> Mailbox:
    return Mailbox([Letter("jobcenter", JOBCENTER), Letter("auslaenderbehoerde", AMT)])


def sid(mailbox: Mailbox, text: str) -> str:
    return next(s for s, (_, sentence) in mailbox.sentences.items() if text in sentence)


def answer(say, *claims):
    return {"tool": "answer", "say": say, "claims": list(claims)}


def test_grounded_answer_is_spoken(mailbox):
    s = sid(mailbox, "Kontoauszüge")
    llm = ScriptedBackend([
        {"tool": "search_letters", "query": "Kontoauszüge Jobcenter"},
        answer("Bis zum 27.03.2026.", {"type": "date", "value": "2026-03-27", "source": s}),
    ])  # fmt: skip
    r = Agent(llm, mailbox).ask("Bis wann muss ich die Kontoauszüge einreichen?")
    assert r.verified and r.say == "Bis zum 27.03.2026." and not r.fallback
    assert s in llm.calls[1][-1]["content"]  # the search result showed the sentence id


def test_value_from_wrong_sentence_is_rejected_then_corrected(mailbox):
    letter_date, deadline = sid(mailbox, "Karlsruhe, 02.03"), sid(mailbox, "Kontoauszüge")
    llm = ScriptedBackend([
        answer("Bis zum 02.03.2026.", {"type": "date", "value": "2026-03-02", "source": deadline}),
        answer("Bis zum 27.03.2026.", {"type": "date", "value": "2026-03-27", "source": deadline}),
    ])  # fmt: skip
    r = Agent(llm, mailbox).ask("Bis wann muss ich die Kontoauszüge einreichen?")
    assert r.verified and "27.03" in r.say
    assert "NOT spoken" in llm.calls[1][-1]["content"]
    assert letter_date != deadline


def test_unverifiable_answer_escalates_then_falls_back(mailbox):
    bad = answer("Sie zahlen 50,00 Euro.", {"type": "amount", "value": "50.00", "source": "S1"})
    small, large = ScriptedBackend([bad, bad], "small"), ScriptedBackend([bad, bad], "large")
    r = Agent(small, mailbox, escalate_to=large).ask("Wie hoch ist die Gebühr?")
    assert r.escalated and r.fallback and r.say == FALLBACK
    assert [s.model for s in r.steps] == ["small", "small", "large", "large"]


def test_escalation_can_rescue_the_answer(mailbox):
    s = sid(mailbox, "Bearbeitungsgebühr")
    bad = answer("Sie zahlen 50,00 Euro.", {"type": "amount", "value": "50.00", "source": s})
    good = answer(
        "Die Gebühr beträgt 93,00 Euro.", {"type": "amount", "value": "93.00", "source": s}
    )
    r = Agent(ScriptedBackend([bad, bad]), mailbox, escalate_to=ScriptedBackend([good])).ask(
        "Gebühr?"
    )
    assert r.escalated and r.verified and "93,00" in r.say


def test_number_in_speech_without_claim_is_blocked(mailbox):
    llm = ScriptedBackend(
        [answer("Bis zum 27.03.2026."), answer("Das steht nicht in Ihren Briefen.")]
    )
    r = Agent(llm, mailbox).ask("Bis wann?")
    assert r.verified and r.say == "Das steht nicht in Ihren Briefen."


def test_reminder_needs_grounded_date_and_confirmation(mailbox):
    s = sid(mailbox, "Kontoauszüge")
    llm = ScriptedBackend([
        {"tool": "add_reminder", "title": "Kontoauszüge", "date": "2026-03-28", "source": s},
        {"tool": "add_reminder", "title": "Kontoauszüge", "date": "2026-03-27", "source": s},
        answer("Erinnerung für den 27.03.2026 angelegt.", {"type": "date", "value": "2026-03-27", "source": s}),  # noqa: E501
    ])  # fmt: skip
    asked = []
    agent = Agent(llm, mailbox, confirm=lambda r: asked.append(r) or True)
    r = agent.ask("Erinnere mich an die Frist für die Kontoauszüge.")
    assert "REJECTED" in llm.calls[1][-1]["content"]
    assert [x.date for x in r.reminders] == ["2026-03-27"] and len(asked) == 1


def test_declined_reminder_is_not_created(mailbox):
    s = sid(mailbox, "Kontoauszüge")
    llm = ScriptedBackend([
        {"tool": "add_reminder", "title": "x", "date": "2026-03-27", "source": s},
        answer("In Ordnung, keine Erinnerung."),
    ])  # fmt: skip
    r = Agent(llm, mailbox, confirm=lambda r: False, retries=0).ask("Bitte keine Erinnerung.")
    assert r.reminders == [] and "declined" in llm.calls[1][-1]["content"]


def test_reminder_request_cannot_be_answered_without_reminder(mailbox):
    llm = ScriptedBackend([answer("Die Frist ist bald."), answer("Erledigt.")])
    r = Agent(llm, mailbox, retries=1).ask("Erinnere mich an die Frist.")
    assert "asked for a reminder" in llm.calls[1][-1]["content"]
    assert r.fallback


def test_read_letter_accepts_sentence_id_and_repeated_actions_are_stopped(mailbox):
    s = sid(mailbox, "Kontoauszüge")
    llm = ScriptedBackend([
        {"tool": "read_letter", "letter_id": s},
        {"tool": "read_letter", "letter_id": s},
        answer("Das steht nicht in Ihren Briefen."),
    ])  # fmt: skip
    Agent(llm, mailbox).ask("Was steht im Brief?")
    assert llm.calls[1][-1]["content"].startswith("Letter jobcenter:")
    assert "already did exactly this" in llm.calls[2][-1]["content"]


def test_without_verification_anything_is_spoken(mailbox):
    llm = ScriptedBackend([answer("Sie zahlen 50,00 Euro.")])
    r = Agent(llm, mailbox, verify=False).ask("Gebühr?")
    assert r.say == "Sie zahlen 50,00 Euro." and not r.verified


def test_step_limit(mailbox):
    llm = ScriptedBackend([{"tool": "search_letters", "query": f"q{i}"} for i in range(3)])
    r = Agent(llm, mailbox, max_steps=3).ask("Frage")
    assert r.fallback and "step limit reached" in r.problems


def test_evidence_from_another_letter_is_rejected_when_question_names_a_letter(mailbox):
    # The Ausländerbehörde letter has a "Termin" sentence too; the question is about the Jobcenter.
    amt = Mailbox([Letter("jobcenter", JOBCENTER),
                   Letter("auslaenderbehoerde", AMT + "\n\nBitte buchen Sie bis zum 20.03.2026 einen Termin.")])  # fmt: skip  # noqa: E501
    wrong, right = sid(amt, "20.03.2026"), sid(amt, "Gespräch am 16.03")
    llm = ScriptedBackend([
        answer("Am 20.03.2026.", {"type": "date", "value": "2026-03-20", "source": wrong}),
        answer("Am 16.03.2026.", {"type": "date", "value": "2026-03-16", "source": right}),
    ])  # fmt: skip
    r = Agent(llm, amt).ask("Wann ist mein Termin beim Jobcenter?")
    assert r.verified and "16.03" in r.say
    assert "question is about letter jobcenter" in llm.calls[1][-1]["content"]


def test_unspoken_extra_claim_does_not_block_a_correct_answer(mailbox):
    s = sid(mailbox, "Kontoauszüge")
    llm = ScriptedBackend([
        answer("Bis zum 27.03.2026.", {"type": "date", "value": "2026-03-27", "source": s},
               {"type": "amount", "value": "93.00", "source": s}),
    ])  # fmt: skip
    assert Agent(llm, mailbox).ask("Bis wann?").verified
