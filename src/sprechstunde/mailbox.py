"""The user's letters, split into sentences and searchable.

Letters arrive as OCR text: paragraphs re-wrapped at the page width, words hyphenated
across lines. Sentences are reassembled before indexing so that a search hit is a
complete, quotable sentence, which is what the agent must cite as evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_STOPWORDS = {
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem", "und", "oder",
    "ich", "mein", "meine", "meinen", "mir", "mich", "sie", "ihr", "ihre", "ihren", "wir",
    "ist", "sind", "bin", "wird", "werden", "muss", "müssen", "kann", "soll", "habe", "hat",
    "bis", "wann", "wie", "was", "wo", "wer", "welche", "welcher", "viel", "hoch", "lautet",
    "zu", "zum", "zur", "beim", "bei", "von", "vom", "für", "auf", "an", "am", "im", "in",
    "mit", "nach", "noch", "nicht", "kein", "keine", "bitte", "the", "my", "is", "what",
}  # fmt: skip


def _fold(word: str) -> str:
    word = word.casefold()
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        word = word.replace(a, b)
    return word


_STOPWORDS = {_fold(w) for w in _STOPWORDS}  # compare in the same folded form as tokens


def _tokens(text: str) -> list[str]:
    return [t for t in (_fold(w) for w in re.findall(r"[^\W_]+", text)) if len(t) > 2]


def _related(a: str, b: str) -> bool:
    """German inflection and compounds: "zahlen" ~ "Zahlung", "Gebühr" ~
    "Bearbeitungsgebühr", "einreichen" ~ "reichen"."""
    n = min(len(a), len(b), 5)
    if n >= 4 and a[:n] == b[:n]:
        return True
    short, long_ = sorted((a, b), key=len)
    return len(short) >= 5 and short in long_


def reassemble(text: str) -> list[str]:
    """Undo page layout: join soft-wrapped lines and hyphenation, then split sentences."""
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        lines = [ln.strip() for ln in para.splitlines() if ln.strip()]
        buf = ""
        for ln in lines:
            if not buf:
                buf = ln
            elif buf.endswith("-") and ln[:1].islower():
                buf = buf[:-1] + ln
            elif len(buf) > 45 and not buf.endswith((".", ":", "!", "?", ",")):
                buf += " " + ln  # a wrapped body line, not a letterhead line
            elif ln[:1].islower():
                buf += " " + ln
            else:
                units.append(buf)
                buf = ln
        if buf:
            units.append(buf)
    sentences = []
    for unit in units:
        # split after . ! ? unless it ends a number ("5. März", "15.03.") or an abbreviation
        parts = re.split(r"(?<=[^\d\s][.!?])\s+(?=[A-ZÄÖÜ])", unit)
        sentences += [p.strip() for p in parts if p.strip()]
    return sentences


@dataclass
class Letter:
    id: str
    text: str
    sentences: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.sentences:
            self.sentences = reassemble(self.text)

    @property
    def sender(self) -> str:
        return self.sentences[0] if self.sentences else ""

    @property
    def subject(self) -> str:
        return next((s for s in self.sentences if _fold(s).startswith("betreff")), "")


@dataclass(frozen=True)
class Hit:
    sid: str  # sentence id, e.g. "S12": the agent cites sentences by id instead of copying them
    letter_id: str
    sentence: str
    score: float


class Mailbox:
    def __init__(self, letters: list[Letter]) -> None:
        self.letters = {lt.id: lt for lt in letters}
        self._sid: dict[tuple[str, int], str] = {}
        self.sentences: dict[str, tuple[str, str]] = {}  # sid -> (letter_id, sentence)
        for lt in letters:
            for i, sentence in enumerate(lt.sentences):
                sid = f"S{len(self.sentences) + 1}"
                self._sid[(lt.id, i)] = sid
                self.sentences[sid] = (lt.id, sentence)

    def get(self, letter_id: str) -> Letter | None:
        return self.letters.get(letter_id)

    def sentence(self, sid: str) -> tuple[str, str] | None:
        return self.sentences.get(sid.strip().upper())

    def numbered(self, letter_id: str) -> list[tuple[str, str]]:
        lt = self.letters[letter_id]
        return [(self._sid[(lt.id, i)], s) for i, s in enumerate(lt.sentences)]

    def overview(self) -> str:
        return "\n".join(f"- {lt.id}: {lt.sender} | {lt.subject}" for lt in self.letters.values())

    def _header(self, lt: Letter) -> list[str]:
        return _tokens(lt.sender + " " + lt.subject + " " + lt.id)

    def mentioned(self, text: str) -> set[str]:
        """Letters whose sender or subject ``text`` names ("beim Jobcenter"). Stricter than
        search matching: whole words or long substrings only, so it can be used to reject
        evidence from the wrong letter. Empty if no letter is named (e.g. OCR damaged it)."""
        words = [t for t in _tokens(text) if t not in _STOPWORDS]
        named = set()
        for lt in self.letters.values():
            for t in words:
                for h in self._header(lt):
                    short, long_ = sorted((t, h), key=len)
                    if t == h or (len(short) >= 6 and short in long_):
                        named.add(lt.id)
        return named

    def search(self, query: str, k: int = 5) -> list[Hit]:
        """Lexical search with prefix matching for German word forms. If the query names a
        letter's sender or subject ("Jobcenter"), that letter's sentences are included even
        when they share no other word with the query ("Termin" vs. "Gespräch am ...")."""
        q = [t for t in _tokens(query) if t not in _STOPWORDS]
        if not q:
            return []
        hits = []
        for lt in self.letters.values():
            header = self._header(lt)
            boost = sum(any(_related(t, h) for h in header) for t in q)
            for i, s in enumerate(lt.sentences):
                words = _tokens(s)
                score = sum(any(_related(t, w) for w in words) for t in q)
                if score or boost:
                    informative = 0.25 if len(s) > 30 and re.search(r"\d", s) else 0.0
                    hits.append(
                        Hit(self._sid[(lt.id, i)], lt.id, s, score + 0.5 * boost + informative)
                    )
        hits.sort(key=lambda h: -h.score)
        return hits[:k]
