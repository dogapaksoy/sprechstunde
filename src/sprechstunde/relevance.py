"""Topic relevance: is the cited letter about what the question asks?

Value grounding (belegt) proves that a spoken date or amount is in the cited sentence.
It cannot tell whether that sentence answers the *question*: asked for the dog tax
("Hundesteuer"), a small model happily reads out the income-tax deadline from the
Finanzamt letter. The value is real, the answer is wrong.

German nouns are capitalised, so the question's topics are its capitalised words, minus
generic ones ("Frist", "Gebühr") that every letter matches. Each topic must be entailed
by the evidence: the letter's sender and subject plus the cited sentences, checked by a
multilingual NLI model with the hypothesis "Es geht um <topic>." World knowledge comes
for free: "Krankenkasse" is entailed by a letter from the AOK about a "Beitragsrückstand"
even though the word never appears in it.
"""

from __future__ import annotations

import re
from collections.abc import Callable

# Words that ask *what* is wanted, not *about which matter*; pronouns of formal address.
GENERIC = {
    "Betrag", "Brief", "Briefe", "Briefen", "Datum", "Euro", "Frist", "Gebühr", "Höhe",
    "Termin", "Zahlungsfrist", "Sie", "Ihr", "Ihre", "Ihnen", "Ihren", "Ihrem", "Ihrer",
}  # fmt: skip

TopicCheck = Callable[[str, list[str]], list[str]]
"""(evidence, topics) -> the topics the evidence is not about."""


def topics(question: str) -> list[str]:
    words = re.findall(r"[^\W\d_][\w-]*", question)[1:]  # the first word is capitalised anyway
    return list(dict.fromkeys(w for w in words if w[0].isupper() and w not in GENERIC))


class NLITopicCheck:
    """``TopicCheck`` backed by belegt's NLI model (needs ``belegt[nli]``); loads lazily."""

    def __init__(self, threshold: float = 0.5, model_name: str | None = None) -> None:
        self.threshold = threshold
        self.model_name = model_name
        self._model = None
        self._cache: dict[tuple[str, str], float] = {}

    def entailment(self, evidence: str, topic: str) -> float:
        key = (evidence, topic)
        if key not in self._cache:
            if self._model is None:
                from belegt.nli import DEFAULT_MODEL, NLIModel

                self._model = NLIModel(self.model_name or DEFAULT_MODEL)
            (p,) = self._model.predict([(evidence, f"Es geht um {topic}.")])
            self._cache[key] = p["entailment"]
        return self._cache[key]

    def __call__(self, evidence: str, topics_: list[str]) -> list[str]:
        return [t for t in topics_ if self.entailment(evidence, t) < self.threshold]
