from sprechstunde.agent import FALLBACK
from sprechstunde.evaluate import QUESTIONS, Task, episodes, outcome, without_escalation
from sprechstunde.mailbox import Letter, Mailbox, reassemble

SCANNED = """Jobcenter Karlsruhe
Kaiserstraße 12

Bitte reichen Sie die fehlenden Konto-
auszüge der letzten drei Monate bis zum
27.03.2026 ein. Kommen Sie ohne wichtigen Grund nicht, kann Ihr Bürgergeld gemindert werden."""


def test_reassemble_joins_wrapped_lines_and_hyphenation():
    s = reassemble(SCANNED)
    assert s[:2] == ["Jobcenter Karlsruhe", "Kaiserstraße 12"]
    assert s[2] == ("Bitte reichen Sie die fehlenden Kontoauszüge der letzten drei Monate bis zum "
                    "27.03.2026 ein.")  # fmt: skip
    assert s[3].startswith("Kommen Sie")


def test_sentences_do_not_split_on_dates():
    assert reassemble("Ihr Titel läuft am 13. Juli 2026 ab. Bitte buchen Sie.") == [
        "Ihr Titel läuft am 13. Juli 2026 ab.",
        "Bitte buchen Sie.",
    ]


def test_search_handles_compounds_and_misses_absent_topics():
    mb = Mailbox(
        [Letter("jobcenter", SCANNED), Letter("amt", "Die Bearbeitungsgebühr beträgt 93,00 €.")]
    )
    assert mb.search("Wie hoch ist die Gebühr?")[0].letter_id == "amt"
    assert "Kontoauszüge" in mb.search("Kontoauszüge einreichen")[0].sentence
    assert mb.search("Wie hoch ist meine Stromrechnung?") == []


def test_sentence_ids_are_unique_across_letters():
    mb = Mailbox([Letter("a", "Satz eins. Satz zwei."), Letter("b", "Satz drei.")])
    assert list(mb.sentences) == ["S1", "S2", "S3"] and mb.sentence("s3") == ("b", "Satz drei.")


def test_episodes_are_deterministic_and_well_formed():
    a, b = episodes(2, seed=1), episodes(2, seed=1)
    assert [t.question for _, ts in a for t in ts] == [t.question for _, ts in b for t in ts]
    mailbox, tasks = a[0]
    assert len(mailbox.letters) == 5
    assert [t.kind for t in tasks] == ["value", "value", "value", "reminder", "unanswerable"]
    assert all(t.question in QUESTIONS.values() for t in tasks[:3])


def test_outcome_scores_what_is_said():
    t = Task(0, "value", "Bis wann?", "date", "2026-03-27")
    assert outcome(t, "Bis zum 27.03.2026.", []) == "correct"
    assert outcome(t, "Bis zum 02.03.2026.", []) == "wrong"
    assert outcome(t, FALLBACK, []) == "abstained"
    u = Task(0, "unanswerable", "Stromrechnung?")
    assert outcome(u, "Das steht nicht in Ihren Briefen.", []) == "correct"
    assert outcome(u, "Sie zahlen 93,00 Euro.", []) == "wrong"
    r = Task(0, "reminder", "Erinnere mich", "date", "2026-03-27")
    assert outcome(r, "", [{"date": "2026-03-27"}]) == "correct"
    assert outcome(r, "", []) == "missing"


def test_without_escalation_replaces_escalated_answers_with_fallback():
    row = {"episode": 0, "kind": "value", "question": "q", "gold": "2026-03-27",
           "gold_type": "date", "say": "Bis zum 27.03.2026.", "escalated": True,
           "outcome": "correct", "reminders": [], "primary_reminders": 0,
           "steps": [{"model": "small", "seconds": 2.0}, {"model": "large", "seconds": 5.0}]}  # fmt: skip  # noqa: E501
    (derived,) = without_escalation([row], "small+verify")
    assert derived["say"] == FALLBACK and derived["outcome"] == "abstained"
    assert derived["llm_seconds"] == 2.0 and row["outcome"] == "correct"


def test_mentioned_letters_use_folded_stopwords():
    mb = Mailbox([Letter("a", "Finanzamt Köln\n\nBetreff: Bescheid für 2025"),
                  Letter("b", "Jobcenter Berlin")])  # fmt: skip
    assert mb.mentioned("Wann ist mein Termin beim Jobcenter?") == {"b"}
    assert mb.mentioned("Wie viel muss ich für den Rundfunkbeitrag zahlen?") == set()


def test_search_includes_sentences_of_a_named_letter():
    jc = "Jobcenter Berlin\n\nWir laden Sie zu einem Gespräch am 16.03.2026 ein."
    mb = Mailbox(
        [Letter("jc", jc), Letter("amt", "Bitte buchen Sie bis zum 20.03.2026 einen Termin.")]
    )
    found = [h.sentence for h in mb.search("Termin beim Jobcenter")]
    assert any("Gespräch am 16.03.2026" in s for s in found)
