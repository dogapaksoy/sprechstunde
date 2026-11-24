"""Text side of the voice layer; the audio models are exercised by ``sprechstunde voice-eval``."""

from sprechstunde.voice import readback_ok, speakable


def test_dates_amounts_and_times_are_rewritten_for_speech():
    assert speakable("Bis zum 27.03.2026.") == "Bis zum 27. März 2026."
    assert speakable("Die Gebühr beträgt 93,00 €.") == "Die Gebühr beträgt 93 Euro."
    assert speakable("Sie zahlen EUR 4.799,10 bis 1.2.2027.") == (
        "Sie zahlen 4799,10 Euro bis 1. Februar 2027."
    )
    assert speakable("Termin um 10:00 Uhr, Raum 2.14.") == "Termin um 10 Uhr, Raum 2.14."
    assert speakable("Aktenzeichen ABH-2026-22225") == "Aktenzeichen ABH-2026-22225"


def test_speakable_keeps_the_verified_values():
    for text in ["Bis zum 27.03.2026.", "Sie zahlen EUR 4.799,10.", "Am 7.6.2026 um 9:30 Uhr."]:
        assert readback_ok(text, speakable(text))


def test_readback_catches_a_misheard_year_or_amount():
    assert not readback_ok("Bis zum 27.03.2026.", "bis zum 27.03.2006")
    assert not readback_ok("Die Gebühr beträgt 98,00 €.", "Die Gebühr beträgt 8 Euro.")
    assert not readback_ok("Die Gebühr beträgt 98,00 €.", "Die Gebühr beträgt achtundneunzig Euro.")
    assert readback_ok("Das steht nicht in Ihren Briefen.", "Das steht nicht in Ihren Briefen.")


def test_repair_numbers_undoes_literal_german_number_words():
    from sprechstunde.voice import repair_numbers

    assert repair_numbers("bis zum 8 und 20. März 2006 und 20") == "bis zum 28. März 2026"
    assert repair_numbers("Sie zahlen 105 und 60,24 Euro") == "Sie zahlen 165,24 Euro"
    assert (
        repair_numbers("am 26, Juni 2026, also 2037,5 Euro")
        == "am 26. Juni 2026, also 2037,50 Euro"
    )
    assert repair_numbers("Seite 110 und 20 Zeilen") == "Seite 110 und 20 Zeilen"  # not a unit
    assert readback_ok("Bis zum 28.03.2026.", "bis zum 8 und 20. März 2006 und 20")
