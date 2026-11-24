"""Voice layer: Whisper in, Piper out, and a readback check in between.

The agent proves every date and amount it *writes*. In a voice interface that is not
enough: the user only gets what they *hear*, and can't scroll back. Synthesised from raw
text, "27.03.2026" is read digit group by digit group; written out, it is fine.

So before an answer is played:

1. ``speakable`` rewrites dates and amounts into the forms people say and TTS engines
   pronounce reliably: "27.03.2026" -> "27. März 2026", "4.799,10 €" -> "4799,10 Euro".
2. The audio is synthesised, transcribed again with Whisper, and the dates and amounts
   found in the transcript must equal those in the text (``readback_ok``). If they don't,
   it is synthesised once more, slower. If that fails too, the assistant does not play a
   possibly wrong number; it says it will show the answer on screen instead.

The readback check is only as good as the ear that listens. Piper's phonemes for numbers
are right ("acht und zwanzig"), but Whisper writes German compound numbers literally
("8 und 20", "2006 und 20"). Two fixes, measured in ``voice_eval``: a prompt that shows
Whisper numbers written as digits, and ``repair_numbers`` for the "A und B0" pattern.
Whisper large-v3-turbo then reads back 98% of verified values (small: 84%).

Everything runs locally on a CPU: faster-whisper (int8) and Piper (ONNX).
"""

from __future__ import annotations

import io
import re
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

from belegt.values import find_amounts, find_dates, parse_date

MONTHS = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August",
          "September", "Oktober", "November", "Dezember"]  # fmt: skip

NOT_SPOKEN = (
    "Ich kann die Zahl nicht sicher vorlesen. Ich zeige Ihnen die Antwort auf dem Bildschirm."
)

_NUMERIC_DATE = re.compile(r"\b(\d{1,2})\.\s?(\d{1,2})\.\s?(\d{4})\b")
_NUM = r"(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d{2}))?"
_AMOUNT = re.compile(rf"(?:€|\bEUR)\s?{_NUM}|\b{_NUM}\s?(?:€|EUR\b|Euro\b)")
_TIME = re.compile(r"\b(\d{1,2}):(\d{2})\s?Uhr\b")

# Shows Whisper how numbers should be written; used when reading an answer back.
NUMBER_PROMPT = (
    "Ihr Termin ist am 28. März 2026 um 9 Uhr. Bitte zahlen Sie 1035,32 Euro bis zum "
    "14. August 2026."
)
# Names of German authorities, given to Whisper as hotwords when it hears a question.
# Without them it heard "Jobcenter" as "Jobcounter", and the topic check then (correctly)
# found no letter from a "Jobcounter". A fixed domain list, not the user's letters: biasing
# with the letters' words made Whisper hear them in questions about other matters.
AUTHORITIES = (
    "Finanzamt Jobcenter Agentur für Arbeit Ausländerbehörde Bürgeramt Krankenkasse "
    "Rentenversicherung Familienkasse Beitragsservice Rundfunkbeitrag Standesamt "
    "Zulassungsstelle"
)
_UND = re.compile(r"\b(\d+) und ([2-9]0)\b")
_DAY_COMMA = re.compile(rf"\b(\d{{1,2}}), ({'|'.join(MONTHS)})\b")
_ONE_CENT_DIGIT = re.compile(r"\b(\d+),(\d)(?=\s*(?:€|Euro|EUR)\b)")


def repair_numbers(transcript: str) -> str:
    """Undo Whisper's literal transcription of German number words: "achtundzwanzig" is
    written "8 und 20", "zweitausendsechsundzwanzig" "2006 und 20". Also "26, Juni" ->
    "26. Juni" and "2037,5 Euro" -> "2037,50 Euro"."""

    def join(m: re.Match[str]) -> str:
        a, tens = int(m.group(1)), int(m.group(2))
        units = a % 10
        if a % 100 >= 10 or units == 0:  # only a bare unit can precede "und <tens>"
            return m.group(0)
        return str(a - a % 100 + tens + units)

    transcript = _UND.sub(join, transcript)
    transcript = _DAY_COMMA.sub(r"\1. \2", transcript)
    return _ONE_CENT_DIGIT.sub(r"\1,\g<2>0", transcript)


def _say_amount(euros: str, cents: str | None) -> str:
    euros = euros.replace(".", "")
    if not cents or cents == "00":
        return f"{euros} Euro"
    return f"{euros},{cents} Euro"  # "4799,10 Euro": Whisper writes it back the same way


def speakable(text: str) -> str:
    """Rewrite dates, amounts and times into the forms a German speaker would say."""

    def date(m: re.Match[str]) -> str:
        d = parse_date(m.group(0))
        return f"{d.day}. {MONTHS[d.month - 1]} {d.year}" if d else m.group(0)

    def amount(m: re.Match[str]) -> str:
        euros, cents = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        return _say_amount(euros, cents)

    def clock(m: re.Match[str]) -> str:
        h, mins = int(m.group(1)), int(m.group(2))
        return f"{h} Uhr" if mins == 0 else f"{h} Uhr {mins}"

    text = _NUMERIC_DATE.sub(date, text)
    text = _TIME.sub(clock, text)
    return _AMOUNT.sub(amount, text)


def values(text: str) -> tuple[set, set]:
    return set(find_dates(text)), set(find_amounts(text))


def readback_ok(text: str, transcript: str) -> bool:
    """Do the dates and amounts the user hears equal the ones that were verified?"""
    return values(text) == values(repair_numbers(transcript))


@dataclass
class Spoken:
    text: str  # what was synthesised
    audio: bytes  # WAV
    transcript: str  # what Whisper heard in that audio
    ok: bool
    attempts: int
    seconds: dict[str, float] = field(default_factory=dict)


class Voice:
    """Local speech in and out. Models are loaded on first use and cached by huggingface_hub."""

    def __init__(
        self, whisper: str = "large-v3-turbo", piper: str = "de_DE-thorsten-medium"
    ) -> None:
        self.whisper_name, self.piper_name = whisper, piper
        self._stt = self._tts = None

    # ------------------------------------------------------------------ models

    @property
    def stt(self):
        if self._stt is None:
            from faster_whisper import WhisperModel

            self._stt = WhisperModel(self.whisper_name, device="cpu", compute_type="int8")
        return self._stt

    @property
    def tts(self):
        if self._tts is None:
            from huggingface_hub import hf_hub_download
            from piper import PiperVoice

            lang, name, quality = self.piper_name.split("-")
            base = f"{lang[:2]}/{lang}/{name}/{quality}/{self.piper_name}.onnx"
            model = hf_hub_download("rhasspy/piper-voices", base)
            hf_hub_download("rhasspy/piper-voices", base + ".json")
            self._tts = PiperVoice.load(model)
        return self._tts

    # ------------------------------------------------------------------ io

    def transcribe(
        self, audio: str | Path | bytes, prompt: str | None = None, hotwords: str | None = None
    ) -> str:
        source = io.BytesIO(audio) if isinstance(audio, bytes) else str(audio)
        segments, _ = self.stt.transcribe(
            source, language="de", beam_size=1, vad_filter=True, initial_prompt=prompt,
            hotwords=hotwords,
        )  # fmt: skip
        return " ".join(s.text.strip() for s in segments).strip()

    def listen(self, audio: str | Path | bytes) -> str:
        """Transcribe a user's question, biased towards the names of authorities."""
        return self.transcribe(audio, hotwords=AUTHORITIES)

    def synthesize(self, text: str, length_scale: float = 1.0) -> bytes:
        from piper import SynthesisConfig

        buf = io.BytesIO()
        with wave.open(buf, "wb") as f:
            self.tts.synthesize_wav(text, f, syn_config=SynthesisConfig(length_scale=length_scale))
        return buf.getvalue()

    def say(self, text: str, check: bool = True) -> Spoken:
        """Synthesise ``text`` for playback; with ``check``, only if the readback matches."""
        spoken = speakable(text)
        seconds: dict[str, float] = {}
        audio, transcript = b"", ""
        for attempt, scale in enumerate((1.0, 1.25), start=1):
            t0 = time.perf_counter()
            audio = self.synthesize(spoken, scale)
            seconds["tts"] = seconds.get("tts", 0) + time.perf_counter() - t0
            if not check:
                return Spoken(spoken, audio, "", True, attempt, seconds)
            t0 = time.perf_counter()
            transcript = self.transcribe(audio, prompt=NUMBER_PROMPT)
            seconds["readback"] = seconds.get("readback", 0) + time.perf_counter() - t0
            if readback_ok(text, transcript):
                return Spoken(spoken, audio, transcript, True, attempt, seconds)
        return Spoken(NOT_SPOKEN, self.synthesize(NOT_SPOKEN), transcript, False, 2, seconds)
