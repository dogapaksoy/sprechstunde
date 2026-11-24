"""Voice layer: Whisper in, Piper out, and a readback check in between.

The agent proves every date and amount it *writes*. In a voice interface that is not
enough: the user only gets what they *hear*. Text-to-speech can garble a number ("27.03.2026"
comes out as something Whisper hears as "27.03.2006"), and the user can't scroll back.

So before an answer is played:

1. ``speakable`` rewrites dates and amounts into the forms people say and TTS engines
   pronounce reliably: "27.03.2026" -> "27. März 2026", "4.799,10 €" -> "4799,10 Euro".
2. The audio is synthesised, transcribed again with Whisper, and the dates and amounts
   found in the transcript must equal those in the text (``readback_ok``). If they don't,
   it is synthesised once more, slower. If that fails too, the assistant does not play a
   possibly wrong number; it says it will show the answer on screen instead.

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
    return values(text) == values(transcript)


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

    def __init__(self, whisper: str = "small", piper: str = "de_DE-thorsten-medium") -> None:
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

    def transcribe(self, audio: str | Path | bytes) -> str:
        source = io.BytesIO(audio) if isinstance(audio, bytes) else str(audio)
        segments, _ = self.stt.transcribe(source, language="de", beam_size=1, vad_filter=True)
        return " ".join(s.text.strip() for s in segments).strip()

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
            transcript = self.transcribe(audio)
            seconds["readback"] = seconds.get("readback", 0) + time.perf_counter() - t0
            if readback_ok(text, transcript):
                return Spoken(spoken, audio, transcript, True, attempt, seconds)
        return Spoken(NOT_SPOKEN, self.synthesize(NOT_SPOKEN), transcript, False, 2, seconds)
