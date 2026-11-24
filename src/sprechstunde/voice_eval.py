"""Benchmark the voice layer without a microphone: Piper speaks, Whisper listens.

1. Questions: every benchmark question is synthesised, optionally mixed with noise, and
   transcribed. Does the transcript keep the question's topics ("Hundesteuer") and does
   it still name the same letter? A misheard topic silently changes the question.
2. Answers: every distinct answer with a date or amount that the agent produced in the
   stored runs is synthesised and transcribed back. How often does the user hear a
   different value than the one that was verified: raw text vs. ``speakable`` text, and
   how often does the readback check have to fall back?
"""

from __future__ import annotations

import io
import json
import re
import statistics
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

from sprechstunde.evaluate import QUESTIONS, REMINDERS, UNANSWERABLE
from sprechstunde.mailbox import _fold
from sprechstunde.relevance import topics
from sprechstunde.voice import Voice, readback_ok, speakable, values


def add_noise(audio: bytes, snr_db: float, seed: int = 0) -> bytes:
    """Mix white noise into 16-bit mono WAV audio at the given signal-to-noise ratio."""
    with wave.open(io.BytesIO(audio)) as f:
        params, frames = f.getparams(), f.readframes(f.getnframes())
    x = np.frombuffer(frames, dtype=np.int16).astype(np.float64)
    noise = np.random.default_rng(seed).normal(size=x.shape)
    noise *= np.sqrt(np.mean(x**2) / (10 ** (snr_db / 10)) / np.mean(noise**2))
    y = np.clip(x + noise, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setparams(params)
        f.writeframes(y.tobytes())
    return buf.getvalue()


def _norm(text: str) -> list[str]:
    return re.findall(r"\w+", _fold(text))


def wer(ref: str, hyp: str) -> float:
    r, h = _norm(ref), _norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(len(r), 1)


def questions(voice: Voice, snrs: tuple[float | None, ...] = (None, 10.0, 5.0)) -> list[dict]:
    qs = sorted(set(QUESTIONS.values()) | set(REMINDERS.values()) | set(UNANSWERABLE))
    rows = []
    for q in qs:
        audio = voice.synthesize(q)
        for snr in snrs:
            heard_audio = audio if snr is None else add_noise(audio, snr)
            t0 = time.perf_counter()
            heard = voice.transcribe(heard_audio)
            folded = _fold(heard)
            kept = all(_fold(t) in folded for t in topics(q))
            rows.append({"question": q, "snr_db": snr, "heard": heard, "wer": wer(q, heard),
                         "topics_kept": kept, "stt_seconds": time.perf_counter() - t0})  # fmt: skip
    return rows


def answers(voice: Voice, runs: Path) -> list[dict]:
    said = {}
    for path in sorted(runs.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            say = json.loads(line)["say"]
            if any(values(say)):
                said[say] = None
    rows = []
    for say in said:
        raw = voice.transcribe(voice.synthesize(say))
        normed = voice.transcribe(voice.synthesize(speakable(say)))
        spoken = voice.say(say)
        rows.append({"say": say, "raw_heard": raw, "raw_ok": readback_ok(say, raw),
                     "speakable": speakable(say), "speakable_heard": normed,
                     "speakable_ok": readback_ok(say, normed), "final_ok": spoken.ok,
                     "attempts": spoken.attempts, **{f"{k}_seconds": v for k, v in spoken.seconds.items()}})  # fmt: skip  # noqa: E501
    return rows


def _pct(xs: list[bool]) -> str:
    return f"{sum(xs) / len(xs):.0%}" if xs else "–"


def report(q_rows: list[dict[str, Any]], a_rows: list[dict[str, Any]]) -> str:
    lines = ["### Questions (Piper -> Whisper small)", "",
             "| audio | word error rate | topics kept |", "|---|---|---|"]  # fmt: skip
    for snr in dict.fromkeys(r["snr_db"] for r in q_rows):
        rs = [r for r in q_rows if r["snr_db"] == snr]
        label = "clean" if snr is None else f"white noise, SNR {snr:g} dB"
        kept = _pct([r["topics_kept"] for r in rs])
        mean_wer = statistics.mean(r["wer"] for r in rs)
        lines.append(f"| {label} | {mean_wer:.1%} | {kept} ({len(rs)} questions) |")
    n = len(a_rows)
    lines += ["", f"### Answers with a date or amount ({n} distinct answers)", "",
              "| what is synthesised | user hears the verified values |", "|---|---|",
              f"| raw answer text | {_pct([r['raw_ok'] for r in a_rows])} |",
              f"| `speakable()` text | {_pct([r['speakable_ok'] for r in a_rows])} |",
              f"| `speakable()` + readback check (else fallback) | 100% by construction; "
              f"fallback on {_pct([not r['final_ok'] for r in a_rows])} |"]  # fmt: skip
    tts = sorted(r.get("tts_seconds", 0) for r in a_rows)
    back = sorted(r.get("readback_seconds", 0) for r in a_rows)
    stt = sorted(r["stt_seconds"] for r in q_rows)
    lines += ["", "Median seconds on the CPU: question transcription "
              f"{statistics.median(stt):.2f}, answer synthesis {statistics.median(tts):.2f}, "
              f"readback check {statistics.median(back):.2f}."]  # fmt: skip
    return "\n".join(lines)


def main(runs: Path, out: Path) -> str:
    voice = Voice()
    out.mkdir(parents=True, exist_ok=True)
    q_rows, a_rows = questions(voice), answers(voice, runs)
    for name, rows in (("questions", q_rows), ("answers", a_rows)):
        with (out / f"voice-{name}.jsonl").open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    text = report(q_rows, a_rows)
    (out / "voice-summary.md").write_text(text + "\n", encoding="utf-8")
    return text
