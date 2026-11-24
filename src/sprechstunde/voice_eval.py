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

import numpy as np

from sprechstunde.evaluate import QUESTIONS, REMINDERS, UNANSWERABLE
from sprechstunde.mailbox import _fold
from sprechstunde.relevance import topics
from sprechstunde.voice import NUMBER_PROMPT, Voice, readback_ok, speakable, values


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
            heard = voice.listen(heard_audio)
            folded = _fold(heard)
            kept = all(_fold(t) in folded for t in topics(q))
            rows.append({"question": q, "snr_db": snr, "heard": heard, "wer": wer(q, heard),
                         "topics_kept": kept, "stt_seconds": time.perf_counter() - t0})  # fmt: skip
    return rows


def answers(voice: Voice, runs: Path) -> list[dict]:
    """Every distinct spoken answer with a date or amount from the stored agent runs."""
    said = {}
    for path in sorted(runs.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            say = json.loads(line)["say"]
            if any(values(say)):
                said[say] = None
    rows = []
    for say in said:
        raw = voice.transcribe(voice.synthesize(say))
        spoken_audio = voice.synthesize(speakable(say))
        plain = voice.transcribe(spoken_audio)
        prompted = voice.transcribe(spoken_audio, prompt=NUMBER_PROMPT)
        final = voice.say(say)
        rows.append({"say": say, "raw_heard": raw, "speakable": speakable(say),
                     "speakable_heard": plain, "prompted_heard": prompted,
                     "raw_ok": values(say) == values(raw),
                     "speakable_ok": values(say) == values(plain),
                     "readback_ok": readback_ok(say, prompted), "final_ok": final.ok,
                     "attempts": final.attempts,
                     **{f"{k}_seconds": v for k, v in final.seconds.items()}})  # fmt: skip
    return rows


def spoken_agent(voice: Voice, n: int, runs: Path, model: str, escalate: str) -> list[dict]:
    """The full agent on the benchmark, with every question spoken by Piper and heard by
    Whisper (clean audio). Same tasks and scoring as ``sprechstunde eval``."""
    from sprechstunde.agent import Agent
    from sprechstunde.evaluate import run
    from sprechstunde.llm import LlamaCppBackend
    from sprechstunde.relevance import NLITopicCheck

    heard: dict[str, str] = {}

    def hear(question: str) -> str:
        if question not in heard:
            heard[question] = voice.listen(voice.synthesize(question))
        return heard[question]

    small, large, topic = LlamaCppBackend(model), LlamaCppBackend(escalate), NLITopicCheck()
    name = f"{model}+verify+topic+{escalate}+voice"
    make = lambda mb: Agent(small, mb, escalate_to=large, relevance=topic)  # noqa: E731
    return run(name, make, n, runs, hear=hear)


def _pct(xs: list[bool]) -> str:
    return f"{sum(xs) / len(xs):.0%}" if xs else "–"


def report(q_rows: list[dict], a_rows: list[dict], typed: list[dict], spoken: list[dict]) -> str:
    from sprechstunde.evaluate import Summary, to_markdown

    lines = ["### Questions: Piper -> Whisper large-v3-turbo with authority hotwords", "",
             "| audio | word error rate | topic words heard |", "|---|---|---|"]  # fmt: skip
    for snr in dict.fromkeys(r["snr_db"] for r in q_rows):
        rs = [r for r in q_rows if r["snr_db"] == snr]
        label = "clean" if snr is None else f"white noise, SNR {snr:g} dB"
        kept = _pct([r["topics_kept"] for r in rs])
        mean_wer = statistics.mean(r["wer"] for r in rs)
        lines.append(f"| {label} | {mean_wer:.1%} | {kept} ({len(rs)} questions) |")
    n = len(a_rows)
    sp_ok = _pct([r["speakable_ok"] for r in a_rows])
    lines += ["", f"### Answers: does the user hear the verified values? ({n} distinct answers)",
              "", "| synthesised from | heard as | values match |", "|---|---|---|",
              f"| raw answer text | plain Whisper | {_pct([r['raw_ok'] for r in a_rows])} |",
              f"| `speakable()` text | plain Whisper | {sp_ok} |",
              f"| `speakable()` text | number prompt + `repair_numbers` | "
              f"{_pct([r['readback_ok'] for r in a_rows])} |",
              f"| + one slower retry | | {_pct([r['final_ok'] for r in a_rows])}; the rest falls "
              "back to a safe sentence |"]  # fmt: skip
    if spoken:
        lines += ["", "### End to end: the full agent on spoken questions", "",
                  to_markdown([Summary.of("typed questions", typed),
                               Summary.of("spoken questions (clean audio)", spoken)])]  # fmt: skip
    tts = sorted(r.get("tts_seconds", 0) for r in a_rows)
    back = sorted(r.get("readback_seconds", 0) for r in a_rows)
    stt = sorted(r["stt_seconds"] for r in q_rows)
    lines += ["", "Median seconds on the CPU: question transcription "
              f"{statistics.median(stt):.1f}, answer synthesis {statistics.median(tts):.1f}, "
              f"readback check {statistics.median(back):.1f}."]  # fmt: skip
    return "\n".join(lines)


def main(runs: Path, out: Path, n: int = 12, model: str = "qwen3-1.7b",
         escalate: str = "qwen3-4b", agent: bool = True) -> str:  # fmt: skip
    voice = Voice()
    out.mkdir(parents=True, exist_ok=True)
    spoken = spoken_agent(voice, n, runs, model, escalate) if agent else []
    typed_path = runs / f"{model}+verify+topic+{escalate}.jsonl"
    typed = [json.loads(x) for x in typed_path.read_text(encoding="utf-8").splitlines()] if (
        typed_path.exists()) else []  # fmt: skip
    q_rows, a_rows = questions(voice), answers(voice, runs)
    for name, rows in (("questions", q_rows), ("answers", a_rows)):
        with (out / f"voice-{name}.jsonl").open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    text = report(q_rows, a_rows, typed, spoken)
    (out / "voice-summary.md").write_text(text + "\n", encoding="utf-8")
    return text
