"""Command line: ``sprechstunde chat`` and ``sprechstunde eval``."""

from __future__ import annotations

import argparse
from pathlib import Path

from sprechstunde.agent import Agent, Reminder
from sprechstunde.mailbox import Letter, Mailbox


def _backend(name: str):
    from sprechstunde.llm import LlamaCppBackend

    return LlamaCppBackend(name)


def _relevance(on: bool):
    if not on:
        return None
    from sprechstunde.relevance import NLITopicCheck

    return NLITopicCheck()


def _cmd_chat(args: argparse.Namespace) -> int:
    paths = sorted(Path(args.letters).glob("*.txt"))
    if not paths:
        print(f"no .txt letters in {args.letters}")
        return 1
    mailbox = Mailbox([Letter(p.stem, p.read_text(encoding="utf-8")) for p in paths])

    def confirm(r: Reminder) -> bool:
        answer = input(f'  Erinnerung "{r.title}" am {r.date} anlegen? [j/n] ')
        return answer.strip().lower() in ("j", "ja", "y", "yes")

    agent = Agent(
        _backend(args.model),
        mailbox,
        escalate_to=_backend(args.escalate) if args.escalate else None,
        confirm=confirm,
        relevance=_relevance(args.relevance),
    )
    print(f"{len(paths)} Briefe geladen. Frage stellen (leer = Ende).")
    while question := input("> ").strip():
        r = agent.ask(question)
        print(r.say)
        if args.trace:
            for s in r.steps:
                print(f"    [{s.model} {s.seconds:.1f}s] {s.action}")
    return 0


def _load_mailbox(folder: str) -> Mailbox | None:
    paths = sorted(Path(folder).glob("*.txt"))
    if not paths:
        print(f"no .txt letters in {folder}")
        return None
    return Mailbox([Letter(p.stem, p.read_text(encoding="utf-8")) for p in paths])


def _cmd_ask(args: argparse.Namespace) -> int:
    """One spoken turn: question WAV in, answer WAV out (no microphone needed)."""
    from sprechstunde.voice import Voice

    mailbox = _load_mailbox(args.letters)
    if mailbox is None:
        return 1
    voice = Voice()
    question = voice.transcribe(args.audio)
    print(f"heard:  {question}")
    agent = Agent(_backend(args.model), mailbox,
                  escalate_to=_backend(args.escalate) if args.escalate else None,
                  relevance=_relevance(args.relevance))  # fmt: skip
    r = agent.ask(question)
    spoken = voice.say(r.say)
    print(f"answer: {r.say}")
    print(f"spoken: {spoken.text}" + ("" if spoken.ok else "  (readback failed)"))
    Path(args.out).write_bytes(spoken.audio)
    print(f"wrote {args.out}")
    return 0


def _cmd_voice_eval(args: argparse.Namespace) -> int:
    from sprechstunde import voice_eval

    print(voice_eval.main(Path(args.runs), Path(args.out), args.n, agent=not args.no_agent))
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from sprechstunde import evaluate as ev

    out = Path(args.out)
    small, large = _backend(args.model), _backend(args.escalate)
    topic = _relevance(True)
    m, big = args.model, args.escalate

    def run(name, **kw):
        backend = kw.pop("backend", small)
        return ev.run(name, lambda mb: Agent(backend, mb, **kw), args.n, out, args.noise, args.seed)

    grounding = run(f"{m}+verify+{big}", escalate_to=large)
    full = run(f"{m}+verify+topic+{big}", escalate_to=large, relevance=topic)
    summaries = [
        ev.Summary.of(f"{m}, no checks", run(m, verify=False)),
        ev.Summary.of(f"{m} + grounding", ev.without_escalation(grounding, f"{m}+verify")),
        ev.Summary.of(f"{m} + grounding + topic", ev.without_escalation(full, f"{m}+topic")),
        ev.Summary.of(f"{m} + grounding + escalation to {big}", grounding),
        ev.Summary.of(f"**{m} + grounding + topic + escalation to {big}**", full),
        ev.Summary.of(f"{big}, no checks", run(big, verify=False, backend=large)),
    ]  # fmt: skip
    table = ev.to_markdown(summaries)
    print(table)
    (out / "summary.md").write_text(table + "\n", encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sprechstunde", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("chat", help="ask questions about the letters in a folder")
    c.add_argument("letters", help="folder with one .txt file per letter")
    c.add_argument("--model", default="qwen3-1.7b")
    c.add_argument("--escalate", default="qwen3-4b", help="larger model; '' to disable")
    c.add_argument("--trace", action="store_true", help="print every agent step")
    c.add_argument("--no-relevance", dest="relevance", action="store_false",
                   help="skip the NLI topic check (no torch needed)")  # fmt: skip
    c.set_defaults(func=_cmd_chat)

    a = sub.add_parser("ask", help="answer a spoken question (WAV) with a spoken answer (WAV)")
    a.add_argument("letters", help="folder with one .txt file per letter")
    a.add_argument("audio", help="question as a WAV file")
    a.add_argument("--out", default="answer.wav")
    a.add_argument("--model", default="qwen3-1.7b")
    a.add_argument("--escalate", default="qwen3-4b", help="larger model; '' to disable")
    a.add_argument("--no-relevance", dest="relevance", action="store_false")
    a.set_defaults(func=_cmd_ask)

    v = sub.add_parser("voice-eval", help="benchmark speech in/out and the readback check")
    v.add_argument("--runs", default="results/runs", help="agent outputs whose answers to speak")
    v.add_argument("--out", default="results/voice")
    v.add_argument("-n", type=int, default=12, help="mailboxes for the end-to-end agent run")
    v.add_argument("--no-agent", action="store_true", help="skip the end-to-end agent run")
    v.set_defaults(func=_cmd_voice_eval)

    e = sub.add_parser("eval", help="benchmark the agent on synthetic mailboxes")
    e.add_argument("-n", type=int, default=12, help="number of mailboxes (5 tasks each)")
    e.add_argument("--model", default="qwen3-1.7b")
    e.add_argument("--escalate", default="qwen3-4b")
    e.add_argument("--noise", type=float, default=0.03)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--out", default="results/runs")
    e.set_defaults(func=_cmd_eval)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
