"""Command line: ``sprechstunde chat`` and ``sprechstunde eval``."""

from __future__ import annotations

import argparse
from pathlib import Path

from sprechstunde.agent import Agent, Reminder
from sprechstunde.mailbox import Letter, Mailbox


def _backend(name: str):
    from sprechstunde.llm import LlamaCppBackend

    return LlamaCppBackend(name)


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
    )
    print(f"{len(paths)} Briefe geladen. Frage stellen (leer = Ende).")
    while question := input("> ").strip():
        r = agent.ask(question)
        print(r.say)
        if args.trace:
            for s in r.steps:
                print(f"    [{s.model} {s.seconds:.1f}s] {s.action}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from sprechstunde import evaluate as ev

    out = Path(args.out)
    small, large = _backend(args.model), _backend(args.escalate)
    escalating = ev.run(
        f"{args.model}+verify+{args.escalate}",
        lambda mb: Agent(small, mb, escalate_to=large),
        args.n, out, args.noise, args.seed,
    )  # fmt: skip
    summaries = [
        ev.Summary.of(f"{args.model} (no verification)", ev.run(
            f"{args.model}", lambda mb: Agent(small, mb, verify=False),
            args.n, out, args.noise, args.seed)),
        ev.Summary.of(f"{args.model} + verification",
                      ev.without_escalation(escalating, f"{args.model}+verify")),
        ev.Summary.of(f"{args.model} + verification + escalation to {args.escalate}", escalating),
        ev.Summary.of(f"{args.escalate} (no verification)", ev.run(
            f"{args.escalate}", lambda mb: Agent(large, mb, verify=False),
            args.n, out, args.noise, args.seed)),
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
    c.set_defaults(func=_cmd_chat)

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
