"""Command line entry points: generate, analyze, ask, bench, serve, prefetch-embeddings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from clausewise.config import load_settings
from clausewise.logging_setup import configure_logging


def _cmd_generate(args: argparse.Namespace) -> int:
    from clausewise.corpus.generator import corpus_path, load_or_generate

    settings = load_settings()
    contracts = load_or_generate(settings.data_dir, settings.seed, args.size or settings.corpus_size)
    print(f"{len(contracts)} contracts cached at {corpus_path(settings.data_dir, settings.seed, len(contracts))}")
    if args.show:
        print(contracts[args.show].text)
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    from clausewise.pipeline import Analyzer

    text = Path(args.file).read_text(encoding="utf-8")
    result = Analyzer(load_settings()).analyze(text).result
    if args.json:
        print(result.model_dump_json(indent=2))
        return 0
    print(result.summary)
    for clause in result.clauses:
        print(f"  {clause.clause_id} {clause.label:<20} {clause.confidence:.2f}  {clause.heading or ''}")
    for finding in result.risks:
        print(f"RISK {finding.severity:<6} {finding.risk_type:<24} {finding.explanation}")
        print(f"     evidence: {finding.evidence}")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    from clausewise.pipeline import Analyzer

    text = Path(args.file).read_text(encoding="utf-8")
    answer = Analyzer(load_settings()).analyze_and_ask(text, args.question)
    print(answer.model_dump_json(indent=2) if args.json else _format_answer(answer))
    return 0


def _format_answer(answer: object) -> str:
    from clausewise.schemas import Answer

    assert isinstance(answer, Answer)
    if answer.abstained:
        return "No grounded answer found in the document."
    lines = [f"{s.text} [{', '.join(s.citations)}]" for s in answer.sentences]
    if answer.short_answer:
        lines.append(f"Short answer: {answer.short_answer}")
    return "\n".join(lines)


def _cmd_bench(args: argparse.Namespace) -> int:
    from clausewise.bench import main as bench_main

    argv = ["--out", str(args.out)]
    if args.check:
        argv += ["--check", str(args.check)]
    return bench_main(argv)


def _cmd_serve(args: argparse.Namespace) -> int:
    from clausewise.api.app import run

    run(host=args.host, port=args.port)
    return 0


def _cmd_prefetch(_args: argparse.Namespace) -> int:
    from clausewise.embeddings import embedder_name, load_embedder

    print(f"embeddings backend: {embedder_name(load_embedder(load_settings()))}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The argparse tree."""
    parser = argparse.ArgumentParser(prog="clausewise", description="Contract intelligence toolkit")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="generate and cache the seeded corpus")
    gen.add_argument("--size", type=int, default=None)
    gen.add_argument("--show", type=int, default=None, help="print contract number N")
    gen.set_defaults(func=_cmd_generate)
    an = sub.add_parser("analyze", help="analyze a contract text file")
    an.add_argument("file")
    an.add_argument("--json", action="store_true")
    an.set_defaults(func=_cmd_analyze)
    ask = sub.add_parser("ask", help="ask a grounded question about a contract text file")
    ask.add_argument("file")
    ask.add_argument("question")
    ask.add_argument("--json", action="store_true")
    ask.set_defaults(func=_cmd_ask)
    bench = sub.add_parser("bench", help="run the benchmark")
    bench.add_argument("--out", type=Path, default=Path("benchmarks"))
    bench.add_argument("--check", type=Path, default=None)
    bench.set_defaults(func=_cmd_bench)
    serve = sub.add_parser("serve", help="start the API and dashboard")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=_cmd_serve)
    pre = sub.add_parser("prefetch-embeddings", help="download the embedding model once (or report the fallback)")
    pre.set_defaults(func=_cmd_prefetch)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``clausewise`` console script."""
    args = build_parser().parse_args(argv)
    configure_logging(load_settings().log_level)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
