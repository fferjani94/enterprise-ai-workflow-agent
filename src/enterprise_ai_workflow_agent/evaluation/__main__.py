"""python -m enterprise_ai_workflow_agent.evaluation: run the offline benchmark and report."""

import argparse
import json
import logging
import sys
from pathlib import Path

from enterprise_ai_workflow_agent.evaluation.dataset import DEFAULT_BENCHMARK, load_benchmark
from enterprise_ai_workflow_agent.evaluation.harness import run_case_safely
from enterprise_ai_workflow_agent.evaluation.metrics import (
    build_report,
    render_markdown,
    score_case,
)

MODE = "offline"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m enterprise_ai_workflow_agent.evaluation",
        description="Run the deterministic offline benchmark (no Ollama or network needed).")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--out", type=Path, default=Path("evals") / "results",
                        help="directory for the JSON and Markdown reports")
    parser.add_argument("--case", action="append", default=[], metavar="ID",
                        help="run only this case (repeatable)")
    parser.add_argument("--verbose", action="store_true", help="show agent log messages")
    args = parser.parse_args(argv)
    benchmark = load_benchmark(args.benchmark)
    cases = [case for case in benchmark.cases if not args.case or case.id in args.case]
    unknown = sorted(set(args.case) - {case.id for case in cases})
    if unknown:
        parser.error(f"unknown case ID(s): {', '.join(unknown)}")

    previous = logging.root.manager.disable
    if not args.verbose:
        # Withheld answers and scripted model errors are logged; expected in this benchmark.
        logging.disable(logging.CRITICAL)
    try:
        scores = [score_case(case, run_case_safely(case)) for case in cases]
    finally:
        logging.disable(previous)
    report = build_report(benchmark, scores, MODE)
    markdown = render_markdown(report)

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / f"{benchmark.name}-{MODE}"
    if args.case:
        # Subset runs never overwrite the full-suite report.
        stem = stem.with_name(f"{stem.name}-subset-" + "-".join(c.id for c in cases))
    stem.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    stem.with_suffix(".md").write_text(markdown, encoding="utf-8")
    sys.stdout.write(markdown)
    for score in scores:
        if not score.passed:
            sys.stderr.write(score.explain() + "\n")
    sys.stdout.write(f"\nReports written to {stem}.json and {stem}.md\n")
    return 0 if all(score.passed for score in scores) else 1


if __name__ == "__main__":
    sys.exit(main())
