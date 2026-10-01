"""Benchmark CLI: ``python -m brier.bench run --model <id> --task banking20 --out results/``."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from brier.backends.base import Backend
from brier.bench.run import check_levels, run_task
from brier.bench.tasks import TASKS, Task, load_task


def build_parser() -> argparse.ArgumentParser:
    """Argument parser for the benchmark CLI."""
    parser = argparse.ArgumentParser(prog="python -m brier.bench")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="evaluate correction levels on a task")
    run.add_argument("--model", required=True, help="Hugging Face model id")
    run.add_argument("--revision", default=None, help="model commit sha (recommended)")
    run.add_argument("--task", default="banking20", choices=TASKS)
    run.add_argument("--levels", default="raw,L0", help="comma-separated, e.g. raw,L0")
    run.add_argument("--out", required=True, help="output directory")
    run.add_argument("--limit", type=int, default=None, help="first N test/pool items only")
    run.add_argument("--seed", type=int, default=0, help="split seed")
    run.add_argument("--batch-size", type=int, default=8)
    run.add_argument("--dtype", default=None, choices=("float32", "bfloat16", "float16"))
    run.add_argument("--device", default=None)
    run.add_argument("--n-resamples", type=int, default=1000)
    return parser


def _load_task(name: str, seed: int) -> Task:
    return load_task(name, seed=seed)


def _make_backend(args: Any) -> Backend:
    from brier.backends.hf import HFBackend

    return HFBackend(
        args.model,
        revision=args.revision,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
    )


def _git(*args: str) -> str:
    here = Path(__file__).resolve().parent
    out = subprocess.run(  # noqa: S603 - literal args from this module only, no shell
        ["git", *args],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
        cwd=here,
    )
    return out.stdout.strip()


def _git_commit() -> str | None:
    """HEAD of the brier source checkout, or None (e.g. installed inside another repo)."""
    try:
        _git("ls-files", "--error-unmatch", Path(__file__).name)  # this file is tracked here
        return _git("rev-parse", "HEAD") or None
    except (OSError, subprocess.SubprocessError):
        return None


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI; returns the process exit code."""
    args = build_parser().parse_args(argv)
    levels = check_levels([lv.strip() for lv in args.levels.split(",") if lv.strip()])
    task = _load_task(args.task, args.seed)
    backend = _make_backend(args)
    summary = run_task(
        backend,
        task,
        levels,
        Path(args.out),
        limit=args.limit,
        n_resamples=args.n_resamples,
        git_commit=_git_commit(),
    )
    for level, metrics in summary["levels"].items():
        acc, ece_, flip = metrics["accuracy"][0], metrics["ece"][0], metrics["flip_rate"][0]
        print(f"{level:>4}: accuracy={acc:.3f} ece={ece_:.3f} flip_rate={flip:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
