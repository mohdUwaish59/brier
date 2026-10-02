"""Command line: ``brier check <model-id>`` (also ``python -m brier check``)."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from brier.backends.base import Backend
from brier.check import check_backend, format_report


def build_parser() -> argparse.ArgumentParser:
    """Argument parser for the ``brier`` command."""
    parser = argparse.ArgumentParser(prog="brier")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="check whether a model works with brier")
    check.add_argument("model", help="Hugging Face model id")
    check.add_argument("--revision", default=None, help="model commit sha (recommended)")
    check.add_argument("--dtype", default=None, choices=("float32", "bfloat16", "float16"))
    check.add_argument("--device", default=None)
    check.add_argument("--batch-size", type=int, default=8)
    check.add_argument("--attn-implementation", default=None, help="e.g. eager (Gemma 3)")
    check.add_argument("--json", action="store_true", help="print the report as JSON")
    return parser


def _make_backend(args: Any) -> Backend:
    from brier.backends.hf import HFBackend

    return HFBackend(
        args.model,
        revision=args.revision,
        dtype=args.dtype,
        device=args.device,
        batch_size=args.batch_size,
        attn_implementation=args.attn_implementation,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI; returns the process exit code (1 if a required check fails)."""
    args = build_parser().parse_args(argv)
    try:
        backend = _make_backend(args)
    except Exception as exc:  # any load failure is the answer to "does it work"
        message = f"{type(exc).__name__}: {exc}"[:500]
        if args.json:
            print(json.dumps({"model": {"id": args.model}, "ok": False, "load_error": message}))
        else:
            print(f"brier check: {args.model}\n\n  FAIL  load  {message}\n\nResult: FAILED")
        return 1
    report = check_backend(backend)
    print(json.dumps(report.to_dict(), indent=2) if args.json else format_report(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
