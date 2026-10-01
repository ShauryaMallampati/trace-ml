"""Command-line interface for TRACE-ML."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from trace_ml import __version__
from trace_ml.verification.verify_claim import verify


def _reject_nonstandard_number(value: str):
    raise ValueError(f"non-standard JSON numeric constant {value}")


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle, parse_constant=_reject_nonstandard_number)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read valid JSON from {path}: {exc}") from exc


def _runs_payload(value: Any) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and isinstance(value.get("runs"), list):
        return value["runs"]
    raise ValueError("runs JSON must be a list or an object containing a 'runs' list")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trace-ml",
        description="Verify a structured ML metric claim against supplied run records.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"TRACE-ML {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    verify_parser = subparsers.add_parser(
        "verify",
        help="verify one claim against a run ledger",
    )
    verify_parser.add_argument(
        "--claim",
        required=True,
        type=Path,
        help="JSON file containing one claim object",
    )
    verify_parser.add_argument(
        "--runs",
        required=True,
        type=Path,
        help="JSON file containing a run list",
    )
    verify_parser.add_argument("--output", type=Path, help="optional JSON output path")
    verify_parser.add_argument(
        "--require-supported",
        action="store_true",
        help="exit 1 for a violation or 2 for insufficient evidence",
    )
    verify_parser.add_argument(
        "--compact",
        action="store_true",
        help="emit compact JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        claim = _read_json(args.claim)
        if not isinstance(claim, dict):
            raise ValueError("claim JSON must contain one object")
        runs = _runs_payload(_read_json(args.runs))
        result = verify(claim, runs)
    except ValueError as exc:
        parser.error(str(exc))

    rendered = json.dumps(
        result,
        indent=None if args.compact else 2,
        sort_keys=True,
        allow_nan=False,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)

    if args.require_supported:
        if result["verdict"] == "supported":
            return 0
        if result["verdict"] == "violation":
            return 1
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
