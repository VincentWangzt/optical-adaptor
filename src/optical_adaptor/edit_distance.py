from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Hashable, Sequence
from pathlib import Path
from typing import Any

from rapidfuzz.distance.Levenshtein import distance


def levenshtein_distance(reference: Sequence[Hashable], hypothesis: Sequence[Hashable]) -> int:
    """Exact unit-cost edit distance, accelerated for full-document evaluation."""
    return distance(reference, hypothesis, weights=(1, 1, 1), processor=None)


def evaluate_edit_distance(reference: str, hypothesis: str, *, unit: str) -> dict[str, Any]:
    if unit == "character":
        reference_units: Sequence[str] = reference
        hypothesis_units: Sequence[str] = hypothesis
    elif unit == "word":
        reference_units = reference.split()
        hypothesis_units = hypothesis.split()
    else:
        raise ValueError("unit must be 'character' or 'word'")

    distance = levenshtein_distance(reference_units, hypothesis_units)
    reference_count = len(reference_units)
    hypothesis_count = len(hypothesis_units)
    error_rate = distance / reference_count if reference_count else float(distance > 0)
    return {
        "unit": unit,
        "distance": distance,
        "reference_units": reference_count,
        "hypothesis_units": hypothesis_count,
        "error_rate": error_rate,
        "similarity": 1.0 - distance / max(reference_count, hypothesis_count, 1),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute exact Levenshtein distance between two UTF-8 text files."
    )
    parser.add_argument("reference", type=Path, help="ground-truth text file")
    parser.add_argument("hypothesis", type=Path, help="OCR/generated text file")
    parser.add_argument(
        "--unit",
        choices=("character", "word"),
        default="character",
        help="comparison unit (default: character)",
    )
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        reference = args.reference.expanduser().read_text(encoding="utf-8")
        hypothesis = args.hypothesis.expanduser().read_text(encoding="utf-8")
    except OSError as exc:
        print(f"ocr-edit-distance: error: {exc}", file=sys.stderr)
        return 2

    result = evaluate_edit_distance(reference, hypothesis, unit=args.unit)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Edit distance ({args.unit}): {result['distance']}")
        print(f"Reference units: {result['reference_units']}")
        print(f"Hypothesis units: {result['hypothesis_units']}")
        print(f"Error rate: {result['error_rate']:.6f}")
        print(f"Similarity: {result['similarity']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
