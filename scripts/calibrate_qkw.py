#!/usr/bin/env python3
"""Fit a frozen language-aware q_kw calibrator on an independent labeled set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quick.io import read_jsonl, write_json  # noqa: E402
from quick.qkw import SCHEMA, calibrator_hash, fit_logistic  # noqa: E402
from quick.signatures import file_sha256  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="fit independent dev-set q_kw calibration")
    parser.add_argument("--input", "--labeled-jsonl", dest="input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-field", default="ctc_align_score")
    parser.add_argument("--label-field", default="label")
    parser.add_argument("--lang-field", default="lang")
    parser.add_argument("--lower-is-better", action="store_true", help="use for NLL; calibration input becomes -score")
    parser.add_argument("--min-per-class", type=int, default=20)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    grouped: dict[str, list[tuple[float, int]]] = {}
    for number, row in enumerate(rows, 1):
        try:
            score = float(row[args.score_field])
            label = int(row[args.label_field])
            lang = str(row[args.lang_field]).strip().lower()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid calibration row {number}: {exc}") from exc
        if lang not in {"zh", "en"} or label not in {0, 1}:
            raise ValueError(f"invalid calibration lang/label at row {number}: lang={lang!r} label={label!r}")
        grouped.setdefault(lang, []).append((-score if args.lower_is_better else score, label))

    models = {}
    for lang, values in sorted(grouped.items()):
        positive = sum(label == 1 for _, label in values)
        negative = sum(label == 0 for _, label in values)
        if min(positive, negative) < args.min_per_class:
            raise ValueError(
                f"lang={lang} needs >= {args.min_per_class} samples per class; positive={positive} negative={negative}"
            )
        models[lang] = fit_logistic([x for x, _ in values], [y for _, y in values])
    if not models:
        raise ValueError("empty q_kw calibration set")
    payload = {
        "schema": SCHEMA,
        "score_field": args.score_field,
        "score_direction": "lower_is_better" if args.lower_is_better else "higher_is_better",
        "label_contract": "1=target_wake_present,0=target_wake_absent",
        "source_sha256": file_sha256(args.input),
        "independent_dev_required": True,
        "models": models,
    }
    payload["calibrator_hash"] = calibrator_hash(payload)
    write_json(args.output, payload)
    print(json.dumps({"ok": True, "output": str(args.output), "calibrator_hash": payload["calibrator_hash"], "models": models}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
