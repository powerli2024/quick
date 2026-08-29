#!/usr/bin/env python3
"""Apply a frozen q_kw calibrator and stamp every output row with its hash."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quick.io import read_jsonl, write_jsonl  # noqa: E402
from quick.qkw import apply_calibrator, load_calibrator  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="apply frozen q_kw calibration")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--calibrator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload, digest = load_calibrator(args.calibrator)
    field = str(payload["score_field"])
    lower = payload.get("score_direction") == "lower_is_better"
    output = []
    for number, row in enumerate(read_jsonl(args.input), 1):
        try:
            raw = float(row[field])
            lang = str(row["lang"]).strip().lower()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid q_kw input row {number}: {exc}") from exc
        if not math.isfinite(raw):
            raise ValueError(f"non-finite q_kw source score at row {number}")
        score = -raw if lower else raw
        output.append({
            **row,
            "q_kw": apply_calibrator(payload, lang=lang, score=score),
            "score_kind": "calibrated_qkw",
            "qkw_calibrator_hash": digest,
        })
    write_jsonl(args.output, output)
    print(json.dumps({"ok": True, "n": len(output), "output": str(args.output), "calibrator_hash": digest}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
