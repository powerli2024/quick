#!/usr/bin/env python3
"""Export a deterministic, shadow-only crop review package."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path


def _read(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _write(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _materialize(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True, help="score_crop_candidates.py output")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    groups = defaultdict(list)
    for row in _read(args.manifest):
        groups[(str(row.get("split") or "unknown"), str(row.get("uid") or "unknown"))].append(row)
    index = []
    for (split, uid), rows in sorted(groups.items()):
        rankable = [r for r in rows if r.get("crop_validity") == "rankable" and isinstance(r.get("cer_route"), (int, float))]
        selected = min(rankable, key=lambda r: (float(r["cer_route"]), str(r.get("candidate_id")))) if rankable else None
        folder = args.output_dir / split
        if selected and selected.get("source_wav"):
            _materialize(Path(selected["source_wav"]), folder / f"{uid}__00_SELECTED__{selected.get('view') or 'crop'}.wav")
        reason = {
            "schema": "quick_crop_review_reason/v1", "uid": uid, "split": split,
            "status": "selected_crop" if selected else "CROP_REJECT_KEEP_FULL",
            "selected_candidate_id": selected.get("candidate_id") if selected else None,
            "candidates": [
                {k: r.get(k) for k in ("candidate_id", "view", "cer_route", "hyp", "q_kw", "nll", "path", "crop_key", "crop_reason")}
                for r in sorted(rows, key=lambda x: str(x.get("candidate_id")))
            ],
        }
        reason_path = folder / f"{uid}__crop_reason.json"
        _write(reason_path, reason)
        index.append({"uid": uid, "split": split, "selected": selected.get("candidate_id") if selected else None, "reason": str(reason_path.relative_to(args.output_dir))})
    out_index = args.output_dir / "index.jsonl"
    out_index.parent.mkdir(parents=True, exist_ok=True)
    out_index.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in index), encoding="utf-8")
    print(json.dumps({"ok": True, "n_uid": len(index), "output": str(args.output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
