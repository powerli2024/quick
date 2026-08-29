#!/usr/bin/env python3
"""Materialize bounded crop candidates from an alignment sidecar.

This stage never overwrites full audio and never invents an alignment.  Missing
or ambiguous alignment simply produces a full-audio fallback reason.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quick.crop import alignment_key, materialize_crop


def _read(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _write(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _lookup(row: dict, alignments: dict[str, dict]) -> dict | None:
    keys = [row.get("alignment_key"), row.get("score_key")]
    keys.append("pcm:{}|{}|{}".format(row.get("pcm_sha256"), row.get("wake_text"), row.get("lang")))
    for key in keys:
        if key and str(key) in alignments:
            return alignments[str(key)]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--alignment", type=Path, required=True)
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--audio-cache-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--reasons", type=Path, default=None)
    args = ap.parse_args()

    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    alignments: dict[str, dict] = {}
    for row in _read(args.alignment):
        if row.get("alignment_key"):
            alignments[str(row["alignment_key"])] = row
        if row.get("score_key"):
            alignments.setdefault(str(row["score_key"]), row)
        fallback = "pcm:{}|{}|{}".format(row.get("pcm_sha256"), row.get("wake_text"), row.get("lang"))
        alignments.setdefault(fallback, row)

    candidates = []
    reasons = []
    materialized: dict[tuple[str, str], list[dict]] = {}
    for row in _read(args.manifest):
        source = row.get("source_wav") or row.get("input") or row.get("path")
        alignment = _lookup(row, alignments)
        if not source or alignment is None:
            reasons.append({
                "schema": "quick_crop_reason/v1", "candidate_id": row.get("candidate_id"),
                "uid": row.get("uid"), "split": row.get("split"),
                "reason_code": "CROP_REJECT_KEEP_FULL", "detail": "missing_source_or_alignment",
            })
            continue
        cache_key = (str(row.get("pcm_sha256") or source), str(alignment.get("alignment_key") or alignment_key(alignment)))
        if cache_key not in materialized:
            materialized[cache_key] = materialize_crop(Path(source), args.audio_cache_root / "crop" / "v1", alignment, policy)
        specs = materialized[cache_key]
        if not specs:
            reasons.append({
                "schema": "quick_crop_reason/v1", "candidate_id": row.get("candidate_id"),
                "uid": row.get("uid"), "split": row.get("split"),
                "reason_code": "CROP_REJECT_KEEP_FULL", "detail": "invalid_or_ambiguous_alignment",
            })
            continue
        for spec in specs:
            candidates.append({
                "schema": "quick_crop_candidate/v1",
                "candidate_id": f"{row.get('candidate_id')}-crop-{spec['view']}",
                "group_key": row.get("group_key"), "uid": row.get("uid"), "split": row.get("split"),
                "wake_text": row.get("wake_text"), "lang": row.get("lang"),
                "role": row.get("role"), "arm": row.get("arm"), "stream": row.get("stream"),
                "parent_candidate_id": row.get("candidate_id"), "source_wav": spec["path"],
                "view": spec["view"], "crop_key": spec["crop_key"],
                "source_pcm_sha256": spec["source_pcm_sha256"], "pcm_sha256": spec["crop_pcm_sha256"],
                "start_sample": spec["start_sample"], "end_sample": spec["end_sample"],
                "core_start_sec": spec["core_start_sec"], "core_end_sec": spec["core_end_sec"],
                "alignment_key": cache_key[1], "validity": "pending", "selected": False,
                "reason_codes": [],
            })

    _write(args.output, candidates)
    _write(args.reasons or args.output.with_name("crop_reasons.jsonl"), reasons)
    print(json.dumps({
        "ok": True, "n_input": sum(1 for _ in _read(args.manifest)),
        "n_crop_candidates": len(candidates), "n_rejected_keep_full": len(reasons),
        "output": str(args.output), "reasons": str(args.reasons or args.output.with_name("crop_reasons.jsonl")),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
