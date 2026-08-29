#!/usr/bin/env python3
"""Audit Q0/Q1 Qwen sidecars without rerunning ASR."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _rows(path: Path) -> dict[str, dict]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            key = str(row.get("score_key") or row.get("asr_key") or row.get("candidate_id") or "")
            if key:
                if key in out and out[key].get("hyp") != row.get("hyp"):
                    raise ValueError(f"duplicate key with different hyp: {key}")
                out[key] = row
    return out


def _norm(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]", "", str(value or "")).lower()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--free", type=Path, required=True)
    p.add_argument("--context", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    manifest = _rows(args.manifest)
    free, context = _rows(args.free), _rows(args.context)
    missing_free = sorted(set(manifest) - set(free))
    missing_context = sorted(set(manifest) - set(context))
    context_only = 0
    changed = 0
    for key in set(manifest) & set(free) & set(context):
        q0, q1 = _norm(free[key].get("hyp")), _norm(context[key].get("hyp"))
        if q0 != q1:
            changed += 1
        wake = _norm(manifest[key].get("wake_text"))
        if wake and wake in q1 and wake not in q0:
            context_only += 1
    result = {
        "schema": "qwen_asr_audit/v1",
        "n_manifest": len(manifest), "n_free": len(free), "n_context": len(context),
        "missing_free": missing_free[:100], "missing_context": missing_context[:100],
        "n_missing_free": len(missing_free), "n_missing_context": len(missing_context),
        "n_changed_q0_q1": changed, "n_context_only_target": context_only,
        "q0_is_complete": not missing_free, "q1_is_complete": not missing_context,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["q0_is_complete"] and result["q1_is_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
