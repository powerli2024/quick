#!/usr/bin/env python3
"""Attach ASR/CER and optional q_kw fields to crop candidates."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from quick.cer import detail, load_alias_table  # noqa: E402


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--asr-jsonl", type=Path, required=True)
    ap.add_argument("--qkw-jsonl", type=Path, default=None)
    ap.add_argument("--alias-json", type=Path, default=None)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    aliases = load_alias_table(args.alias_json)

    def index(path: Path | None):
        out = {}
        if path is None:
            return out
        for row in _read(path):
            for key in (row.get("score_key"), row.get("candidate_id"), row.get("crop_key")):
                if key:
                    out[str(key)] = row
        return out

    asr = index(args.asr_jsonl)
    qkw = index(args.qkw_jsonl)
    out = []
    for row in _read(args.manifest):
        keys = [row.get("score_key"), row.get("candidate_id"), row.get("crop_key")]
        a = next((asr.get(str(k)) for k in keys if k is not None and str(k) in asr), None)
        q = next((qkw.get(str(k)) for k in keys if k is not None and str(k) in qkw), None)
        hyp = str((a or {}).get("hyp") or (a or {}).get("text") or "")
        scored = dict(row)
        scored.update(detail(hyp, str(row.get("wake_text") or ""), aliases))
        scored["q_kw"] = (q or {}).get("q_kw")
        scored["qkw_calibrated"] = (q or {}).get("score_kind") == "calibrated_qkw"
        scored["nll"] = (q or {}).get("nll", (a or {}).get("nll"))
        scored["crop_validity"] = "rankable" if a is not None else "fatal_invalid"
        if a is None:
            scored["crop_reason"] = "CROP_REJECT_KEEP_FULL"
        out.append(scored)
    _write(args.output, out)
    print(json.dumps({"ok": True, "n": len(out), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
