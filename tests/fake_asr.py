#!/usr/bin/env python3
"""Deterministic ASR stub for tests. Hyp depends on candidate_id, not audio."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_CID = re.compile(r"C-(pos|neg)-(.+)-(s1|s7)-(raw|moss48k)-(.+)$")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = []
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cid = str(row.get("candidate_id") or "")
        wake = str(row.get("wake_text") or "hicolmo")
        hyp = wake
        m = _CID.match(cid)
        if m and m.group(2) == "pos_0002" and m.group(3) == "s1":
            hyp = "colmo"
        rows.append({
            "candidate_id": cid,
            "pcm_sha256": row.get("pcm_sha256"),
            "wake_text": wake,
            "lang": row.get("lang"),
            "hyp": hyp,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
