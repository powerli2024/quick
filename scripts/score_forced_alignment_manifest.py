#!/usr/bin/env python3
"""Normalize forced-alignment output into ``quick_alignment/v1`` JSONL.

The actual aligner is intentionally external: Qwen3-ForcedAligner and MMS-FA
have different runtimes.  Pass an existing sidecar or an ``--align-command``
with ``{manifest}`` and ``{output}``; without either, this command fails closed.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path


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
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--alignment-jsonl", type=Path, default=None)
    ap.add_argument("--align-command", default=None)
    ap.add_argument("--qwen-aligner-dir", default=None)
    ap.add_argument("--mms-fallback", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    source = args.alignment_jsonl
    if source is None and args.align_command:
        rendered = str(args.align_command).replace("{manifest}", str(args.manifest)).replace("{output}", str(args.output))
        subprocess.run(shlex.split(rendered, posix=os.name != "nt"), check=True)
        source = args.output
    if source is None:
        raise SystemExit(
            "no aligner sidecar: provide --alignment-jsonl or --align-command; "
            "--qwen-aligner-dir alone does not invent timestamps"
        )
    if not source.is_file():
        raise SystemExit(f"alignment output not found: {source}")
    if source.resolve() != args.output.resolve():
        shutil.copyfile(source, args.output)

    # Validate the minimum contract while preserving additional aligner data.
    rows = list(_read(args.output))
    manifest_by_key = {}
    for row in _read(args.manifest):
        manifest_by_key[str(row.get("score_key") or row.get("candidate_id") or "")] = row
    normalized = []
    for row in rows:
        base = manifest_by_key.get(str(row.get("score_key") or row.get("candidate_id") or ""), {})
        normalized.append({
            **row,
            "schema": "quick_alignment/v1",
            "alignment_key": row.get("alignment_key"),
            "pcm_sha256": row.get("pcm_sha256", base.get("pcm_sha256")),
            "wake_text": row.get("wake_text", base.get("wake_text")),
            "lang": row.get("lang", base.get("lang")),
            "occurrences": row.get("occurrences") or [],
            "coverage": row.get("coverage"),
            "flags": row.get("flags") or [],
            "aligner": row.get("aligner") or ("qwen3_forced_aligner" if args.qwen_aligner_dir else "external"),
        })
    _write(args.output, normalized)
    print(json.dumps({"ok": True, "n_rows": len(normalized), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
