#!/usr/bin/env python3
"""Run the official WeNet Python recognizer on a quick manifest.

This is a thin data-contract bridge, not a second decoder: it creates the
Kaldi-style ``wav.scp``/``text`` files expected by WeNet and converts CTC
greedy output back to JSONL for ``score_wenet_ctc_manifest.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--wenet-repo", type=Path, default=Path("/root/wenet"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    config = next(iter(sorted(args.model_dir.rglob("train.yaml"))), None)
    checkpoint = next(iter(sorted(args.model_dir.rglob("final.pt"))), None)
    recognize = args.wenet_repo / "wenet" / "bin" / "recognize.py"
    if config is None or checkpoint is None or not recognize.is_file():
        raise SystemExit("WeNet requires model-dir/{train.yaml,final.pt} and wenet-repo/wenet/bin/recognize.py")

    unique = {}
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        lang = str(row.get("lang") or "").lower()
        if lang != "zh":
            continue
        sk = str(row.get("score_key") or row.get("candidate_id") or "")
        if sk and sk not in unique:
            unique[sk] = row
    if not unique:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("", encoding="utf-8")
        return 0

    with tempfile.TemporaryDirectory(prefix="quick_wenet_") as td:
        root = Path(td)
        data = root / "data"
        result = root / "result"
        data.mkdir()
        wav_lines, text_lines = [], []
        for sk, row in unique.items():
            wav = row.get("input") or row.get("source_wav")
            if not wav:
                continue
            wav_lines.append(f"{sk} {wav}")
            # Dataset requires a text field; recognition itself uses only the
            # features.  Use the reference phrase as a harmless placeholder.
            text_lines.append(f"{sk} {row.get('wake_text', '')}")
        (data / "wav.scp").write_text("\n".join(wav_lines) + "\n", encoding="utf-8")
        (data / "text").write_text("\n".join(text_lines) + "\n", encoding="utf-8")
        cuda = str(args.device).startswith("cuda")
        gpu = str(int(str(args.device).split(":", 1)[1])) if cuda and ":" in str(args.device) else "0"
        cmd = [
            sys.executable, str(recognize), "--config", str(config),
            "--test_data", str(data), "--data_type", "raw",
            "--checkpoint", str(checkpoint), "--modes", "ctc_greedy_search",
            "--result_dir", str(result), "--batch_size", str(args.batch_size),
            "--gpu", gpu if cuda else "-1", "--device", "cuda" if cuda else "cpu",
        ]
        proc = subprocess.run(
            cmd, check=False, cwd=str(args.wenet_repo),
            capture_output=True, text=True,
        )
        diagnostic = (proc.stdout or "") + "\n" + (proc.stderr or "")
        mismatch = bool(re.search(r"(?:missing tensor|unexpected tensor):", diagnostic))
        if proc.returncode != 0 or mismatch:
            reason = "checkpoint_architecture_mismatch" if mismatch else f"recognize_exit_{proc.returncode}"
            print(f"[A3][WeNet] unavailable: {reason}; CTC sidecar will be empty", file=sys.stderr)
            raise SystemExit(3)
        decoded = result / "ctc_greedy_search" / "text"
        if not decoded.is_file():
            raise SystemExit(f"WeNet did not create {decoded}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as out:
            for line in decoded.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    out.write(json.dumps({"score_key": parts[0], "hyp": parts[1]}, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
