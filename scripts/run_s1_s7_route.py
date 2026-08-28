#!/usr/bin/env python3
"""Strict I0-I8 s1/s7 + MossFormer SE route runner.

Model weights stay outside quick. Every result is tied to explicit hashes and a
reproducible policy. Missing external evidence never becomes a production pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quick.pipeline import run_from_args  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="strict I0-I8 s1/s7 MossFormer route")
    p.add_argument("--pos-neg", type=Path, required=True)
    p.add_argument("--s1-arm", required=True)
    p.add_argument("--s7-arm", required=True)
    p.add_argument("--expected-uids", type=int, default=0, help="combined pos+neg UID count; 0 disables")
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--asr-jsonl", type=Path, default=None)
    p.add_argument("--asr-command", default=None, help="template containing {manifest} and {output}")
    p.add_argument("--asr-model-dir", type=Path, default=None, help="AutoDL Qwen3-ASR dir; hashed into signatures")
    p.add_argument("--nll-jsonl", type=Path, default=None)
    p.add_argument("--nll-command", default=None, help="template containing {manifest} and {output}")
    p.add_argument("--embedding-command", default=None, help="template containing {manifest} and {output}")
    p.add_argument("--qkw-jsonl", type=Path, default=None)
    p.add_argument("--qkw-calibrated", action="store_true")
    p.add_argument("--embedding-jsonl", type=Path, default=None)
    p.add_argument("--noise-jsonl", type=Path, default=None)
    p.add_argument("--alias-json", type=Path, default=None)
    p.add_argument("--allow-missing-asr", action="store_true", help="audit-only; never production eligible")
    p.add_argument("--se-backend", choices=["command", "precomputed", "spectral", "none"], default="command")
    p.add_argument("--se-command", default=None)
    p.add_argument("--se-batch-command", default=None)
    p.add_argument("--precomputed-se-dir", type=Path, default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--qkw-low-thr", type=float, default=None)
    p.add_argument("--qkw-switch-margin", type=float, default=0.01)
    p.add_argument("--nll-switch-margin", type=float, default=0.01)
    p.add_argument("--speaker-switch-margin", type=float, default=0.01)
    p.add_argument("--cer-accept-thr", type=float, default=None)
    p.add_argument("--flat-dir", type=Path, default=None)
    p.add_argument("--selected-only-dir", type=Path, default=None)
    p.add_argument("--cmd-result-json", type=Path, default=None)
    p.add_argument("--presence-result-json", type=Path, default=None)
    p.add_argument("--contest-result-json", type=Path, default=None)
    p.add_argument("--cmd-command", default=None)
    p.add_argument("--presence-command", default=None)
    p.add_argument("--contest-command", default=None)
    p.add_argument("--cer-mean-max", type=float, default=0.03)
    p.add_argument("--cer0-drop-max", type=float, default=0.02)
    p.add_argument("--extract-sep-run-id", default=None)
    p.add_argument("--asr-model-hash", default=None)
    p.add_argument("--asr-context-mode", default=None)
    p.add_argument("--mossformer-model-hash", default=None)
    p.add_argument("--speaker-encoder-hash", default=None)
    p.add_argument("--qkw-calibrator-hash", default=None)
    p.add_argument("--inference-signature", default=None)
    return p.parse_args()


def main() -> int:
    report = run_from_args(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
