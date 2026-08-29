#!/usr/bin/env python3
"""Produce Q0 free and Q1 wake-context Qwen3-ASR sidecars.

The two modes have separate cache namespaces and output files.  This wrapper
is intentionally explicit so a context result can never be mistaken for the
free-transcription CER used by the route.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from score_asr_manifest import score_manifest


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--free-output", type=Path, required=True)
    p.add_argument("--context-output", type=Path, default=None)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--device", default=os.environ.get("DEVICE", "cuda:0"))
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--vocab-json", type=Path, default=None, help="deprecated; Q1 uses row wake_text only")
    p.add_argument("--duration-bucket-sec", type=float, default=0.5)
    args = p.parse_args()
    context_output = args.context_output or args.free_output.with_name("asr_q1_context.jsonl")
    common = {"model_dir": args.model_dir, "device": args.device, "dtype": args.dtype, "batch_size": args.batch_size, "cache_dir": args.cache_dir, "duration_bucket_sec": args.duration_bucket_sec}
    score_manifest(args.manifest, args.free_output, context_mode="none", **common)
    score_manifest(args.manifest, context_output, context_mode="wake", vocab_json=args.vocab_json, **common)
    print(f"[OK] Q0 free={args.free_output} Q1 context={context_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
