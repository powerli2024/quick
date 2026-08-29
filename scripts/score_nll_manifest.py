#!/usr/bin/env python3
"""Optional target-NLL sidecar. Reuses kws.qkw_nll on AutoDL."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _ensure_kws() -> None:
    here = Path(__file__).resolve()
    candidates = []
    if os.environ.get("KWS_SRC"):
        candidates.append(Path(os.environ["KWS_SRC"]))
    candidates.extend([Path("/root/kws/src"), here.parents[2] / "kws" / "src"])
    for path in candidates:
        if path and (path / "kws" / "qkw_nll.py").is_file():
            sys.path.insert(0, str(path))
            return
    raise SystemExit("[ERR] kws src not found; set KWS_SRC=/root/kws/src")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, default=None)
    p.add_argument("--device", default=os.environ.get("DEVICE", "cuda:0"))
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--batch-size", type=int, default=1)
    args = p.parse_args()
    _ensure_kws()
    from kws.audio import load_wav_mono, resampler_name
    from kws.qkw_nll import Qwen3ASRNLLScorer

    model_dir = args.model_dir or os.environ.get("ASR_MODEL_DIR")
    if not model_dir:
        raise SystemExit("[ERR] pass --model-dir or set ASR_MODEL_DIR")
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    unique = {}
    for row in rows:
        if not row.get("input"):
            continue
        key = (str(row.get("pcm_sha256") or row["input"]), str(row.get("wake_text") or ""), str(row.get("lang") or ""))
        unique.setdefault(key, row)
    scorer = Qwen3ASRNLLScorer(str(model_dir), device=args.device, dtype=args.dtype, max_batch_size=args.batch_size)
    pending = list(unique.values())
    nll_by_key = {}
    for start in range(0, len(pending), args.batch_size):
        chunk = pending[start : start + args.batch_size]
        results = scorer.score_batch(
            [load_wav_mono(Path(row["input"]))[0] for row in chunk],
            [str(row.get("wake_text") or "") for row in chunk],
            [str(row.get("lang") or "") for row in chunk],
        )
        for row, result in zip(chunk, results):
            key = (str(row.get("pcm_sha256") or row["input"]), str(row.get("wake_text") or ""), str(row.get("lang") or ""))
            nll_by_key[key] = (float(result.nll), int(result.token_count))
        print(f"\r[NLL] {min(start + len(chunk), len(pending))}/{len(pending)}", end="", flush=True)
    print()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            if not row.get("input"):
                continue
            key = (str(row.get("pcm_sha256") or row["input"]), str(row.get("wake_text") or ""), str(row.get("lang") or ""))
            nll, tokens = nll_by_key[key]
            handle.write(json.dumps({
                "candidate_id": row.get("candidate_id"), "pcm_sha256": row.get("pcm_sha256"),
                "nll": nll, "token_count": tokens, "score_kind": "nll",
                "sample_rate": 16000, "resampler": resampler_name(),
            }, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
