#!/usr/bin/env python3
"""Score an ASR sidecar from a quick candidate manifest.

AutoDL usage (Qwen3-ASR via kws, model loaded once):

  python scripts/score_asr_manifest.py \\
    --manifest {manifest} --output {output} \\
    --model-dir /root/autodl-tmp/Qwen3-ASR-1.7B

Unique (pcm_sha256, wake_text, lang) rows are transcribed once.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def _ensure_kws() -> Path:
    here = Path(__file__).resolve()
    candidates = []
    if os.environ.get("KWS_SRC"):
        candidates.append(Path(os.environ["KWS_SRC"]))
    candidates.extend([
        Path("/root/kws/src"),
        here.parents[2] / "kws" / "src",
        here.parents[1].parent / "kws" / "src",
    ])
    for path in candidates:
        if path and (path / "kws" / "asr_transcribe.py").is_file():
            sys.path.insert(0, str(path))
            return path
    raise SystemExit("[ERR] kws src not found; set KWS_SRC=/root/kws/src (AutoDL)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qwen3-ASR sidecar for quick manifests")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, default=None)
    p.add_argument("--device", default=os.environ.get("DEVICE", "cuda:0"))
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--context-mode", choices=("wake", "none"), default="wake")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    kws_src = _ensure_kws()
    from kws.asr_transcribe import Qwen3ASRTranscriber  # noqa: E402
    from kws.audio import load_wav_mono  # noqa: E402

    model_dir = args.model_dir
    if model_dir is None:
        for key in ("ASR_MODEL_DIR", "QWEN3_ASR_DIR"):
            if os.environ.get(key):
                model_dir = Path(os.environ[key])
                break
    if model_dir is None or not Path(model_dir).is_dir():
        raise SystemExit("[ERR] pass --model-dir or set ASR_MODEL_DIR on AutoDL")

    rows = []
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    unique: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        if not row.get("input"):
            continue
        key = (str(row.get("pcm_sha256") or row.get("input")), str(row.get("wake_text") or ""), str(row.get("lang") or ""))
        if key not in unique:
            unique[key] = row

    asr = Qwen3ASRTranscriber(model_dir, device=args.device, dtype=args.dtype, max_batch_size=args.batch_size)
    hyp_by_key: dict[tuple[str, str, str], str] = {}
    pending = list(unique.values())
    done = 0
    by_wake: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in pending:
        by_wake[(str(row.get("wake_text") or ""), str(row.get("lang") or ""))].append(row)
    for (wake, lang), items in by_wake.items():
        for start in range(0, len(items), args.batch_size):
            chunk = items[start : start + args.batch_size]
            wavs = [load_wav_mono(Path(row["input"]))[0] for row in chunk]
            hyps = asr.transcribe_many(wavs, language=lang, wake_text=wake, context_mode=args.context_mode)
            for row, hyp in zip(chunk, hyps):
                key = (str(row.get("pcm_sha256") or row["input"]), wake, lang)
                hyp_by_key[key] = str(hyp or "")
            done += len(chunk)
            print(f"\r[ASR] {done}/{len(pending)} kws_src={kws_src}", end="", flush=True)
    print()

    out_rows = []
    for row in rows:
        if not row.get("input"):
            continue
        key = (str(row.get("pcm_sha256") or row["input"]), str(row.get("wake_text") or ""), str(row.get("lang") or ""))
        out_rows.append({
            "candidate_id": row.get("candidate_id"),
            "pcm_sha256": row.get("pcm_sha256"),
            "wake_text": row.get("wake_text"),
            "lang": row.get("lang"),
            "hyp": hyp_by_key[key],
            "score_key": row.get("score_key"),
            "path": row.get("input"),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[OK] wrote {args.output} n={len(out_rows)} unique={len(unique)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
