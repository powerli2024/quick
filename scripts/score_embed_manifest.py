#!/usr/bin/env python3
"""Optional speaker-embedding sidecar. Reuses kws.eres on AutoDL."""

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
        if path and (path / "kws" / "eres.py").is_file():
            sys.path.insert(0, str(path))
            return
    raise SystemExit("[ERR] kws src not found; set KWS_SRC=/root/kws/src")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backend", default=os.environ.get("SPEAKER_BACKEND", "eres2netv2"))
    p.add_argument("--model-dir", type=Path, default=None)
    p.add_argument("--device", default=os.environ.get("DEVICE", "cuda:0"))
    args = p.parse_args()
    _ensure_kws()
    from kws.audio import load_wav_mono
    from kws.eres import load_embedder

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    unique = {}
    for row in rows:
        pcm = str(row.get("pcm_sha256") or "")
        if pcm and row.get("input"):
            unique.setdefault(pcm, row)
    encoder = load_embedder(args.backend, model_dir=args.model_dir, device=args.device)
    vectors = {}
    for i, (pcm, row) in enumerate(sorted(unique.items()), 1):
        wav, sr = load_wav_mono(Path(row["input"]))
        vectors[pcm] = encoder.embed(wav, sr).astype(float).tolist()
        print(f"\r[SPK] {i}/{len(unique)}", end="", flush=True)
    print()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            pcm = str(row.get("pcm_sha256") or "")
            if pcm not in vectors:
                continue
            handle.write(json.dumps({
                "candidate_id": row.get("candidate_id"),
                "pcm_sha256": pcm,
                "embedding": vectors[pcm],
            }, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
