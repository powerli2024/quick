#!/usr/bin/env python3
"""Score a manifest with local/ModelScope Paraformer zh and en models.

The manifest is deduplicated by ``score_key`` before inference.  One model is
loaded per language and every output keeps the candidate-id aliases so the
quick scorer can join it without UID collisions.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
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


def _text(result) -> str:
    item = result[0] if isinstance(result, list) and result else result
    if isinstance(item, dict):
        value = item.get("text") or item.get("sentence") or item.get("transcript")
    else:
        value = item
    return str(value or "").strip()


def _one(model, row: dict, backend: str, device: str) -> dict:
    try:
        result = model.generate(
            input=str(row["input"]),
            cache={},
            batch_size_s=300,
            merge_vad=False,
            disable_pbar=True,
            disable_log=True,
        )
        return {
            "score_key": row.get("score_key"),
            "pcm_sha256": row.get("pcm_sha256"),
            "wake_text": row.get("wake_text"),
            "lang": row.get("lang"),
            "hyp": _text(result),
            "asr_backend": backend,
            "device": device,
            "candidate_ids": row.get("candidate_ids", []),
        }
    except Exception as exc:
        return {
            "score_key": row.get("score_key"),
            "pcm_sha256": row.get("pcm_sha256"),
            "wake_text": row.get("wake_text"),
            "lang": row.get("lang"),
            "hyp": "",
            "asr_backend": backend,
            "error": f"{type(exc).__name__}: {exc}",
            "device": device,
            "candidate_ids": row.get("candidate_ids", []),
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--zh-model-dir", default="paraformer-zh")
    ap.add_argument("--en-model-dir", default="paraformer-en")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    try:
        from funasr import AutoModel
    except Exception as exc:
        raise SystemExit("Paraformer requires FunASR/ModelScope; install requirements-a3.txt") from exc

    unique: OrderedDict[str, dict] = OrderedDict()
    for row in _read(args.manifest):
        key = str(row.get("score_key") or "")
        if not key:
            key = "pcm:{}|{}|{}".format(row.get("pcm_sha256"), row.get("wake_text"), row.get("lang"))
        if key not in unique:
            unique[key] = dict(row, candidate_ids=[row.get("candidate_id")])
        else:
            unique[key].setdefault("candidate_ids", []).append(row.get("candidate_id"))

    models = {}
    for lang, model_dir in (("zh", args.zh_model_dir), ("en", args.en_model_dir)):
        if not model_dir:
            continue
        models[lang] = AutoModel(
            model=str(model_dir),
            vad_model=None,
            punc_model=None,
            device=args.device,
            disable_update=True,
            disable_pbar=True,
            disable_log=True,
            trust_remote_code=True,
        )

    total = len(unique)
    out = []
    for idx, row in enumerate(unique.values(), 1):
        lang = str(row.get("lang") or "zh").lower()
        selected_lang = "en" if lang == "en" else "zh"
        model = models.get(selected_lang)
        if model is None:
            out.append({
                "score_key": row.get("score_key"), "pcm_sha256": row.get("pcm_sha256"),
                "wake_text": row.get("wake_text"), "lang": row.get("lang"), "hyp": "",
                "asr_backend": f"paraformer_{selected_lang}",
                "error": f"missing model for lang={selected_lang}",
                "device": args.device, "candidate_ids": row.get("candidate_ids", []),
            })
        else:
            out.append(_one(model, row, f"paraformer_{selected_lang}", args.device))
        if idx == total or idx % 50 == 0:
            print(f"\r[A3][Paraformer] {idx}/{total}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    _write(args.output, out)
    print(f"[A3][Paraformer] unique={total} output={args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
