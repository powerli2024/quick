#!/usr/bin/env python3
"""Run SenseVoiceSmall once per canonical audio/wake/language key.

The quick pipeline passes a manifest containing one row per candidate.  This
adapter deliberately deduplicates before inference and writes one JSONL row per
``score_key``; that prevents duplicate UID aliases from being treated as
conflicting ASR results by the scorer.
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


def _clean(text: object) -> str:
    value = str(text or "")
    try:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        return str(rich_transcription_postprocess(value)).strip()
    except Exception:
        # Keep the adapter usable with older FunASR builds.  SenseVoice tags
        # are metadata, not spoken words, and must not enter CER.
        import re

        value = re.sub(r"<[^>]*>", " ", value)
        return re.sub(r"\s+", " ", value).strip()


def _one(model, row: dict, device: str) -> dict:
    lang = str(row.get("lang") or "auto").lower()
    language = {"zh": "zh", "en": "en"}.get(lang, "auto")
    try:
        result = model.generate(
            input=str(row["input"]),
            cache={},
            language=language,
            use_itn=False,
            batch_size_s=300,
            merge_vad=True,
            disable_pbar=True,
            disable_log=True,
        )
        item = result[0] if isinstance(result, list) and result else result
        if isinstance(item, dict):
            text = item.get("text") or item.get("sentence") or item.get("transcript")
            events = item.get("event") or item.get("events")
        else:
            text, events = item, None
        out = {
            "score_key": row.get("score_key"),
            "pcm_sha256": row.get("pcm_sha256"),
            "wake_text": row.get("wake_text"),
            "lang": row.get("lang"),
            "hyp": _clean(text),
            "asr_backend": "sensevoice_small",
            "event_tags": events,
            "device": device,
            "candidate_ids": row.get("candidate_ids", []),
        }
        return out
    except Exception as exc:  # preserve coverage; scorer will mark it bad
        return {
            "score_key": row.get("score_key"),
            "pcm_sha256": row.get("pcm_sha256"),
            "wake_text": row.get("wake_text"),
            "lang": row.get("lang"),
            "hyp": "",
            "asr_backend": "sensevoice_small",
            "error": f"{type(exc).__name__}: {exc}",
            "device": device,
            "candidate_ids": row.get("candidate_ids", []),
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    try:
        from funasr import AutoModel
    except Exception as exc:
        raise SystemExit("SenseVoice requires FunASR/ModelScope; install requirements-a3.txt") from exc

    # Ordered deduplication keeps output deterministic and audit-friendly.
    unique: OrderedDict[str, dict] = OrderedDict()
    for row in _read(args.manifest):
        key = str(row.get("score_key") or "")
        if not key:
            key = "pcm:{}|{}|{}".format(row.get("pcm_sha256"), row.get("wake_text"), row.get("lang"))
        if key not in unique:
            unique[key] = dict(row, candidate_ids=[row.get("candidate_id")])
        else:
            unique[key].setdefault("candidate_ids", []).append(row.get("candidate_id"))

    model = AutoModel(
        model=str(args.model_dir),
        vad_model=None,
        device=args.device,
        disable_update=True,
        disable_pbar=True,
        disable_log=True,
        trust_remote_code=True,
    )
    total = len(unique)
    def results():
        for idx, row in enumerate(unique.values(), 1):
            yield _one(model, row, args.device)
            if idx == total or idx % 50 == 0:
                print(f"\r[A3][SenseVoice] {idx}/{total}", end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)
    _write(args.output, results())
    print(f"[A3][SenseVoice] unique={len(unique)} output={args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
