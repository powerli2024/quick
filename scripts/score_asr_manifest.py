#!/usr/bin/env python3
"""Qwen3-ASR Q0/Q1 sidecar with persistent, provenance-bound caching."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _ensure_kws() -> Path:
    here = Path(__file__).resolve()
    candidates = [Path(os.environ["KWS_SRC"])] if os.environ.get("KWS_SRC") else []
    candidates.extend([Path("/root/kws/src"), here.parents[2] / "kws" / "src"])
    for path in candidates:
        if (path / "kws" / "asr_transcribe.py").is_file():
            sys.path.insert(0, str(path))
            return path
    raise SystemExit("[ERR] kws src not found; set KWS_SRC=/root/kws/src")


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _hash_payload(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _model_hash(model_dir: Path) -> str:
    try:
        from quick.signatures import hash_model_dir

        value = hash_model_dir(model_dir)
        if value:
            return value
    except Exception:
        pass
    return "unhashed-" + _hash_payload(str(model_dir.resolve()))[:32]


def _package_version(name: str) -> str:
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:
        return "unavailable"


def _cache_path(cache_dir: Path, mode: str, model_hash: str, runtime_hash: str) -> Path:
    return cache_dir / f"qwen3_{mode}_{model_hash[:16]}_{runtime_hash[:16]}.jsonl"


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in _read(path):
        key = str(row.get("asr_key") or "")
        if key and row.get("hyp") is not None:
            out[key] = row
    return out


def _write_cache(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def score_manifest(
    manifest: Path,
    output: Path,
    *,
    model_dir: Path,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    batch_size: int = 1,
    context_mode: str = "none",
    cache_dir: Path | None = None,
    vocab_json: Path | None = None,
    duration_bucket_sec: float = 0.5,
) -> dict[str, Any]:
    kws_src = _ensure_kws()
    from kws.asr_transcribe import Qwen3ASRTranscriber
    # ``resampler_name`` was added to newer kws builds.  Keep the sidecar
    # runnable against an older /root/kws checkout: the actual samples still
    # go through the same ``load_wav_mono`` implementation, while the cache
    # provenance records a conservative compatibility label.
    from kws.audio import load_wav_mono
    try:
        from kws.audio import resampler_name as _kws_resampler_name
    except ImportError:
        def _kws_resampler_name() -> str:
            return "kws_audio_legacy"

    mode = str(context_mode).strip().lower()
    if mode not in {"none", "wake"}:
        raise ValueError(f"context_mode must be none or wake, got {context_mode!r}")
    rows = _read(manifest)
    if vocab_json is not None:
        raise ValueError("Qwen Q1 custom vocabulary is temporarily disabled; use each row's wake_text as context")

    def context_for(row: dict[str, Any], mode: str) -> str:
        if mode == "none":
            return ""
        # Temporary baseline: Q1 context is exactly the sample's wake_text.
        # Do not inject a language-wide vocabulary list until its leakage and
        # hallucination effects have been measured separately.
        return str(row.get("wake_text") or "").strip()

    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        if not row.get("input"):
            continue
        key = (str(row.get("pcm_sha256") or row["input"]), str(row.get("wake_text") or ""), str(row.get("lang") or ""))
        unique.setdefault(key, row)

    model_hash = _model_hash(model_dir)
    if duration_bucket_sec <= 0:
        raise ValueError("duration_bucket_sec must be positive")
    runtime_probe = {"adapter": "kws.Qwen3ASRTranscriber", "device": device, "dtype": dtype, "sample_rate": 16000, "resampler": _kws_resampler_name(), "mode": mode, "duration_bucket_sec": float(duration_bucket_sec), "qwen_asr_version": _package_version("qwen_asr"), "transformers_version": _package_version("transformers")}
    runtime_hash = _hash_payload(runtime_probe)
    cache_path = _cache_path(cache_dir, mode, model_hash, runtime_hash) if cache_dir else None
    cache = _load_cache(cache_path) if cache_path else {}
    pending: list[tuple[str, dict[str, Any], str]] = []
    cached = 0
    for (pcm, wake, lang), row in unique.items():
        context_text = context_for(row, mode)
        asr_key = _hash_payload({"pcm": pcm, "wake_text": wake, "lang": lang, "context_mode": mode, "context_text": context_text, "model_hash": model_hash, "runtime_hash": runtime_hash})
        if asr_key not in cache:
            pending.append((asr_key, {**row, "context_text": context_text}, lang))
        else:
            cached += 1

    by_wake: dict[tuple[str, str], list[tuple[str, dict[str, Any], str, Any]]] = defaultdict(list)
    for item in pending:
        wav, _ = load_wav_mono(Path(item[1]["input"]), sr=16000)
        bucket = int(math.floor((len(wav) / 16000.0) / float(duration_bucket_sec)))
        by_wake[(str(item[1].get("context_text") or ""), item[2], bucket)].append((*item, wav))
    asr = None
    if pending:
        asr = Qwen3ASRTranscriber(model_dir, device=device, dtype=dtype, max_batch_size=max(1, int(batch_size)))
    done = 0
    step = max(1, int(batch_size))
    for (context_text, lang, bucket), items in sorted(by_wake.items(), key=lambda pair: pair[0]):
        for start in range(0, len(items), step):
            chunk = items[start : start + step]
            wavs = [item[3] for item in chunk]
            details = asr.transcribe_many_detailed(wavs, language=lang, wake_text=context_text, context_mode=mode)
            for (asr_key, row, _, _wav), detail in zip(chunk, details):
                cache[asr_key] = {"asr_key": asr_key, "hyp": str(detail.get("hyp") or ""), "model_hash": model_hash, "runtime_hash": runtime_hash, "context_mode": mode, "context_text": context_text, "context_hash": _hash_payload(context_text), "wake_text": row.get("wake_text"), "lang": lang, "pcm_sha256": row.get("pcm_sha256"), "input": row.get("input"), "sample_rate": 16000, "num_samples": detail.get("num_samples"), "duration_sec": detail.get("duration_sec"), "resampler": _kws_resampler_name()}
            done += len(chunk)
            print(f"\r[ASR:{mode}] {done}/{len(pending)} cache={cached} backend=qwen3", end="", flush=True)
    if pending:
        print()
    if cache_path:
        _write_cache(cache_path, list(cache.values()))

    out_rows = []
    for row in rows:
        if not row.get("input"):
            continue
        pcm, wake, lang = str(row.get("pcm_sha256") or row["input"]), str(row.get("wake_text") or ""), str(row.get("lang") or "")
        context_text = context_for(row, mode)
        asr_key = _hash_payload({"pcm": pcm, "wake_text": wake, "lang": lang, "context_mode": mode, "context_text": context_text, "model_hash": model_hash, "runtime_hash": runtime_hash})
        result = cache.get(asr_key)
        if result is None:
            raise RuntimeError(f"internal ASR cache miss after inference: {asr_key}")
        out_rows.append({"candidate_id": row.get("candidate_id"), "pcm_sha256": row.get("pcm_sha256"), "wake_text": row.get("wake_text"), "lang": row.get("lang"), "hyp": result.get("hyp", ""), "score_key": row.get("score_key"), "path": row.get("input"), "asr_key": asr_key, "context_mode": mode, "context_hash": result.get("context_hash", _hash_payload(context_text)), "model_hash": model_hash, "runtime_hash": runtime_hash, "sample_rate": result.get("sample_rate", 16000), "duration_sec": result.get("duration_sec"), "resampler": result.get("resampler", _kws_resampler_name())})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[OK] wrote {output} n={len(out_rows)} unique={len(unique)} cache_hit={cached} cache_miss={len(pending)}")
    return {"output": str(output), "n": len(out_rows), "unique": len(unique), "cache_hit": cached, "cache_miss": len(pending), "model_hash": model_hash, "runtime_hash": runtime_hash, "context_mode": mode, "kws_src": str(kws_src)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qwen3-ASR Q0/Q1 sidecar for quick manifests")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, default=None)
    p.add_argument("--device", default=os.environ.get("DEVICE", "cuda:0"))
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--context-mode", choices=("wake", "none"), default="none")
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--vocab-json", type=Path, default=None, help="deprecated: Q1 temporarily uses row wake_text only")
    p.add_argument("--duration-bucket-sec", type=float, default=0.5)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = args.model_dir or (Path(os.environ["ASR_MODEL_DIR"]) if os.environ.get("ASR_MODEL_DIR") else None)
    if model_dir is None or not model_dir.is_dir():
        raise SystemExit("[ERR] pass --model-dir or set ASR_MODEL_DIR")
    cache_dir = args.cache_dir or (Path(os.environ["ASR_CACHE_DIR"]) if os.environ.get("ASR_CACHE_DIR") else None)
    score_manifest(args.manifest, args.output, model_dir=model_dir, device=args.device, dtype=args.dtype, batch_size=args.batch_size, context_mode=args.context_mode, cache_dir=cache_dir, vocab_json=args.vocab_json, duration_bucket_sec=args.duration_bucket_sec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
