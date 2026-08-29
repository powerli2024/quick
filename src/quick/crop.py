from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .audio import file_sha256, pcm_sha256, read_wav, write_wav


def _finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def crop_policy_hash(policy: dict[str, Any]) -> str:
    payload = json.dumps(policy, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def alignment_key(row: dict[str, Any]) -> str:
    payload = {
        "pcm_sha256": row.get("pcm_sha256"),
        "wake_text": row.get("wake_text"),
        "lang": row.get("lang"),
        "alias_set_hash": row.get("alias_set_hash"),
        "text_normalizer_hash": row.get("text_normalizer_hash"),
        "aligner": row.get("aligner"),
        "model_hash": row.get("model_hash"),
        "runtime_hash": row.get("runtime_hash"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def crop_key(alignment: dict[str, Any], policy: dict[str, Any], start_sample: int, end_sample: int, sr: int) -> str:
    payload = {
        "alignment_key": alignment.get("alignment_key") or alignment_key(alignment),
        "crop_policy_hash": crop_policy_hash(policy),
        "start_sample": int(start_sample),
        "end_sample": int(end_sample),
        "sample_rate": int(sr),
        "fade_ms": policy.get("fade_ms", 0),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def valid_occurrences(alignment: dict[str, Any], duration_sec: float) -> list[dict[str, Any]]:
    out = []
    for index, occurrence in enumerate(alignment.get("occurrences") or []):
        if not isinstance(occurrence, dict) or occurrence.get("valid") is False:
            continue
        start = _finite(occurrence.get("start_sec"))
        end = _finite(occurrence.get("end_sec"))
        if start is None or end is None or start < 0 or end <= start or end > duration_sec + 1e-6:
            continue
        out.append({**occurrence, "occurrence_id": occurrence.get("occurrence_id", index), "start_sec": start, "end_sec": end})
    return out


def build_crop_specs(alignment: dict[str, Any], policy: dict[str, Any], *, sample_rate: int, n_samples: int) -> list[dict[str, Any]]:
    duration = n_samples / max(sample_rate, 1)
    occurrences = valid_occurrences(alignment, duration)
    if not occurrences:
        return []
    if not policy.get("allow_multiple_occurrences", False) and len(occurrences) > 1:
        # Ambiguous text positions are deliberately not auto-published.
        return []
    occurrence = max(occurrences, key=lambda x: (_finite(x.get("target_score")) or 0.0, -float(x["end_sec"] - x["start_sec"])))
    specs = []
    for name, pads in (policy.get("pads_sec") or {}).items():
        if not isinstance(pads, (list, tuple)) or len(pads) != 2:
            continue
        left, right = float(pads[0]), float(pads[1])
        start = max(0, int(round((occurrence["start_sec"] - left) * sample_rate)))
        end = min(n_samples, int(round((occurrence["end_sec"] + right) * sample_rate)))
        if end <= start or (end - start) / max(sample_rate, 1) < float(policy.get("min_duration_sec", 0.0)):
            continue
        if policy.get("max_duration_sec") is not None and (end - start) / sample_rate > float(policy["max_duration_sec"]):
            continue
        specs.append({
            "view": name,
            "occurrence_id": occurrence.get("occurrence_id"),
            "core_start_sec": occurrence["start_sec"],
            "core_end_sec": occurrence["end_sec"],
            "start_sample": start,
            "end_sample": end,
            "boundary_clamped": start == 0 or end == n_samples,
            "flags": list(alignment.get("flags") or []),
        })
    return specs


def materialize_crop(
    source: Path,
    destination_root: Path,
    alignment: dict[str, Any],
    policy: dict[str, Any],
) -> list[dict[str, Any]]:
    wav, sr = read_wav(source)
    specs = build_crop_specs(alignment, policy, sample_rate=sr, n_samples=len(wav))
    if not specs:
        return []
    out = []
    source_pcm = pcm_sha256(wav, sr)
    for spec in specs:
        key = crop_key({**alignment, "pcm_sha256": source_pcm}, policy, spec["start_sample"], spec["end_sample"], sr)
        dest = destination_root / key[:2] / f"{key}.wav"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.is_file():
            data = np.asarray(wav[spec["start_sample"] : spec["end_sample"]], dtype=np.float32)
            fade_ms = float(policy.get("fade_ms") or 0.0)
            fade_n = min(int(round(sr * fade_ms / 1000.0)), len(data) // 2)
            if fade_n > 0:
                ramp = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
                data[:fade_n] *= ramp
                data[-fade_n:] *= ramp[::-1]
            write_wav(dest, data, sr)
        check, check_sr = read_wav(dest)
        out.append({
            "schema": "quick_crop_candidate/v1",
            "crop_key": key,
            "view": spec["view"],
            "source_pcm_sha256": source_pcm,
            "crop_pcm_sha256": pcm_sha256(check, check_sr),
            "source_file_sha256": file_sha256(source),
            "file_sha256": file_sha256(dest),
            "sample_rate": sr,
            "start_sample": spec["start_sample"],
            "end_sample": spec["end_sample"],
            "core_start_sec": spec["core_start_sec"],
            "core_end_sec": spec["core_end_sec"],
            "boundary_clamped": spec["boundary_clamped"],
            "path": str(dest.resolve()),
            "selected": False,
            "reason_codes": [],
        })
    return out
