from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .cer import detail
from .io import read_jsonl


def _key(row: dict[str, Any]) -> list[str]:
    keys = []
    for key in ("candidate_id", "score_key", "pcm_sha256", "audio_sha256", "path", "wav"):
        value = row.get(key)
        if value is not None and str(value):
            keys.append(str(value))
    return keys


def load_sidecar(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    rows = read_jsonl(path)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        keys = _key(row)
        if not keys:
            raise ValueError(f"sidecar row has no key: {path}")
        for key in keys:
            if key in out and out[key] != row:
                raise ValueError(f"conflicting sidecar records for {key}: {path}")
            out[key] = row
    return out


def find_sidecar(side: dict[str, dict[str, Any]], row: dict[str, Any]) -> dict[str, Any] | None:
    for key in _key(row):
        if key in side:
            return side[key]
    return None


def _finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def score_rows(
    rows: list[dict[str, Any]],
    registry: dict[str, Any],
    *,
    asr_sidecar: str | Path | None,
    nll_sidecar: str | Path | None = None,
    qkw_sidecar: str | Path | None = None,
    embedding_sidecar: str | Path | None = None,
    noise_sidecar: str | Path | None = None,
    aliases: dict[str, list[str]] | None = None,
    allow_missing_asr: bool = False,
    qkw_calibrated: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    asr = load_sidecar(asr_sidecar)
    nll = load_sidecar(nll_sidecar)
    qkw = load_sidecar(qkw_sidecar)
    embeds = load_sidecar(embedding_sidecar)
    noise = load_sidecar(noise_sidecar)
    scored: list[dict[str, Any]] = []
    missing_asr = []
    for source in rows:
        row = dict(source)
        can = registry[row["canonical_id"]]
        a = find_sidecar(asr, row)
        if a is None:
            missing_asr.append(row["candidate_id"])
            if not allow_missing_asr:
                continue
            a = {"hyp": "", "score_kind": "missing"}
        hyp = str(a.get("hyp") or a.get("text") or a.get("transcript") or "")
        text = detail(hyp, row["wake_text"], aliases)
        row.update(text)
        n = find_sidecar(nll, row) or a or {}
        q = find_sidecar(qkw, row) or {}
        qvalue = _finite(q.get("q_kw"))
        nvalue = _finite(n.get("nll", a.get("nll")))
        row["nll"] = nvalue
        row["q_kw"] = qvalue
        row["qkw_calibrated"] = bool(qvalue is not None and (qkw_calibrated or q.get("score_kind") == "calibrated_qkw"))
        row["score_kind"] = "calibrated_qkw" if row["qkw_calibrated"] else "nll"
        e = find_sidecar(embeds, row)
        if e:
            vector = e.get("embedding") or e.get("vector")
            row["embedding"] = vector if isinstance(vector, list) else None
            row["speaker_ref_score"] = _finite(e.get("speaker_ref_score", e.get("cos_to_ref")))
        nse = find_sidecar(noise, row) or {}
        metrics = dict(can.metrics)
        for key in ("p_music", "p_overlap", "noise_class", "music_class", "speaker_count", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl"):
            if key in nse:
                metrics[key] = nse[key]
        row["audio_quality"] = metrics
        row["delta_cer"] = None
        row["validity"] = "rankable"
        if metrics.get("fatal_invalid") or not math.isfinite(float(row["cer_route"])):
            row["validity"] = "fatal_invalid"
        elif row["lang"] == "en" and not row["core_hit"]:
            row["validity"] = "semantic_ineligible"
        row["content_class"] = classify_content(row)
        if row["validity"] == "rankable" and row["content_class"] == "uncertain_or_hallucination" and row["cer_route"] > 0.5:
            row["validity"] = "semantic_ineligible"
        scored.append(row)
    if missing_asr and not allow_missing_asr:
        raise RuntimeError(f"ASR sidecar missing {len(missing_asr)} candidate references; first={missing_asr[:5]}")
    # Paired SE deltas are assigned after all rows are scored.
    by_id = {r["candidate_id"]: r for r in scored}
    for row in scored:
        parent = by_id.get(row.get("raw_parent_candidate_id"))
        if parent is not None:
            row["delta_cer"] = float(row["cer_route"]) - float(parent["cer_route"])
            row["cos_se_raw"] = cosine(row.get("embedding"), parent.get("embedding"))
    meta = {"n_rows": len(scored), "n_unique_pcm": len({r["canonical_id"] for r in scored}), "n_missing_asr": len(missing_asr), "qkw_calibrated": qkw_calibrated}
    return scored, meta


def classify_content(row: dict[str, Any]) -> str:
    q = row.get("audio_quality") or {}
    p_music = _finite(q.get("p_music")) or 0.0
    p_overlap = _finite(q.get("p_overlap")) or 0.0
    speech = _finite(q.get("speech_ratio")) or 0.0
    cov = _finite(row.get("wake_coverage")) or 0.0
    extra = _finite(row.get("extra_ratio")) or 0.0
    hyp = str(row.get("hyp") or "").strip()
    if cov >= 0.8 and bool(row.get("core_hit")) and extra <= 0.35:
        return "target_plus_interference" if p_overlap >= 0.35 or extra > 0.1 else "target_wake"
    if p_music >= 0.65 and speech < 0.35:
        return "noise_or_music"
    if speech >= 0.35 and cov < 0.5 and hyp:
        return "competing_speech"
    if not hyp or (speech < 0.2 and p_music < 0.65):
        return "noise_or_music"
    return "uncertain_or_hallucination"


def cosine(a: Any, b: Any) -> float | None:
    try:
        import numpy as np

        x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if x.ndim != 1 or y.ndim != 1 or x.size == 0 or x.shape != y.shape:
            return None
        nx, ny = np.linalg.norm(x), np.linalg.norm(y)
        if nx <= 1e-12 or ny <= 1e-12:
            return None
        return float(np.dot(x, y) / (nx * ny))
    except Exception:
        return None
