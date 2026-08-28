from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .cer import detail
from .io import json_hash, read_jsonl, write_jsonl


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


_IDENTITY_FIELDS = frozenset({
    "candidate_id", "path", "wav", "source_wav", "input", "output", "role", "arm",
    "stream", "view", "uid", "split", "group_key", "dest_rel", "export_name", "slot",
    "is_selected", "score_key",
})
_BIND_FIELDS = ("pcm_sha256", "wake_text", "lang")
_FEATURE_COMPARE = (
    "hyp", "text", "transcript", "nll", "q_kw", "token_count", "score_kind",
    "embedding", "vector", "speaker_ref_score", "cos_to_ref",
    "p_music", "p_overlap", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl",
    "pcm_sha256", "audio_sha256", "wake_text", "lang",
)


def _pcm_of(row: dict[str, Any]) -> str | None:
    value = row.get("pcm_sha256") or row.get("canonical_id") or row.get("audio_sha256")
    return str(value) if value else None


def score_key(row: dict[str, Any]) -> str:
    return json_hash([_pcm_of(row), row.get("wake_text"), row.get("lang")])


def _lookup_keys(row: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    if row.get("candidate_id"):
        keys.append(f"id:{row['candidate_id']}")
    pcm = _pcm_of(row)
    wake, lang = row.get("wake_text"), row.get("lang")
    if pcm and wake is not None and lang:
        keys.append(f"asr:{json_hash([pcm, str(wake), str(lang)])}")
    elif pcm:
        keys.append(f"pcm:{pcm}")
    if row.get("score_key"):
        keys.append(f"score:{row['score_key']}")
    for field in ("path", "wav", "source_wav"):
        if row.get(field):
            keys.append(f"path:{row[field]}")
    return keys


def _compatible(left: dict[str, Any], right: dict[str, Any]) -> bool:
    for key in _FEATURE_COMPARE:
        if key in left and key in right and left[key] != right[key]:
            return False
    return True


def _assert_bind(stored: dict[str, Any], query: dict[str, Any]) -> None:
    s_pcm, q_pcm = _pcm_of(stored), _pcm_of(query)
    if s_pcm and q_pcm and s_pcm != q_pcm:
        raise ValueError(
            f"sidecar pcm mismatch stored={s_pcm} query={q_pcm} candidate={query.get('candidate_id')}"
        )
    for field in ("wake_text", "lang"):
        stored_v, query_v = stored.get(field), query.get(field)
        if stored_v not in (None, "") and query_v not in (None, "") and str(stored_v) != str(query_v):
            raise ValueError(
                f"sidecar {field} mismatch stored={stored_v!r} query={query_v!r} candidate={query.get('candidate_id')}"
            )


def load_sidecar(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    rows = read_jsonl(path)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        keys = _lookup_keys(row)
        if not keys:
            raise ValueError(f"sidecar row has no key: {path}")
        for key in keys:
            if key in out and not _compatible(out[key], row):
                raise ValueError(f"conflicting sidecar records for {key}: {path}")
            if key not in out:
                out[key] = row
    return out


def find_sidecar(side: dict[str, dict[str, Any]], row: dict[str, Any]) -> dict[str, Any] | None:
    found = None
    for key in _lookup_keys(row):
        if key in side:
            found = side[key]
            break
    if found is None:
        sk = row.get("score_key")
        if sk and f"score:{sk}" in side:
            found = side[f"score:{sk}"]
        elif sk and sk in side:
            found = side[sk]
    if found is None and row.get("candidate_id") and str(row["candidate_id"]) in side:
        found = side[str(row["candidate_id"])]
    if found is None:
        return None
    _assert_bind(found, row)
    return found


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


def classify_content(row: dict[str, Any]) -> str:
    q = row.get("audio_quality") or {}
    p_music = finite(q.get("p_music")) or 0.0
    p_overlap = finite(q.get("p_overlap")) or 0.0
    speech = finite(q.get("speech_ratio")) or 0.0
    cov = finite(row.get("wake_coverage")) or 0.0
    extra = finite(row.get("extra_ratio")) or 0.0
    hyp = str(row.get("hyp") or "").strip()
    if row.get("validity") == "fatal_invalid":
        return "uncertain_or_hallucination"
    if cov >= 0.8 and bool(row.get("core_hit")) and extra <= 0.35:
        return "target_plus_interference" if p_overlap >= 0.35 or extra > 0.1 else "target_wake"
    if p_music >= 0.65 and speech < 0.35:
        return "noise_or_music"
    if speech >= 0.35 and cov < 0.5 and hyp:
        return "competing_speech"
    if not hyp or (speech < 0.2 and p_music < 0.65):
        return "noise_or_music"
    return "uncertain_or_hallucination"


def _merge_quality(can_metrics: dict[str, Any] | None, noise: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(can_metrics or {})
    for key in (
        "p_music", "p_overlap", "noise_class", "music_class", "speaker_count",
        "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "srmr", "drr", "t60",
        "music_top_k", "instrument_top_k", "singing_top_k",
    ):
        if key in noise:
            metrics[key] = noise[key]
    return metrics


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
    feature_dir: str | Path | None = None,
    policy: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    asr = load_sidecar(asr_sidecar)
    nll = load_sidecar(nll_sidecar)
    qkw = load_sidecar(qkw_sidecar)
    embeds = load_sidecar(embedding_sidecar)
    noise = load_sidecar(noise_sidecar)

    audio_features: dict[str, dict[str, Any]] = {}
    asr_features: dict[str, dict[str, Any]] = {}
    cache_hit = 0
    cache_miss = 0
    missing_asr: list[str] = []

    for source in rows:
        pcm = source.get("canonical_id") or source.get("pcm_sha256")
        if not pcm or source.get("validity") == "fatal_invalid":
            continue
        if pcm not in audio_features:
            can = registry.get(pcm)
            nse = find_sidecar(noise, source) or find_sidecar(noise, {"pcm_sha256": pcm}) or {}
            e = find_sidecar(embeds, source) or find_sidecar(embeds, {"pcm_sha256": pcm})
            embedding = None
            ref_score = None
            if e:
                vector = e.get("embedding") or e.get("vector")
                embedding = vector if isinstance(vector, list) else None
                ref_score = finite(e.get("speaker_ref_score", e.get("cos_to_ref")))
            audio_features[pcm] = {
                "quality": _merge_quality(getattr(can, "metrics", None) if can is not None else None, nse),
                "embedding": embedding,
                "speaker_ref_score": ref_score,
            }
        sk = score_key({**source, "pcm_sha256": pcm})
        if sk in asr_features:
            cache_hit += 1
            continue
        lookup = dict(source)
        lookup["score_key"] = sk
        a = find_sidecar(asr, lookup)
        if a is None:
            missing_asr.append(source["candidate_id"])
            cache_miss += 1
            if not allow_missing_asr:
                asr_features[sk] = {"missing": True}
                continue
            a = {"hyp": "", "score_kind": "missing"}
        else:
            cache_hit += 1
        hyp = str(a.get("hyp") or a.get("text") or a.get("transcript") or "")
        text = detail(hyp, source["wake_text"], aliases)
        n = find_sidecar(nll, lookup) or a or {}
        q = find_sidecar(qkw, lookup) or {}
        qvalue = finite(q.get("q_kw"))
        nvalue = finite(n.get("nll", a.get("nll")))
        calibrated = bool(qvalue is not None and (qkw_calibrated or q.get("score_kind") == "calibrated_qkw"))
        asr_features[sk] = {
            **text,
            "nll": nvalue,
            "q_kw": qvalue,
            "qkw_calibrated": calibrated,
            "score_kind": "calibrated_qkw" if calibrated else "nll",
            "token_count": n.get("token_count", a.get("token_count")),
            "missing": False,
        }

    if missing_asr and not allow_missing_asr:
        raise RuntimeError(f"ASR sidecar missing {len(missing_asr)} candidate references; first={missing_asr[:5]}")

    scored: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        row["score_key"] = score_key(row) if row.get("pcm_sha256") or row.get("canonical_id") else None
        if row.get("validity") == "fatal_invalid" or not row.get("canonical_id"):
            row.setdefault("cer_route", None)
            row.setdefault("hyp", "")
            row["validity"] = "fatal_invalid"
            row["content_class"] = classify_content(row)
            row["audio_quality"] = {"fatal_invalid": True, "reason": row.get("fatal_reason")}
            row["delta_cer"] = None
            scored.append(row)
            continue
        pcm = row["canonical_id"]
        feats = audio_features[pcm]
        asr_row = asr_features[row["score_key"]]
        if asr_row.get("missing"):
            continue
        row.update({k: asr_row[k] for k in asr_row if k != "missing"})
        row["embedding"] = feats.get("embedding")
        row["speaker_ref_score"] = feats.get("speaker_ref_score")
        row["audio_quality"] = feats["quality"]
        row["delta_cer"] = None
        row["validity"] = "rankable"
        if feats["quality"].get("fatal_invalid") or finite(row.get("cer_route")) is None:
            row["validity"] = "fatal_invalid"
        elif row["lang"] == "en" and not row.get("core_hit"):
            row["validity"] = "semantic_ineligible"
        row["content_class"] = classify_content(row)
        if (
            row["validity"] == "rankable"
            and row["content_class"] == "uncertain_or_hallucination"
            and float(row["cer_route"]) > 0.5
        ):
            row["validity"] = "semantic_ineligible"
        scored.append(row)

    by_id = {r["candidate_id"]: r for r in scored}
    for row in scored:
        parent = by_id.get(row.get("raw_parent_candidate_id") or "")
        if parent is not None and finite(row.get("cer_route")) is not None and finite(parent.get("cer_route")) is not None:
            row["delta_cer"] = float(row["cer_route"]) - float(parent["cer_route"])
            row["cos_se_raw"] = cosine(row.get("embedding"), parent.get("embedding"))
            row["raw_parent_pcm_sha256"] = parent.get("pcm_sha256")
            cand_ref = finite(row.get("speaker_ref_score"))
            par_ref = finite(parent.get("speaker_ref_score"))
            row["gain_ref"] = None if cand_ref is None or par_ref is None else cand_ref - par_ref
        else:
            row.setdefault("cos_se_raw", None)
            row.setdefault("gain_ref", None)

    _assign_ranks(scored, policy)
    meta = {
        "n_rows": len(scored),
        "n_unique_pcm": len({r.get("canonical_id") for r in scored if r.get("canonical_id")}),
        "n_unique_score_key": len(asr_features),
        "n_missing_asr": len(missing_asr),
        "feature_cache_hit": cache_hit,
        "feature_cache_miss": cache_miss,
        "qkw_calibrated": qkw_calibrated,
        "n_candidate_refs": len(scored),
        "n_unique_raw_pcm": len({r["canonical_id"] for r in scored if r.get("view") == "raw" and r.get("canonical_id")}),
        "n_unique_se_pcm": len({r["canonical_id"] for r in scored if r.get("view") == "moss48k" and r.get("canonical_id")}),
    }
    if feature_dir is not None:
        path = Path(feature_dir) / "feature_registry.jsonl"
        records = []
        for pcm, feat in sorted(audio_features.items()):
            records.append({"kind": "audio", "pcm_sha256": pcm, "quality": feat["quality"], "has_embedding": feat.get("embedding") is not None})
        for sk, feat in sorted(asr_features.items()):
            records.append({"kind": "asr", "score_key": sk, **{k: feat[k] for k in feat if k != "missing"}})
        write_jsonl(path, records)
        meta["feature_registry"] = str(path)
    return scored, meta


def _assign_ranks(rows: list[dict[str, Any]], policy: Any = None) -> None:
    from .route import RoutePolicy, stage_winner

    policy = policy or RoutePolicy()
    by_group_role: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group_role[(str(row.get("group_key")), str(row.get("role")))].append(row)
    for (_, role), group_rows in by_group_role.items():
        rankable_rows = [r for r in group_rows if r.get("validity") == "rankable" and finite(r.get("cer_route")) is not None]
        remaining = list(rankable_rows)
        rank = 1
        while remaining:
            winner, _ = stage_winner(remaining, role, policy)
            if winner is None:
                break
            winner_row = next(r for r in remaining if r["candidate_id"] == winner["candidate_id"])
            winner_row["stage_rank"] = rank
            remaining = [r for r in remaining if r["candidate_id"] != winner["candidate_id"]]
            rank += 1
        cer_ordered = sorted(rankable_rows, key=lambda r: (float(r["cer_route"]), str(r["candidate_id"])))
        for i, row in enumerate(cer_ordered, 1):
            row["cer_rank"] = i
        for row in group_rows:
            row.setdefault("stage_rank", None)
            row.setdefault("cer_rank", None)
