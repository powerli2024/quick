from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


STREAM_ORDER = {"original": 0, "spk1": 1, "spk2": 2, "spk3": 3, "spk4": 4}

REASON_CODES = (
    "KEEP_S1_CER0_TRUSTED",
    "KEEP_S1_S7_UNAVAILABLE",
    "KEEP_S1_S7_NO_RANKABLE",
    "KEEP_S1_LOWER_CER",
    "KEEP_S1_TIE_NO_GAIN",
    "SAME_AUDIO_KEEP_S1",
    "SWITCH_S7_LOWER_CER",
    "SWITCH_S7_EQUAL_CER_QKW_GAIN",
    "SWITCH_S7_EQUAL_CER_CTC_GAIN",
    "SWITCH_S7_EQUAL_CER_NLL_GAIN",
    "SWITCH_S7_EQUAL_CER_SPK_GAIN",
    "SWITCH_S7_S1_UNAVAILABLE",
    "AUDIT_FALLBACK_S1_RAW",
    "FAIL_NO_DECODABLE_S1",
)

REASON_TEXT = {
    "KEEP_S1_CER0_TRUSTED": "s1 为 CER0、语义可信，未触发 s7",
    "KEEP_S1_S7_UNAVAILABLE": "已触发，但冻结 s7 arm 无该 UID",
    "KEEP_S1_S7_NO_RANKABLE": "s7 没有可排名候选",
    "KEEP_S1_LOWER_CER": "s7 CER 更高，硬性保持 s1",
    "KEEP_S1_TIE_NO_GAIN": "CER 相同但 q_kw/NLL/独立声纹增益不足",
    "SAME_AUDIO_KEEP_S1": "s1、s7 winner 为同一 PCM，只保留 s1 决策身份",
    "SWITCH_S7_LOWER_CER": "s7 严格降低 CER",
    "SWITCH_S7_EQUAL_CER_QKW_GAIN": "同 CER 下冻结 q_kw 显著改善",
    "SWITCH_S7_EQUAL_CER_CTC_GAIN": "同 CER 下独立 CTC 已知词对齐分数显著改善",
    "SWITCH_S7_EQUAL_CER_NLL_GAIN": "未校准模式下，同 UID 同 CER 的 NLL 显著改善",
    "SWITCH_S7_EQUAL_CER_SPK_GAIN": "同 CER/文本证据下，独立声纹证据显著改善",
    "SWITCH_S7_S1_UNAVAILABLE": "s1 无 rankable 候选，使用 s7",
    "AUDIT_FALLBACK_S1_RAW": "无正常 winner，为审阅完整性回退到可解码 s1 raw",
    "FAIL_NO_DECODABLE_S1": "无法形成审阅 winner，整次导出失败",
}


@dataclass(frozen=True)
class RoutePolicy:
    qkw_low_thr: float | None = None
    qkw_switch_margin: float = 0.01
    nll_switch_margin: float = 0.01
    speaker_switch_margin: float = 0.01
    quality_regression_tolerance: float = 0.10
    cer_accept_thr: float | None = None
    clip_rate_thr: float = 0.01
    min_speech_ratio: float = 0.05


def _stream_key(value: Any) -> tuple[int, str]:
    stream = str(value or "")
    if stream in STREAM_ORDER:
        return STREAM_ORDER[stream], stream
    if stream.startswith("spk"):
        try:
            return int(stream[3:]) + 1, stream
        except ValueError:
            pass
    return 1000, stream


def candidate_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Deterministic identity key. Stage ranking must use stage_winner + policy margins."""
    cer = finite(row.get("cer_route"))
    return (
        float("inf") if cer is None else cer,
        0 if row.get("view") == "raw" else 1,
        0 if row.get("stream") == "original" else 1,
        _stream_key(row.get("stream")),
        str(row.get("candidate_id") or ""),
    )


def _conservative_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        0 if row.get("view") == "raw" else 1,
        0 if row.get("stream") == "original" else 1,
        _stream_key(row.get("stream")),
        str(row.get("candidate_id") or ""),
    )


def _within_margin(best: float, value: float, margin: float, *, higher_is_better: bool) -> bool:
    if higher_is_better:
        return (best - value) < margin - 1e-15 or abs(best - value) <= 1e-12
    return (value - best) < margin - 1e-15 or abs(best - value) <= 1e-12


def _pool_by_score(pool: list[dict[str, Any]], getter, margin: float, *, higher_is_better: bool) -> list[dict[str, Any]]:
    scored = [(row, getter(row)) for row in pool]
    values = [v for _, v in scored if v is not None]
    if len(values) != len(pool):
        return pool
    best = max(values) if higher_is_better else min(values)
    kept = [row for row, value in scored if value is not None and _within_margin(best, value, margin, higher_is_better=higher_is_better)]
    return kept or pool


def rankable(rows: Iterable[dict[str, Any]], role: str | None = None) -> list[dict[str, Any]]:
    return [
        dict(r) for r in rows
        if (role is None or r.get("role") == role)
        and r.get("validity") == "rankable"
        and finite(r.get("cer_route")) is not None
    ]


def _view_reason(winner: dict[str, Any], candidates: list[dict[str, Any]], policy: RoutePolicy) -> str:
    raw = [r for r in candidates if r.get("view") == "raw"]
    moss = [r for r in candidates if r.get("view") == "moss48k"]
    best_raw = min(raw, key=lambda r: stage_sort_tuple(r, policy)) if raw else None
    best_moss = min(moss, key=lambda r: stage_sort_tuple(r, policy)) if moss else None
    if winner.get("view") == "moss48k":
        if best_raw is None:
            return "MOSS_ONLY"
        c_m, c_r = float(winner["cer_route"]), float(best_raw["cer_route"])
        if c_m < c_r - 1e-12:
            return "MOSS_LOWER_CER"
        qgain = _qkw_gain(winner, best_raw)
        if qgain is not None and qgain >= policy.qkw_switch_margin:
            return "MOSS_EQUAL_CER_QKW_GAIN"
        kgain = _keyword_gain(winner, best_raw)
        if kgain is not None and kgain >= policy.qkw_switch_margin:
            return "MOSS_EQUAL_CER_CTC_GAIN"
        ngain = _nll_gain(winner, best_raw)
        if ngain is not None and ngain >= policy.nll_switch_margin:
            return "MOSS_EQUAL_CER_NLL_GAIN"
        return "MOSS_EQUAL_CER_OTHER"
    if best_moss is None:
        return "RAW_ONLY"
    c_w, c_m = float(winner["cer_route"]), float(best_moss["cer_route"])
    if c_w < c_m - 1e-12:
        return "RAW_LOWER_CER"
    return "RAW_CONSERVATIVE_TIE"


def stage_sort_tuple(row: dict[str, Any], policy: RoutePolicy) -> tuple[Any, ...]:
    cer = finite(row.get("cer_route"))
    return (float("inf") if cer is None else cer, *_conservative_key(row))


def stage_winner(
    rows: Iterable[dict[str, Any]],
    role: str,
    policy: RoutePolicy | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    policy = policy or RoutePolicy()
    candidates = rankable(rows, role)
    if not candidates:
        return None, {"role": role, "status": "no_rankable_candidate", "n_candidates": 0, "view_reason_code": None}
    min_cer = min(float(c["cer_route"]) for c in candidates)
    pool = [c for c in candidates if abs(float(c["cer_route"]) - min_cer) <= 1e-12]
    if all(c.get("qkw_calibrated") and finite(c.get("q_kw")) is not None for c in pool):
        pool = _pool_by_score(pool, lambda r: finite(r.get("q_kw")), policy.qkw_switch_margin, higher_is_better=True)
    elif all(finite(c.get("keyword_score")) is not None for c in pool):
        # A3's independent WeNet CTC score is not assumed calibrated across
        # recordings, but is a useful same-UID tie breaker.
        pool = _pool_by_score(pool, lambda r: finite(r.get("keyword_score")), policy.qkw_switch_margin, higher_is_better=True)
    elif all(finite(c.get("nll")) is not None for c in pool):
        pool = _pool_by_score(pool, lambda r: finite(r.get("nll")), policy.nll_switch_margin, higher_is_better=False)
    if all(finite(c.get("speaker_ref_score")) is not None for c in pool):
        pool = _pool_by_score(pool, lambda r: finite(r.get("speaker_ref_score")), policy.speaker_switch_margin, higher_is_better=True)
    if all(finite(c.get("extra_ratio")) is not None for c in pool):
        pool = _pool_by_score(pool, lambda r: finite(r.get("extra_ratio")), policy.quality_regression_tolerance, higher_is_better=False)
    overlaps = [finite((c.get("audio_quality") or {}).get("p_overlap")) for c in pool]
    if all(v is not None for v in overlaps):
        pool = _pool_by_score(
            pool,
            lambda r: finite((r.get("audio_quality") or {}).get("p_overlap")),
            policy.quality_regression_tolerance,
            higher_is_better=False,
        )
    winner = min(pool, key=_conservative_key)
    return winner, {
        "role": role,
        "status": "selected",
        "n_candidates": len(candidates),
        "n_l1_tied": len(pool),
        "winner": winner["candidate_id"],
        "winner_cer_route": winner["cer_route"],
        "winner_view": winner.get("view"),
        "winner_stream": winner.get("stream"),
        "view_reason_code": _view_reason(winner, candidates, policy),
    }


def _qkw_gain(new: dict[str, Any], old: dict[str, Any]) -> float | None:
    if not new.get("qkw_calibrated") or not old.get("qkw_calibrated"):
        return None
    n, o = finite(new.get("q_kw")), finite(old.get("q_kw"))
    return None if n is None or o is None else n - o


def _keyword_gain(new: dict[str, Any], old: dict[str, Any]) -> float | None:
    n, o = finite(new.get("keyword_score")), finite(old.get("keyword_score"))
    return None if n is None or o is None else n - o


def _nll_gain(new: dict[str, Any], old: dict[str, Any]) -> float | None:
    n, o = finite(new.get("nll")), finite(old.get("nll"))
    return None if n is None or o is None else o - n


def _speaker_gain(new: dict[str, Any], old: dict[str, Any]) -> float | None:
    n, o = finite(new.get("speaker_ref_score")), finite(old.get("speaker_ref_score"))
    if n is None or o is None:
        n, o = finite(new.get("gain_ref")), finite(old.get("gain_ref"))
        if n is None or o is None:
            return None
        return n - o
    return n - o


def _quality_not_worse(new: dict[str, Any], old: dict[str, Any], tolerance: float) -> bool:
    nq, oq = new.get("audio_quality") or {}, old.get("audio_quality") or {}
    for key in ("speech_ratio", "snr_vad_db", "dnsmos_ovrl"):
        n, o = finite(nq.get(key)), finite(oq.get(key))
        if n is not None and o is not None and n < o - tolerance:
            return False
    for key in ("p_music", "p_overlap", "clip_rate"):
        n, o = finite(nq.get(key)), finite(oq.get(key))
        if n is not None and o is not None and n > o + tolerance:
            return False
    return True


def trigger_s7(w1: dict[str, Any] | None, policy: RoutePolicy) -> tuple[bool, list[str]]:
    if w1 is None:
        return True, ["s1_no_rankable_candidate"]
    reasons: list[str] = []
    cer = finite(w1.get("cer_route"))
    if cer is not None and cer > 1e-9:
        reasons.append("s1_cer_nonzero")
    q = finite(w1.get("q_kw"))
    if policy.qkw_low_thr is not None and w1.get("qkw_calibrated") and q is not None and q < policy.qkw_low_thr:
        reasons.append("s1_qkw_below_threshold")
    if w1.get("content_class") in {"uncertain_or_hallucination", "competing_speech"}:
        reasons.append("s1_semantic_suspect")
    quality = w1.get("audio_quality") or {}
    clip = finite(quality.get("clip_rate"))
    if clip is not None and clip > policy.clip_rate_thr:
        reasons.append("s1_clipping_risk")
    speech = finite(quality.get("speech_ratio"))
    if speech is not None and speech < policy.min_speech_ratio:
        reasons.append("s1_speech_almost_empty")
    return bool(reasons), reasons


def _s7_available(rows: list[dict[str, Any]]) -> bool:
    flags = [r.get("s7_available") for r in rows if r.get("s7_available") is not None]
    if flags:
        return any(bool(x) for x in flags)
    return any(r.get("role") == "s7" for r in rows)


def route_uid(rows: Iterable[dict[str, Any]], policy: RoutePolicy) -> dict[str, Any]:
    values = [dict(r) for r in rows]
    w1, t1 = stage_winner(values, "s1", policy)
    w7, t7 = stage_winner(values, "s7", policy)
    triggered, trigger_reasons = trigger_s7(w1, policy)
    s7_ok = _s7_available(values)
    trace: list[dict[str, Any]] = [
        {"step": 1, "rule": "s1_stage_rank", "result": t1.get("winner"), "evidence": t1},
        {"step": 2, "rule": "s7_stage_rank", "result": t7.get("winner"), "evidence": t7},
        {"step": 3, "rule": "s7_trigger", "result": triggered, "evidence": {"reasons": trigger_reasons, "s7_available": s7_ok}},
    ]
    selected = w1
    switched = False
    selection_mode = "normal"
    reason = "KEEP_S1_CER0_TRUSTED"

    if w1 is None:
        if w7 is not None:
            selected, switched, reason = w7, True, "SWITCH_S7_S1_UNAVAILABLE"
            trace.append({"step": 4, "rule": "s1_unavailable_use_s7", "result": w7["candidate_id"]})
        else:
            fallback = _audit_fallback(values)
            if fallback is None:
                return {
                    "ok": False, "reason_code": "FAIL_NO_DECODABLE_S1", "triggered_s7": triggered,
                    "switched_s7": False, "trace": trace, "s1_winner": w1, "s7_winner": w7,
                    "s7_available": s7_ok, "selection_mode": "fail",
                }
            selected, reason, selection_mode = fallback, "AUDIT_FALLBACK_S1_RAW", "audit_fallback"
            trace.append({"step": 4, "rule": "audit_fallback_s1_raw", "result": fallback["candidate_id"]})
    elif not triggered:
        selected, reason = w1, "KEEP_S1_CER0_TRUSTED"
        trace.append({"step": 4, "rule": "keep_s1_not_triggered", "result": w1["candidate_id"]})
    elif not s7_ok:
        selected, reason = w1, "KEEP_S1_S7_UNAVAILABLE"
        trace.append({"step": 4, "rule": "s7_unavailable", "result": w1["candidate_id"]})
    elif w7 is None:
        selected, reason = w1, "KEEP_S1_S7_NO_RANKABLE"
        trace.append({"step": 4, "rule": "s7_no_rankable", "result": w1["candidate_id"]})
    elif w1.get("pcm_sha256") and w7.get("pcm_sha256") and w1["pcm_sha256"] == w7["pcm_sha256"]:
        selected, reason = w1, "SAME_AUDIO_KEEP_S1"
        trace.append({"step": 4, "rule": "same_pcm_keep_s1", "result": w1["candidate_id"]})
    else:
        c1, c7 = float(w1["cer_route"]), float(w7["cer_route"])
        if c7 < c1 - 1e-12:
            selected, switched, reason = w7, True, "SWITCH_S7_LOWER_CER"
            trace.append({"step": 4, "rule": "compare_cer_route", "result": "switch", "evidence": {"s1": c1, "s7": c7}})
        elif c7 > c1 + 1e-12:
            selected, reason = w1, "KEEP_S1_LOWER_CER"
            trace.append({"step": 4, "rule": "compare_cer_route", "result": "keep_s1", "evidence": {"s1": c1, "s7": c7}})
        else:
            qgain = _qkw_gain(w7, w1)
            kgain = _keyword_gain(w7, w1)
            ngain = _nll_gain(w7, w1)
            sgain = _speaker_gain(w7, w1)
            if qgain is not None and qgain >= policy.qkw_switch_margin:
                selected, switched, reason = w7, True, "SWITCH_S7_EQUAL_CER_QKW_GAIN"
            elif kgain is not None and kgain >= policy.qkw_switch_margin:
                selected, switched, reason = w7, True, "SWITCH_S7_EQUAL_CER_CTC_GAIN"
            elif qgain is None and ngain is not None and ngain >= policy.nll_switch_margin:
                selected, switched, reason = w7, True, "SWITCH_S7_EQUAL_CER_NLL_GAIN"
            elif sgain is not None and sgain >= policy.speaker_switch_margin and _quality_not_worse(w7, w1, policy.quality_regression_tolerance):
                selected, switched, reason = w7, True, "SWITCH_S7_EQUAL_CER_SPK_GAIN"
            else:
                selected, reason = w1, "KEEP_S1_TIE_NO_GAIN"
            trace.append({
                "step": 4, "rule": "equal_cer_tie_break", "result": "switch" if switched else "keep_s1",
                "evidence": {"qkw_gain": qgain, "ctc_gain": kgain, "nll_gain": ngain, "speaker_gain": sgain, "switched": switched},
            })

    cer = finite(selected.get("cer_route")) if selected is not None else None
    low_status = "zero" if cer is not None and cer <= 1e-12 else "min_nonzero"
    if policy.cer_accept_thr is not None and cer is not None and cer > policy.cer_accept_thr:
        low_status = "above_acceptance_threshold"
    production_eligible = (
        selection_mode == "normal"
        and cer is not None
        and reason != "AUDIT_FALLBACK_S1_RAW"
    )
    trace.append({"step": 5, "rule": "final_selected", "result": selected["candidate_id"] if selected else None, "reason_code": reason})
    return {
        "ok": True,
        "selected": selected,
        "s1_winner": w1,
        "s7_winner": w7,
        "triggered_s7": triggered,
        "switched_s7": switched,
        "trigger_reasons": trigger_reasons,
        "reason_code": reason,
        "reason_text": REASON_TEXT.get(reason, reason),
        "low_cer_status": low_status,
        "selection_mode": selection_mode,
        "s7_available": s7_ok,
        "s1_view_reason_code": t1.get("view_reason_code"),
        "s7_view_reason_code": t7.get("view_reason_code"),
        "has_finite_cer": cer is not None,
        "production_eligible": production_eligible,
        "trace": trace,
    }


def _audit_fallback(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    decodable = [
        r for r in rows
        if r.get("role") == "s1" and r.get("view") == "raw"
        and r.get("decode_ok", bool(r.get("source_wav")))
        and finite(r.get("cer_route")) is not None
    ]
    if not decodable:
        return None
    return min(decodable, key=lambda r: (float(r["cer_route"]), str(r.get("candidate_id"))))
