from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable


def finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


STREAM_ORDER = {"original": 0, "spk1": 1, "spk2": 2, "spk3": 3, "spk4": 4}


@dataclass(frozen=True)
class RoutePolicy:
    qkw_low_thr: float | None = None
    qkw_switch_margin: float = 0.01
    nll_switch_margin: float = 0.01
    speaker_switch_margin: float = 0.01
    quality_regression_tolerance: float = 0.10
    cer_accept_thr: float | None = None


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
    cer = finite(row.get("cer_route"))
    q = finite(row.get("q_kw")) if row.get("qkw_calibrated") else None
    nll = finite(row.get("nll"))
    quality = row.get("audio_quality") or {}
    extra = finite(row.get("extra_ratio"))
    overlap = finite(quality.get("p_overlap"))
    return (
        float("inf") if cer is None else cer,
        float("inf") if q is None else -q,
        float("inf") if nll is None else nll,
        float("inf") if extra is None else extra,
        float("inf") if overlap is None else overlap,
        0 if row.get("view") == "raw" else 1,
        0 if row.get("role") == "s1" else 1,
        _stream_key(row.get("stream")),
        str(row.get("candidate_id") or ""),
    )


def rankable(rows: Iterable[dict[str, Any]], role: str | None = None) -> list[dict[str, Any]]:
    return [dict(r) for r in rows if (role is None or r.get("role") == role) and r.get("validity") == "rankable" and finite(r.get("cer_route")) is not None]


def stage_winner(rows: Iterable[dict[str, Any]], role: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    candidates = rankable(rows, role)
    if not candidates:
        return None, {"role": role, "status": "no_rankable_candidate", "n_candidates": 0}
    winner = min(candidates, key=candidate_key)
    return winner, {
        "role": role,
        "status": "selected",
        "n_candidates": len(candidates),
        "winner": winner["candidate_id"],
        "winner_cer_route": winner["cer_route"],
        "winner_view": winner.get("view"),
        "winner_stream": winner.get("stream"),
    }


def _qkw_gain(new: dict[str, Any], old: dict[str, Any]) -> float | None:
    if not new.get("qkw_calibrated") or not old.get("qkw_calibrated"):
        return None
    n, o = finite(new.get("q_kw")), finite(old.get("q_kw"))
    return None if n is None or o is None else n - o


def _nll_gain(new: dict[str, Any], old: dict[str, Any]) -> float | None:
    n, o = finite(new.get("nll")), finite(old.get("nll"))
    return None if n is None or o is None else o - n


def _speaker_gain(new: dict[str, Any], old: dict[str, Any]) -> float | None:
    n, o = finite(new.get("speaker_ref_score")), finite(old.get("speaker_ref_score"))
    return None if n is None or o is None else n - o


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
    if w1.get("content_class") in {"uncertain_or_hallucination", "target_plus_interference"}:
        reasons.append("s1_semantic_risk")
    quality = w1.get("audio_quality") or {}
    if finite(quality.get("clip_rate")) is not None and float(quality.get("clip_rate")) > 0.01:
        reasons.append("s1_clipping_risk")
    return bool(reasons), reasons


def route_uid(rows: Iterable[dict[str, Any]], policy: RoutePolicy) -> dict[str, Any]:
    values = [dict(r) for r in rows]
    w1, t1 = stage_winner(values, "s1")
    w7, t7 = stage_winner(values, "s7")
    triggered, trigger_reasons = trigger_s7(w1, policy)
    trace: list[dict[str, Any]] = [{"step": 1, "rule": "stage_winner_s1", "result": t1}, {"step": 2, "rule": "stage_winner_s7", "result": t7}, {"step": 3, "rule": "s7_trigger", "result": triggered, "reasons": trigger_reasons}]
    selected = w1
    reason = "KEEP_S1_CER0_TRUSTED" if w1 is not None and finite(w1.get("cer_route")) == 0 and not triggered else "KEEP_S1_S7_NOT_TRIGGERED"
    switched = False
    if w1 is None and w7 is not None:
        selected, switched, reason = w7, True, "SWITCH_S7_S1_UNAVAILABLE"
    elif triggered and w7 is not None and w1 is not None:
        c1, c7 = float(w1["cer_route"]), float(w7["cer_route"])
        if w1.get("pcm_sha256") == w7.get("pcm_sha256"):
            reason = "SAME_AUDIO_KEEP_S1"
        elif c7 < c1 - 1e-12:
            selected, switched, reason = w7, True, "SWITCH_S7_LOWER_CER"
        elif c7 > c1 + 1e-12:
            reason = "KEEP_S1_LOWER_CER"
        else:
            qgain = _qkw_gain(w7, w1)
            ngain = _nll_gain(w7, w1)
            sgain = _speaker_gain(w7, w1)
            if qgain is not None and qgain >= policy.qkw_switch_margin:
                selected, switched, reason = w7, True, "SWITCH_S7_EQUAL_CER_QKW_GAIN"
            elif qgain is None and ngain is not None and ngain >= policy.nll_switch_margin and ngain > 0:
                selected, switched, reason = w7, True, "SWITCH_S7_EQUAL_CER_NLL_GAIN"
            elif sgain is not None and sgain >= policy.speaker_switch_margin and _quality_not_worse(w7, w1, policy.quality_regression_tolerance):
                selected, switched, reason = w7, True, "SWITCH_S7_EQUAL_CER_SPK_GAIN"
            else:
                reason = "KEEP_S1_TIE_NO_GAIN"
            trace.append({"step": 4, "rule": "equal_cer_tie_break", "qkw_gain": qgain, "nll_gain": ngain, "speaker_gain": sgain, "switched": switched})
    elif triggered and w7 is None and w1 is not None:
        reason = "KEEP_S1_S7_NO_RANKABLE"
    if selected is None:
        # Caller may still provide an audit fallback; this is a hard failure here.
        return {"ok": False, "reason_code": "FAIL_NO_RANKABLE_CANDIDATE", "triggered_s7": triggered, "trace": trace, "s1_winner": w1, "s7_winner": w7}
    cer = finite(selected.get("cer_route"))
    low_status = "zero" if cer is not None and cer <= 1e-12 else "min_nonzero"
    if policy.cer_accept_thr is not None and cer is not None and cer > policy.cer_accept_thr:
        low_status = "above_acceptance_threshold"
    trace.append({"step": 5, "rule": "final_selected", "result": selected["candidate_id"], "reason_code": reason})
    return {"ok": True, "selected": selected, "s1_winner": w1, "s7_winner": w7, "triggered_s7": triggered, "switched_s7": switched, "trigger_reasons": trigger_reasons, "reason_code": reason, "low_cer_status": low_status, "trace": trace}
