from __future__ import annotations

from collections import defaultdict
from typing import Any

from .io import json_hash
from .route import RoutePolicy, finite, route_uid, stage_winner

PRODUCTION_SE_BACKENDS = {"command", "precomputed"}
I8_SCHEMA = "kws_i8_eval/v1"


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _far(row: dict[str, Any]) -> float | None:
    if "far" in row:
        try:
            return float(row["far"])
        except (TypeError, ValueError):
            return None
    if "rr" in row:
        try:
            return 1.0 - float(row["rr"])
        except (TypeError, ValueError):
            return None
    return None


def pair_improved(payload: dict[str, Any] | None) -> bool:
    if not payload:
        return False
    base, cand = payload.get("baseline") or {}, payload.get("candidate") or {}
    try:
        frr0, frr1 = float(base["frr"]), float(cand["frr"])
    except (KeyError, TypeError, ValueError):
        return False
    far0, far1 = _far(base), _far(cand)
    if far0 is None or far1 is None:
        return False
    frr_better = frr1 < frr0 - 1e-12
    far_better = far1 < far0 - 1e-12
    frr_worse = frr1 > frr0 + 1e-12
    far_worse = far1 > far0 + 1e-12
    return (frr_better or far_better) and not (frr_worse or far_worse)


def _not_worse_metrics(base: dict[str, Any], cand: dict[str, Any]) -> bool:
    try:
        frr0, frr1 = float(base["frr"]), float(cand["frr"])
    except (KeyError, TypeError, ValueError):
        return False
    far0, far1 = _far(base), _far(cand)
    if far0 is None or far1 is None:
        return False
    return frr1 <= frr0 + 1e-12 and far1 <= far0 + 1e-12


def contest_not_worse(payload: dict[str, Any] | None) -> bool:
    if not payload:
        return False
    cand = float(payload.get("candidate", {}).get("score", payload.get("candidate_score", float("-inf"))))
    base = float(payload.get("baseline", {}).get("score", payload.get("baseline_score", float("inf"))))
    return cand + 1e-12 >= base


def uid_fingerprint(groups: dict[str, list[dict[str, Any]]]) -> str:
    keys = []
    for group_key in groups:
        split, uid = str(group_key).split("\0", 1)
        keys.append((split, uid))
    return json_hash(sorted(keys))


def local_evaluate(
    groups: dict[str, list[dict[str, Any]]],
    policy: RoutePolicy,
    expected: int,
    *,
    cer_mean_max: float,
    cer0_drop_max: float,
    n_missing_asr: int = 0,
    allow_missing_asr: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected = int(expected or len(groups))
    decisions: list[dict[str, Any]] = []
    selected_cer: list[float] = []
    baseline_cer: list[float] = []
    paired: list[float] = []
    n_audit = 0
    n_ok = 0
    for group_key in sorted(groups):
        rows = groups[group_key]
        decision = route_uid(rows, policy)
        decision["group_key"] = group_key
        if decision.get("selection_mode") == "audit_fallback" or decision.get("reason_code") == "AUDIT_FALLBACK_S1_RAW":
            n_audit += 1
        if decision.get("ok"):
            n_ok += 1
            cer = finite((decision.get("selected") or {}).get("cer_route"))
            if cer is not None:
                selected_cer.append(cer)
        s1_raw, _ = stage_winner([r for r in rows if r.get("role") == "s1" and r.get("view") == "raw"], "s1", policy)
        if s1_raw is not None and finite(s1_raw.get("cer_route")) is not None:
            baseline_cer.append(float(s1_raw["cer_route"]))
        if decision.get("ok") and finite((decision.get("selected") or {}).get("cer_route")) is not None and s1_raw is not None:
            paired.append(float(decision["selected"]["cer_route"]) - float(s1_raw["cer_route"]))
        decisions.append(decision)

    n_selected_finite_cer = len(selected_cer)
    n_baseline_finite_cer = len(baseline_cer)
    n_paired = len(paired)
    n_worse = sum(x > 1e-12 for x in paired)
    cer0_selected = sum(x <= 1e-12 for x in selected_cer) / len(selected_cer) if selected_cer else None
    cer0_baseline = sum(x <= 1e-12 for x in baseline_cer) / len(baseline_cer) if baseline_cer else None
    finite_coverage = (
        len(decisions) == expected
        and n_selected_finite_cer == expected
        and n_baseline_finite_cer == expected
        and n_paired == expected
    )
    checks = {
        "coverage": finite_coverage and n_ok == expected,
        "finite_cer_coverage": finite_coverage,
        "n_selected_finite_cer": n_selected_finite_cer == expected,
        "n_baseline_finite_cer": n_baseline_finite_cer == expected,
        "n_paired": n_paired == expected,
        "no_audit_fallback": n_audit == 0,
        # Missing ASR always blocks local_pass; allow_missing_asr only continues scoring for audit.
        "no_missing_asr": n_missing_asr == 0,
        "mean_cer": _mean(selected_cer) is not None and _mean(selected_cer) <= cer_mean_max,
        "paired_no_worsened": n_worse == 0 and n_paired == expected,
        "cer0_drop": cer0_selected is not None and cer0_baseline is not None and cer0_baseline - cer0_selected <= cer0_drop_max + 1e-12,
    }
    local_pass = all(checks.values())
    local = {
        "n_uid": len(decisions),
        "expected_uid": expected,
        "uid_fingerprint": uid_fingerprint(groups),
        "n_selected_finite_cer": n_selected_finite_cer,
        "n_baseline_finite_cer": n_baseline_finite_cer,
        "n_paired": n_paired,
        "n_audit_fallback": n_audit,
        "n_missing_asr": n_missing_asr,
        "mean_cer": _mean(selected_cer),
        "mean_cer_baseline_s1_raw": _mean(baseline_cer),
        "cer0_rate": cer0_selected,
        "cer0_rate_baseline": cer0_baseline,
        "cer0": sum(x <= 1e-12 for x in selected_cer),
        "paired_mean_delta": _mean(paired),
        "n_paired_worsened": n_worse,
        "n_ok": n_ok,
        "n_triggered_s7": sum(bool(d.get("triggered_s7")) for d in decisions if d.get("ok")),
        "n_switched_s7": sum(bool(d.get("switched_s7")) for d in decisions if d.get("ok")),
        "reason_counts": _count(d.get("reason_code") for d in decisions),
        "checks": checks,
        "local_pass": local_pass,
        "status": "LOCAL_PASS_NEEDS_CMD_PRESENCE" if local_pass else "NO_GO",
    }
    return local, decisions


def _coverage_ok(payload: dict[str, Any], local: dict[str, Any]) -> tuple[bool, str | None]:
    cov = payload.get("coverage") or {}
    expected = int(local.get("expected_uid") or 0)
    try:
        n_uid = int(cov.get("n_uid"))
        exp = int(cov.get("expected_uid", expected))
    except (TypeError, ValueError):
        return False, "i8_coverage_missing"
    if n_uid != expected or exp != expected:
        return False, "i8_uid_count_mismatch"
    fp = cov.get("uid_fingerprint")
    if not fp or fp != local.get("uid_fingerprint"):
        return False, "i8_uid_fingerprint_mismatch"
    langs = cov.get("langs") or list((payload.get("baseline") or {}).get("by_lang") or {})
    if not langs:
        return False, "i8_langs_missing"
    return True, None


def _bindings_ok(payload: dict[str, Any], bindings: dict[str, Any]) -> tuple[bool, str | None]:
    got = payload.get("bindings") or {}
    required = (
        "signature_hash", "selected_index_hash", "s1_arm", "s7_arm", "se_backend",
        "asr_model_hash", "route_policy_hash",
    )
    for key in required:
        if not got.get(key):
            return False, f"i8_binding_missing_{key}"
        if bindings.get(key) and str(got[key]) != str(bindings[key]):
            return False, f"i8_binding_mismatch_{key}"
    if got.get("se_backend") not in PRODUCTION_SE_BACKENDS:
        return False, "i8_se_backend_not_production"
    if not got.get("mossformer_model_hash"):
        return False, "i8_mossformer_hash_missing"
    if bindings.get("mossformer_model_hash") and str(got["mossformer_model_hash"]) != str(bindings["mossformer_model_hash"]):
        return False, "i8_mossformer_hash_mismatch"
    return True, None


def _lang_not_worse(payload: dict[str, Any]) -> tuple[bool, str | None]:
    base = (payload.get("baseline") or {}).get("by_lang") or {}
    cand = (payload.get("candidate") or {}).get("by_lang") or {}
    if not base or not cand:
        return False, "i8_by_lang_missing"
    if set(base) != set(cand):
        return False, "i8_lang_set_mismatch"
    for lang, brow in base.items():
        if not _not_worse_metrics(brow, cand[lang]):
            return False, f"i8_lang_regression_{lang}"
    return True, None


def _audit_clean(payload: dict[str, Any], local: dict[str, Any]) -> tuple[bool, str | None]:
    audit = payload.get("audit") or {}
    if int(audit.get("n_audit_fallback", local.get("n_audit_fallback") or 1)) != 0:
        return False, "i8_has_audit_fallback"
    if audit.get("production_se") is False:
        return False, "i8_production_se_false"
    return True, None


def validate_i8_payload(
    payload: dict[str, Any] | None,
    *,
    kind: str,
    local: dict[str, Any],
    bindings: dict[str, Any],
) -> tuple[bool, str | None]:
    if not payload:
        return False, f"missing_{kind}"
    if payload.get("schema") != I8_SCHEMA:
        return False, f"{kind}_schema_invalid"
    if payload.get("kind") not in {kind, None} and payload.get("kind") != kind:
        return False, f"{kind}_kind_mismatch"
    ok, reason = _coverage_ok(payload, local)
    if not ok:
        return False, reason
    ok, reason = _bindings_ok(payload, bindings)
    if not ok:
        return False, reason
    ok, reason = _lang_not_worse(payload)
    if not ok:
        return False, reason
    ok, reason = _audit_clean(payload, local)
    if not ok:
        return False, reason
    if kind == "contest":
        if not contest_not_worse(payload):
            return False, "contest_score_dropped"
    else:
        if not pair_improved(payload):
            return False, f"{kind}_no_improvement_or_regression"
    return True, None


def i8_validate(
    local: dict[str, Any],
    cmd: dict[str, Any] | None,
    presence: dict[str, Any] | None,
    contest: dict[str, Any] | None,
    *,
    bindings: dict[str, Any] | None = None,
    se_backend: str | None = None,
    qkw_calibrated: bool = False,
    qkw_calibrator_hash: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "PENDING_EXTERNAL",
        "production_approved": False,
        "local": local,
        "external": {"cmd": cmd, "presence": presence, "contest": contest},
    }
    if not local.get("local_pass"):
        result["status"] = "NO_GO"
        result["reason"] = "local_checks_failed"
        return result
    if se_backend not in PRODUCTION_SE_BACKENDS:
        result["status"] = "LOCAL_PASS_NEEDS_CMD_PRESENCE"
        result["reason"] = "spectral_or_non_production_se_forbidden"
        return result
    if qkw_calibrated and not qkw_calibrator_hash:
        result["status"] = "NO_GO"
        result["reason"] = "qkw_calibrated_without_calibrator_hash"
        return result
    if cmd is None or presence is None or contest is None:
        result["reason"] = "need_cmd_presence_contest_results"
        result["status"] = "LOCAL_PASS_NEEDS_CMD_PRESENCE"
        return result
    bindings = dict(bindings or {})
    bindings.setdefault("se_backend", se_backend)
    checks = {}
    for kind, payload in (("cmd", cmd), ("presence", presence), ("contest", contest)):
        ok, reason = validate_i8_payload(payload, kind=kind, local=local, bindings=bindings)
        checks[kind] = {"ok": ok, "reason": reason}
        if not ok:
            result.update({"status": "NO_GO", "reason": reason, "checks": checks})
            return result
    result.update({"status": "PASS", "production_approved": True, "reason": "cmd_presence_contest_pass", "checks": checks})
    return result


def _count(values) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for value in values:
        if value:
            out[str(value)] += 1
    return dict(sorted(out.items()))
