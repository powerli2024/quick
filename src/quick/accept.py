from __future__ import annotations

from collections import defaultdict
from typing import Any

from .route import RoutePolicy, finite, route_uid, stage_winner


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def pair_improved(payload: dict[str, Any] | None) -> bool:
    if not payload:
        return False
    base, cand = payload.get("baseline") or {}, payload.get("candidate") or {}

    def far(row: dict[str, Any]) -> float | None:
        if "far" in row:
            return float(row["far"])
        if "rr" in row:
            return 1.0 - float(row["rr"])
        return None

    try:
        frr0, frr1 = float(base["frr"]), float(cand["frr"])
    except (KeyError, TypeError, ValueError):
        return False
    far0, far1 = far(base), far(cand)
    if far0 is None or far1 is None:
        return False
    frr_better = frr1 < frr0 - 1e-12
    far_better = far1 < far0 - 1e-12
    frr_worse = frr1 > frr0 + 1e-12
    far_worse = far1 > far0 + 1e-12
    return (frr_better or far_better) and not (frr_worse or far_worse)


def contest_not_worse(payload: dict[str, Any] | None) -> bool:
    if not payload:
        return False
    cand = float(payload.get("candidate", {}).get("score", payload.get("candidate_score", float("-inf"))))
    base = float(payload.get("baseline", {}).get("score", payload.get("baseline_score", float("inf"))))
    return cand + 1e-12 >= base


def local_evaluate(
    groups: dict[str, list[dict[str, Any]]],
    policy: RoutePolicy,
    expected: int,
    *,
    cer_mean_max: float,
    cer0_drop_max: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    decisions: list[dict[str, Any]] = []
    selected_cer: list[float] = []
    baseline_cer: list[float] = []
    for group_key in sorted(groups):
        rows = groups[group_key]
        decision = route_uid(rows, policy)
        decision["group_key"] = group_key
        if decision.get("ok"):
            cer = finite(decision["selected"].get("cer_route"))
            if cer is not None:
                selected_cer.append(cer)
        s1_raw, _ = stage_winner([r for r in rows if r.get("role") == "s1" and r.get("view") == "raw"], "s1")
        if s1_raw is not None:
            baseline_cer.append(float(s1_raw["cer_route"]))
        decisions.append(decision)
    paired: list[float] = []
    for d in decisions:
        if not d.get("ok"):
            continue
        raws = [
            r for r in groups[d["group_key"]]
            if r.get("role") == "s1" and r.get("view") == "raw" and r.get("validity") == "rankable" and finite(r.get("cer_route")) is not None
        ]
        if not raws:
            continue
        base = min(raws, key=lambda r: float(r["cer_route"]))
        paired.append(float(d["selected"]["cer_route"]) - float(base["cer_route"]))
    n_worse = sum(x > 1e-12 for x in paired)
    cer0_selected = sum(x <= 1e-12 for x in selected_cer) / len(selected_cer) if selected_cer else None
    cer0_baseline = sum(x <= 1e-12 for x in baseline_cer) / len(baseline_cer) if baseline_cer else None
    coverage_ok = (len(decisions) == expected and all(d.get("ok") for d in decisions)) if expected else all(d.get("ok") for d in decisions)
    checks = {
        "coverage": coverage_ok,
        "mean_cer": _mean(selected_cer) is not None and _mean(selected_cer) <= cer_mean_max,
        "paired_no_worsened": n_worse == 0,
        "cer0_drop": cer0_selected is not None and cer0_baseline is not None and cer0_baseline - cer0_selected <= cer0_drop_max + 1e-12,
    }
    local = {
        "n_uid": len(decisions),
        "expected_uid": expected or len(decisions),
        "mean_cer": _mean(selected_cer),
        "mean_cer_baseline_s1_raw": _mean(baseline_cer),
        "cer0_rate": cer0_selected,
        "cer0_rate_baseline": cer0_baseline,
        "cer0": sum(x <= 1e-12 for x in selected_cer),
        "paired_mean_delta": _mean(paired),
        "n_paired_worsened": n_worse,
        "n_ok": sum(bool(d.get("ok")) for d in decisions),
        "n_triggered_s7": sum(bool(d.get("triggered_s7")) for d in decisions if d.get("ok")),
        "n_switched_s7": sum(bool(d.get("switched_s7")) for d in decisions if d.get("ok")),
        "reason_counts": _count(d.get("reason_code") for d in decisions),
        "checks": checks,
        "local_pass": all(checks.values()),
        "status": "LOCAL_PASS_NEEDS_CMD_PRESENCE" if all(checks.values()) else "NO_GO",
    }
    return local, decisions


def i8_validate(
    local: dict[str, Any],
    cmd: dict[str, Any] | None,
    presence: dict[str, Any] | None,
    contest: dict[str, Any] | None,
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
    if cmd is None or presence is None or contest is None:
        result["reason"] = "need_cmd_presence_contest_results"
        result["status"] = "LOCAL_PASS_NEEDS_CMD_PRESENCE"
        return result
    cmd_ok = pair_improved(cmd)
    presence_ok = pair_improved(presence)
    contest_ok = contest_not_worse(contest)
    if cmd_ok and presence_ok and contest_ok:
        result.update({"status": "PASS", "production_approved": True, "reason": "cmd_presence_contest_pass"})
    else:
        result.update({
            "status": "NO_GO",
            "reason": "external_acceptance_failed",
            "checks": {"cmd": cmd_ok, "presence": presence_ok, "contest": contest_ok},
        })
    return result


def _count(values) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for value in values:
        if value:
            out[str(value)] += 1
    return dict(sorted(out.items()))
