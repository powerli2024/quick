#!/usr/bin/env python3
"""Strict I0-I8 s1/s7 + MossFormer SE route runner.

The runner is intentionally sidecar/command based: model weights stay outside
quick, while every result is tied to explicit hashes and a reproducible policy.
Missing external evidence never becomes a production pass.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quick.cer import load_alias_table  # noqa: E402
from quick.export import export_flat  # noqa: E402
from quick.inventory import build_inventory  # noqa: E402
from quick.io import json_hash, read_json, read_jsonl, write_json, write_jsonl  # noqa: E402
from quick.route import RoutePolicy, route_uid, stage_winner  # noqa: E402
from quick.scoring import score_rows  # noqa: E402
from quick.se import add_se_views  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="strict I0-I8 s1/s7 MossFormer route")
    p.add_argument("--pos-neg", type=Path, required=True)
    p.add_argument("--s1-arm", required=True)
    p.add_argument("--s7-arm", required=True)
    p.add_argument("--expected-uids", type=int, default=0, help="combined pos+neg UID count; 0 disables")
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--asr-jsonl", type=Path, default=None)
    p.add_argument("--asr-command", default=None, help="template containing {manifest} and optionally {output}")
    p.add_argument("--nll-jsonl", type=Path, default=None)
    p.add_argument("--qkw-jsonl", type=Path, default=None)
    p.add_argument("--qkw-calibrated", action="store_true")
    p.add_argument("--embedding-jsonl", type=Path, default=None)
    p.add_argument("--noise-jsonl", type=Path, default=None)
    p.add_argument("--alias-json", type=Path, default=None)
    p.add_argument("--allow-missing-asr", action="store_true", help="audit-only; never production eligible")
    p.add_argument("--se-backend", choices=["command", "precomputed", "spectral", "none"], default="command")
    p.add_argument("--se-command", default=None)
    p.add_argument("--se-batch-command", default=None)
    p.add_argument("--precomputed-se-dir", type=Path, default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--qkw-low-thr", type=float, default=None)
    p.add_argument("--qkw-switch-margin", type=float, default=0.01)
    p.add_argument("--nll-switch-margin", type=float, default=0.01)
    p.add_argument("--speaker-switch-margin", type=float, default=0.01)
    p.add_argument("--cer-accept-thr", type=float, default=None)
    p.add_argument("--flat-dir", type=Path, default=None)
    p.add_argument("--selected-only-dir", type=Path, default=None)
    p.add_argument("--cmd-result-json", type=Path, default=None)
    p.add_argument("--presence-result-json", type=Path, default=None)
    p.add_argument("--contest-result-json", type=Path, default=None)
    p.add_argument("--cmd-command", default=None)
    p.add_argument("--presence-command", default=None)
    p.add_argument("--contest-command", default=None)
    p.add_argument("--cer-mean-max", type=float, default=0.03)
    p.add_argument("--cer0-drop-max", type=float, default=0.02)
    return p.parse_args()


def _run_sidecar_command(template: str, *, manifest: Path, output: Path, selected_dir: Path | None = None) -> Path:
    rendered = template.format(manifest=str(manifest), output=str(output), selected_dir=str(selected_dir or ""))
    subprocess.run(shlex.split(rendered, posix=True), check=True)
    if not output.is_file():
        raise RuntimeError(f"external command did not create declared output: {output}")
    return output


def _write_asr_manifest(work: Path, rows: list[dict[str, Any]]) -> Path:
    path = work / "asr_manifest.jsonl"
    write_jsonl(path, [{"candidate_id": r["candidate_id"], "input": r["source_wav"], "pcm_sha256": r["pcm_sha256"], "wake_text": r["wake_text"], "lang": r["lang"]} for r in rows])
    return path


def _load_optional(path: Path | None) -> dict[str, Any] | None:
    return read_json(path) if path is not None else None


def _external_eval(template: str | None, result: Path | None, *, selected_dir: Path | None, work: Path, manifest: Path) -> Path | None:
    if template is None:
        return result
    out = result or work / "external_eval.json"
    return _run_sidecar_command(template, manifest=manifest, output=out, selected_dir=selected_dir)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def local_evaluate(groups: dict[str, list[dict[str, Any]]], policy: RoutePolicy, expected: int, *, cer_mean_max: float, cer0_drop_max: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    decisions: list[dict[str, Any]] = []
    selected_cer: list[float] = []
    baseline_cer: list[float] = []
    for group_key in sorted(groups):
        rows = groups[group_key]
        decision = route_uid(rows, policy)
        if not decision.get("ok"):
            raw_s1 = [r for r in rows if r.get("role") == "s1" and r.get("view") == "raw" and r.get("validity") == "rankable"]
            fallback = min(raw_s1, key=lambda r: float(r["cer_route"]), default=None)
            if fallback is not None:
                decision.update({"ok": True, "selected": fallback, "reason_code": "AUDIT_FALLBACK_S1_RAW", "selection_mode": "audit_fallback"})
        if decision.get("ok"):
            selected_cer.append(float(decision["selected"]["cer_route"]))
        s1_raw, _ = stage_winner([r for r in rows if r.get("role") == "s1" and r.get("view") == "raw"], "s1")
        if s1_raw is not None:
            baseline_cer.append(float(s1_raw["cer_route"]))
        decision["group_key"] = group_key
        decisions.append(decision)
    paired = [float(d["selected"]["cer_route"]) - float(min((r for r in groups[d["group_key"]] if r.get("role") == "s1" and r.get("view") == "raw" and r.get("validity") == "rankable"), key=lambda r: float(r["cer_route"]), default=d["selected"])["cer_route"]) for d in decisions if d.get("ok")]
    n_worse = sum(x > 1e-12 for x in paired)
    cer0_selected = sum(x <= 1e-12 for x in selected_cer) / len(selected_cer) if selected_cer else None
    cer0_baseline = sum(x <= 1e-12 for x in baseline_cer) / len(baseline_cer) if baseline_cer else None
    coverage_ok = len(decisions) == expected if expected else all(d.get("ok") for d in decisions)
    checks = {
        "coverage": coverage_ok,
        "mean_cer": _mean(selected_cer) is not None and _mean(selected_cer) <= cer_mean_max,
        "paired_no_worsened": n_worse == 0,
        "cer0_drop": cer0_selected is not None and cer0_baseline is not None and cer0_baseline - cer0_selected <= cer0_drop_max + 1e-12,
    }
    return {"n_uid": len(decisions), "expected_uid": expected or len(decisions), "mean_cer": _mean(selected_cer), "mean_cer_baseline_s1_raw": _mean(baseline_cer), "cer0_rate": cer0_selected, "cer0_rate_baseline": cer0_baseline, "paired_mean_delta": _mean(paired), "n_paired_worsened": n_worse, "checks": checks, "local_pass": all(checks.values())}, decisions


def i8_validate(local: dict[str, Any], cmd: dict[str, Any] | None, presence: dict[str, Any] | None, contest: dict[str, Any] | None) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "PENDING_EXTERNAL", "production_approved": False, "local": local, "external": {"cmd": cmd, "presence": presence, "contest": contest}}
    if not local.get("local_pass"):
        result["status"] = "NO_GO"
        result["reason"] = "local_checks_failed"
        return result
    if cmd is None or presence is None or contest is None:
        result["reason"] = "need_cmd_presence_contest_results"
        return result
    def pair_ok(payload: dict[str, Any]) -> bool:
        base, cand = payload.get("baseline") or {}, payload.get("candidate") or {}
        if "frr" not in base or "frr" not in cand or "far" not in base or "far" not in cand:
            return False
        return (float(cand["frr"]) <= float(base["frr"]) and float(cand["far"]) <= float(base["far"]) and (float(cand["frr"]) < float(base["frr"]) or float(cand["far"]) < float(base["far"])))
    contest_ok = float(contest.get("candidate", {}).get("score", contest.get("candidate_score", -1))) >= float(contest.get("baseline", {}).get("score", contest.get("baseline_score", float("inf"))))
    if pair_ok(cmd) and pair_ok(presence) and contest_ok:
        result.update({"status": "PASS", "production_approved": True})
    else:
        result.update({"status": "NO_GO", "reason": "external_acceptance_failed", "checks": {"cmd": pair_ok(cmd), "presence": pair_ok(presence), "contest": contest_ok}})
    return result


def main() -> int:
    args = parse_args()
    work = args.work_dir.resolve(); work.mkdir(parents=True, exist_ok=True)
    if args.se_backend == "command" and not args.se_command and not args.se_batch_command:
        raise SystemExit("command SE requires --se-command or --se-batch-command")
    aliases = load_alias_table(args.alias_json)
    # I0/I1: exact arm discovery, coverage, and canonical PCM registry.
    refs, registry, inventory_meta = build_inventory(args.pos_neg, s1_arm=args.s1_arm, s7_arm=args.s7_arm, expected_uids=args.expected_uids)
    write_json(work / "inventory_meta.json", inventory_meta)
    write_jsonl(work / "raw_candidate_refs.jsonl", refs)
    # I2: all unique raw PCM gets exactly one SE view.
    all_rows, se_meta = add_se_views(refs, registry, work_dir=work, backend=args.se_backend, command=args.se_command, batch_command=args.se_batch_command, precomputed_dir=args.precomputed_se_dir, resume=not args.no_resume)
    write_jsonl(work / "candidate_refs_with_se.jsonl", all_rows)
    # I3: if ASR is a command, run it once over all raw/SE candidate refs.
    asr_path = args.asr_jsonl
    asr_manifest = _write_asr_manifest(work, all_rows)
    if args.asr_command:
        asr_path = _run_sidecar_command(args.asr_command, manifest=asr_manifest, output=work / "asr.jsonl")
    if asr_path is None and not args.allow_missing_asr:
        raise SystemExit("strict run needs --asr-jsonl or --asr-command")
    scored, score_meta = score_rows(all_rows, registry, asr_sidecar=asr_path, nll_sidecar=args.nll_jsonl, qkw_sidecar=args.qkw_jsonl, embedding_sidecar=args.embedding_jsonl, noise_sidecar=args.noise_jsonl, aliases=aliases, allow_missing_asr=args.allow_missing_asr, qkw_calibrated=args.qkw_calibrated)
    write_jsonl(work / "scored_candidates.jsonl", scored)
    # I4/I5: stage winners and complete route decisions.
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        groups[row["group_key"]].append(row)
    policy = RoutePolicy(qkw_low_thr=args.qkw_low_thr, qkw_switch_margin=args.qkw_switch_margin, nll_switch_margin=args.nll_switch_margin, speaker_switch_margin=args.speaker_switch_margin, cer_accept_thr=args.cer_accept_thr)
    policy_meta = {"s1_arm": args.s1_arm, "s7_arm": args.s7_arm, "se_backend": args.se_backend, "qkw_calibrated": args.qkw_calibrated, "route_policy_hash": json_hash(vars(args))}
    local, decisions = local_evaluate(groups, policy, args.expected_uids, cer_mean_max=args.cer_mean_max, cer0_drop_max=args.cer0_drop_max)
    write_jsonl(work / "route_decisions.jsonl", decisions)
    # I6/I7: flat all-candidate review and selected-only materialization.
    flat_dir = args.flat_dir or (work / "review_flat")
    summary = export_flat(scored, out_dir=flat_dir, policy=policy, selected_only_dir=args.selected_only_dir, metadata=policy_meta)
    # I8: external frozen CMD, Presence and contest contracts.
    eval_manifest = work / "selected_eval_manifest.jsonl"
    if args.selected_only_dir and (args.selected_only_dir / "index.jsonl").is_file():
        eval_manifest = args.selected_only_dir / "index.jsonl"
    cmd_path = _external_eval(args.cmd_command, args.cmd_result_json, selected_dir=args.selected_only_dir, work=work, manifest=eval_manifest)
    presence_path = _external_eval(args.presence_command, args.presence_result_json, selected_dir=args.selected_only_dir, work=work, manifest=eval_manifest)
    contest_path = _external_eval(args.contest_command, args.contest_result_json, selected_dir=args.selected_only_dir, work=work, manifest=eval_manifest)
    i8 = i8_validate(local, _load_optional(cmd_path), _load_optional(presence_path), _load_optional(contest_path))
    report = {"schema": "quick_s1_s7_route/v1", "status": i8["status"], "production_approved": i8["production_approved"], "phases": {"I0_I1_inventory": inventory_meta, "I2_se": se_meta, "I3_score": score_meta, "I4_I5_local": local, "I6_I7_export": summary, "I8": i8}, "policy": policy_meta, "paths": {"work_dir": str(work), "flat_dir": str(Path(flat_dir).resolve()), "selected_only_dir": str(args.selected_only_dir.resolve()) if args.selected_only_dir else None}}
    write_json(work / "report.json", report)
    lines = ["# quick s1/s7 route report", "", f"- status: **{report['status']}**", f"- production approved: `{report['production_approved']}`", f"- UID: `{local['n_uid']}/{local['expected_uid']}`", f"- mean CER: `{local['mean_cer']}` (baseline `{local['mean_cer_baseline_s1_raw']}`)", f"- paired worsened: `{local['n_paired_worsened']}`", f"- flat review: `{flat_dir}`", "", "I8 requires frozen CMD FRR/FAR, extract Presence and contest results; missing external results are never a pass."]
    (work / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
