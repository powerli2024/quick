from __future__ import annotations

import os
import shlex
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .accept import i8_validate, local_evaluate
from .cer import alias_table_hash, load_alias_table
from .export import export_flat
from .inventory import build_inventory
from .io import read_json, write_json, write_jsonl
from .policy import DEFAULT_POLICY_JSON, build_route_policy, load_policy_file
from .scoring import score_key, score_rows
from .se import add_se_views
from .signatures import assert_work_dir_signature, file_sha256, freeze_signatures, hash_model_dir
from .validate import ExportValidationError, validate_review_flat


@dataclass
class RunConfig:
    pos_neg: Path
    s1_arm: str
    s7_arm: str
    work_dir: Path
    expected_uids: int = 0
    asr_jsonl: Path | None = None
    asr_command: str | None = None
    asr_model_dir: Path | None = None
    nll_jsonl: Path | None = None
    nll_command: str | None = None
    embedding_command: str | None = None
    qkw_jsonl: Path | None = None
    qkw_command: str | None = None
    qkw_calibrated: bool = False
    embedding_jsonl: Path | None = None
    noise_jsonl: Path | None = None
    alias_json: Path | None = None
    allow_missing_asr: bool = False
    se_backend: str = "command"
    se_command: str | None = None
    se_batch_command: str | None = None
    precomputed_se_dir: Path | None = None
    resume: bool = True
    policy_json: Path | None = None
    qkw_low_thr: float | None = None
    qkw_switch_margin: float | None = None
    nll_switch_margin: float | None = None
    speaker_switch_margin: float | None = None
    cer_accept_thr: float | None = None
    flat_dir: Path | None = None
    selected_only_dir: Path | None = None
    cmd_result_json: Path | None = None
    presence_result_json: Path | None = None
    contest_result_json: Path | None = None
    cmd_command: str | None = None
    presence_command: str | None = None
    contest_command: str | None = None
    cer_mean_max: float | None = None
    cer0_drop_max: float | None = None
    extract_sep_run_id: str | None = None
    asr_model_hash: str | None = None
    asr_context_mode: str | None = None
    mossformer_model_hash: str | None = None
    speaker_encoder_hash: str | None = None
    qkw_calibrator_hash: str | None = None
    inference_signature: str | None = None
    noise_model_hashes: Any = None


def _split_cmd(template: str) -> list[str]:
    return shlex.split(template, posix=os.name != "nt")


def _render_command(
    template: str,
    *,
    manifest: Path,
    output: Path,
    selected_dir: Path | None = None,
) -> str:
    """Substitute only the documented placeholders in a sidecar command.

    ``str.format`` is intentionally avoided: callers may need literal braces
    for Python snippets, JSON, or shell expansions inside their command.
    """
    values = {
        "manifest": str(manifest),
        "output": str(output),
        "selected_dir": str(selected_dir or ""),
    }
    rendered = template
    for name, value in values.items():
        rendered = rendered.replace("{" + name + "}", value)
    return rendered


def _run_sidecar_command(template: str, *, manifest: Path, output: Path, selected_dir: Path | None = None) -> Path:
    rendered = _render_command(template, manifest=manifest, output=output, selected_dir=selected_dir)
    subprocess.run(_split_cmd(rendered), check=True)
    if not output.is_file():
        raise RuntimeError(f"external command did not create declared output: {output}")
    return output


def _write_asr_manifest(work: Path, rows: list[dict[str, Any]]) -> Path:
    path = work / "asr_manifest.jsonl"
    write_jsonl(path, [
        {
            "candidate_id": r["candidate_id"], "input": r.get("source_wav"),
            "pcm_sha256": r.get("pcm_sha256"), "wake_text": r["wake_text"], "lang": r["lang"],
            "score_key": r.get("score_key") or score_key(r),
        }
        for r in rows if r.get("source_wav")
    ])
    return path


def _load_optional(path: Path | None) -> dict[str, Any] | None:
    return read_json(path) if path is not None and path.is_file() else None


def _external_eval(
    template: str | None,
    result: Path | None,
    *,
    selected_dir: Path | None,
    work: Path,
    manifest: Path,
    name: str,
) -> Path | None:
    if template is None:
        return result
    out = result or (work / f"{name}.json")
    return _run_sidecar_command(template, manifest=manifest, output=out, selected_dir=selected_dir)


def _selected_index_hash(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    return file_sha256(path)


def run(cfg: RunConfig) -> dict[str, Any]:
    work = Path(cfg.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    if cfg.se_backend == "command" and not cfg.se_command and not cfg.se_batch_command:
        raise ValueError("command SE requires se_command or se_batch_command")
    if cfg.qkw_calibrated and not cfg.qkw_calibrator_hash:
        raise ValueError("--qkw-calibrated requires --qkw-calibrator-hash")

    aliases = load_alias_table(cfg.alias_json)
    policy_path = Path(cfg.policy_json) if cfg.policy_json else (DEFAULT_POLICY_JSON if DEFAULT_POLICY_JSON.is_file() else None)
    policy_file = load_policy_file(policy_path)
    acceptance = dict(policy_file.get("acceptance") or {})
    policy = build_route_policy(
        policy_file,
        overrides={
            "qkw_low_thr": cfg.qkw_low_thr,
            "qkw_switch_margin": cfg.qkw_switch_margin,
            "nll_switch_margin": cfg.nll_switch_margin,
            "speaker_switch_margin": cfg.speaker_switch_margin,
            "cer_accept_thr": cfg.cer_accept_thr,
        },
    )
    cer_mean_max = float(cfg.cer_mean_max if cfg.cer_mean_max is not None else acceptance.get("mean_cer_max", 0.03))
    cer0_drop_max = float(cfg.cer0_drop_max if cfg.cer0_drop_max is not None else acceptance.get("cer0_drop_max", 0.02))
    expected_uids = int(cfg.expected_uids or acceptance.get("expected_uid") or 0)

    asr_model_hash = cfg.asr_model_hash or hash_model_dir(cfg.asr_model_dir)
    # First freeze without post-command sidecar hashes; refreshed after scoring inputs exist.
    signatures = freeze_signatures(
        s1_arm=cfg.s1_arm, s7_arm=cfg.s7_arm,
        extract_sep_run_id=cfg.extract_sep_run_id,
        asr_model_hash=asr_model_hash,
        asr_context_mode=cfg.asr_context_mode,
        asr_sidecar=cfg.asr_jsonl,
        nll_sidecar=cfg.nll_jsonl,
        qkw_sidecar=cfg.qkw_jsonl,
        embedding_sidecar=cfg.embedding_jsonl,
        noise_sidecar=cfg.noise_jsonl,
        asr_command=cfg.asr_command,
        nll_command=cfg.nll_command,
        qkw_command=cfg.qkw_command,
        embedding_command=cfg.embedding_command,
        english_alias_table_hash=alias_table_hash(aliases),
        qkw_calibrator_hash=cfg.qkw_calibrator_hash,
        qkw_calibrated=cfg.qkw_calibrated,
        mossformer_model_hash=cfg.mossformer_model_hash,
        inference_signature=cfg.inference_signature or cfg.se_backend,
        speaker_encoder_hash=cfg.speaker_encoder_hash,
        noise_model_hashes=cfg.noise_model_hashes,
        route_policy=asdict(policy) | {"cer_mean_max": cer_mean_max, "cer0_drop_max": cer0_drop_max, "expected_uid": expected_uids},
        policy_json=policy_path,
        se_backend=cfg.se_backend, se_command=cfg.se_command, se_batch_command=cfg.se_batch_command,
    )
    assert_work_dir_signature(work, signatures)
    write_json(work / "signatures.json", signatures)

    refs, registry, inventory_meta = build_inventory(
        cfg.pos_neg, s1_arm=cfg.s1_arm, s7_arm=cfg.s7_arm, expected_uids=expected_uids,
    )
    write_json(work / "inventory_meta.json", {k: v for k, v in inventory_meta.items() if k != "availability"} | {
        "n_s7_unavailable": inventory_meta.get("n_s7_unavailable"),
    })
    write_json(work / "uid_availability.json", inventory_meta.get("availability") or {})
    write_jsonl(work / "raw_candidate_refs.jsonl", refs)

    all_rows, se_meta = add_se_views(
        refs, registry, work_dir=work, backend=cfg.se_backend, command=cfg.se_command,
        batch_command=cfg.se_batch_command, precomputed_dir=cfg.precomputed_se_dir,
        resume=cfg.resume, mossformer_model_hash=cfg.mossformer_model_hash,
        inference_signature=cfg.inference_signature,
    )
    write_jsonl(work / "candidate_refs_with_se.jsonl", all_rows)

    asr_path = cfg.asr_jsonl
    asr_manifest = _write_asr_manifest(work, all_rows)
    if cfg.asr_command:
        asr_path = _run_sidecar_command(cfg.asr_command, manifest=asr_manifest, output=work / "asr.jsonl")
    if asr_path is None and not cfg.allow_missing_asr:
        raise ValueError("strict run needs asr_jsonl or asr_command")
    nll_path = cfg.nll_jsonl
    if cfg.nll_command:
        nll_path = _run_sidecar_command(cfg.nll_command, manifest=asr_manifest, output=work / "nll.jsonl")
    qkw_path = cfg.qkw_jsonl
    if cfg.qkw_command:
        qkw_path = _run_sidecar_command(cfg.qkw_command, manifest=asr_manifest, output=work / "qkw.jsonl")
    embed_path = cfg.embedding_jsonl
    if cfg.embedding_command:
        embed_path = _run_sidecar_command(cfg.embedding_command, manifest=asr_manifest, output=work / "embed.jsonl")

    # Rebind signatures to the concrete sidecar bytes used for this run.
    signatures = freeze_signatures(
        s1_arm=cfg.s1_arm, s7_arm=cfg.s7_arm,
        extract_sep_run_id=cfg.extract_sep_run_id,
        asr_model_hash=asr_model_hash,
        asr_context_mode=cfg.asr_context_mode,
        asr_sidecar=asr_path,
        nll_sidecar=nll_path,
        qkw_sidecar=qkw_path,
        embedding_sidecar=embed_path,
        noise_sidecar=cfg.noise_jsonl,
        asr_command=cfg.asr_command,
        nll_command=cfg.nll_command,
        qkw_command=cfg.qkw_command,
        embedding_command=cfg.embedding_command,
        english_alias_table_hash=alias_table_hash(aliases),
        qkw_calibrator_hash=cfg.qkw_calibrator_hash,
        qkw_calibrated=cfg.qkw_calibrated,
        mossformer_model_hash=cfg.mossformer_model_hash or se_meta.get("mossformer_model_hash"),
        inference_signature=cfg.inference_signature or se_meta.get("inference_signature") or cfg.se_backend,
        speaker_encoder_hash=cfg.speaker_encoder_hash,
        noise_model_hashes=cfg.noise_model_hashes,
        route_policy=asdict(policy) | {"cer_mean_max": cer_mean_max, "cer0_drop_max": cer0_drop_max, "expected_uid": expected_uids},
        policy_json=policy_path,
        se_backend=cfg.se_backend, se_command=cfg.se_command, se_batch_command=cfg.se_batch_command,
    )
    write_json(work / "signatures.json", signatures)

    scored, score_meta = score_rows(
        all_rows, registry, asr_sidecar=asr_path, nll_sidecar=nll_path,
        qkw_sidecar=qkw_path, embedding_sidecar=embed_path,
        noise_sidecar=cfg.noise_jsonl, aliases=aliases,
        allow_missing_asr=cfg.allow_missing_asr, qkw_calibrated=cfg.qkw_calibrated,
        feature_dir=work, policy=policy,
    )
    write_jsonl(work / "scored_candidates.jsonl", scored)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        groups[row["group_key"]].append(row)
    expected = expected_uids or len(groups)
    local, decisions = local_evaluate(
        groups, policy, expected,
        cer_mean_max=cer_mean_max, cer0_drop_max=cer0_drop_max,
        n_missing_asr=int(score_meta.get("n_missing_asr") or 0),
        allow_missing_asr=cfg.allow_missing_asr,
    )
    write_jsonl(work / "route_decisions.jsonl", [
        {k: v for k, v in d.items() if k != "selected"} | {
            "selected_candidate_id": (d.get("selected") or {}).get("candidate_id"),
            "ok": d.get("ok"), "reason_code": d.get("reason_code"),
        }
        for d in decisions
    ])

    flat_dir = Path(cfg.flat_dir) if cfg.flat_dir else (work / "review_flat")
    summary = export_flat(
        scored, out_dir=flat_dir, policy=policy, selected_only_dir=cfg.selected_only_dir,
        metadata=signatures, score_meta=score_meta,
    )
    export_ok = True
    export_error = None
    try:
        expected_uid_set = {(r["split"], r["uid"]) for r in scored}
        validate_review_flat(flat_dir, expected_groups=len(groups), expected_uids=expected_uid_set)
    except ExportValidationError as exc:
        export_ok = False
        export_error = str(exc)
        local["local_pass"] = False
        local["status"] = "NO_GO"
        local.setdefault("checks", {})["export_validation"] = False
    if not summary.get("ok"):
        export_ok = False
        local["local_pass"] = False
        local["status"] = "NO_GO"

    selected_index = None
    if cfg.selected_only_dir and (Path(cfg.selected_only_dir) / "index.jsonl").is_file():
        selected_index = Path(cfg.selected_only_dir) / "index.jsonl"
    eval_manifest = selected_index or (work / "selected_eval_manifest.jsonl")
    cmd_path = _external_eval(cfg.cmd_command, cfg.cmd_result_json, selected_dir=cfg.selected_only_dir, work=work, manifest=eval_manifest, name="cmd")
    presence_path = _external_eval(cfg.presence_command, cfg.presence_result_json, selected_dir=cfg.selected_only_dir, work=work, manifest=eval_manifest, name="presence")
    contest_path = _external_eval(cfg.contest_command, cfg.contest_result_json, selected_dir=cfg.selected_only_dir, work=work, manifest=eval_manifest, name="contest")

    bindings = {
        "signature_hash": signatures.get("signature_hash"),
        "selected_index_hash": _selected_index_hash(selected_index),
        "s1_arm": cfg.s1_arm,
        "s7_arm": cfg.s7_arm,
        "se_backend": cfg.se_backend,
        "asr_model_hash": asr_model_hash,
        "route_policy_hash": signatures.get("route_policy_hash"),
        "mossformer_model_hash": signatures.get("mossformer_model_hash"),
        "uid_fingerprint": local.get("uid_fingerprint"),
    }
    write_json(work / "i8_bindings.json", bindings)
    i8 = i8_validate(
        local, _load_optional(cmd_path), _load_optional(presence_path), _load_optional(contest_path),
        bindings=bindings, se_backend=cfg.se_backend,
        qkw_calibrated=cfg.qkw_calibrated, qkw_calibrator_hash=cfg.qkw_calibrator_hash,
    )
    if not export_ok:
        i8["status"] = "NO_GO"
        i8["production_approved"] = False
        i8["reason"] = export_error or "export_validation_failed"
    report = {
        "schema": "quick_s1_s7_route/v1",
        "status": i8["status"],
        "production_approved": i8["production_approved"],
        "phases": {
            "I0_signatures": signatures,
            "I1_inventory": {k: inventory_meta[k] for k in inventory_meta if k != "availability"},
            "I2_se": se_meta,
            "I3_score": score_meta,
            "I4_I5_local": local,
            "I6_I7_export": summary,
            "I8": i8,
        },
        "policy_hash": signatures.get("route_policy_hash"),
        "paths": {
            "work_dir": str(work),
            "flat_dir": str(Path(flat_dir).resolve()),
            "selected_only_dir": str(Path(cfg.selected_only_dir).resolve()) if cfg.selected_only_dir else None,
            "policy_json": str(Path(policy_path).resolve()) if policy_path else None,
        },
    }
    write_json(work / "report.json", report)
    lines = [
        "# quick s1/s7 route report", "",
        f"- status: **{report['status']}**",
        f"- production approved: `{report['production_approved']}`",
        f"- UID: `{local['n_uid']}/{local['expected_uid']}`",
        f"- finite CER selected/baseline/paired: `{local['n_selected_finite_cer']}/{local['n_baseline_finite_cer']}/{local['n_paired']}`",
        f"- mean CER: `{local['mean_cer']}` (baseline `{local['mean_cer_baseline_s1_raw']}`)",
        f"- paired worsened: `{local['n_paired_worsened']}`",
        f"- s7 trigger/switch: `{local['n_triggered_s7']}/{local['n_switched_s7']}`",
        f"- se backend: `{cfg.se_backend}`",
        f"- flat review: `{flat_dir}`", "",
        "Spectral SE is audit-only. I8 PASS requires production SE + provenance-bound CMD/Presence/contest JSON.",
    ]
    (work / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_from_args(args: Any) -> dict[str, Any]:
    cfg = RunConfig(
        pos_neg=Path(args.pos_neg), s1_arm=args.s1_arm, s7_arm=args.s7_arm,
        work_dir=Path(args.work_dir), expected_uids=args.expected_uids,
        asr_jsonl=args.asr_jsonl, asr_command=args.asr_command,
        asr_model_dir=getattr(args, "asr_model_dir", None),
        nll_jsonl=args.nll_jsonl, nll_command=getattr(args, "nll_command", None),
        qkw_jsonl=args.qkw_jsonl,
        qkw_command=getattr(args, "qkw_command", None),
        qkw_calibrated=bool(args.qkw_calibrated),
        embedding_jsonl=args.embedding_jsonl,
        embedding_command=getattr(args, "embedding_command", None),
        noise_jsonl=args.noise_jsonl,
        alias_json=args.alias_json, allow_missing_asr=bool(args.allow_missing_asr),
        se_backend=args.se_backend, se_command=args.se_command,
        se_batch_command=args.se_batch_command, precomputed_se_dir=args.precomputed_se_dir,
        resume=not bool(getattr(args, "no_resume", False)),
        policy_json=getattr(args, "policy_json", None),
        qkw_low_thr=args.qkw_low_thr, qkw_switch_margin=args.qkw_switch_margin,
        nll_switch_margin=args.nll_switch_margin, speaker_switch_margin=args.speaker_switch_margin,
        cer_accept_thr=args.cer_accept_thr, flat_dir=args.flat_dir,
        selected_only_dir=args.selected_only_dir,
        cmd_result_json=args.cmd_result_json, presence_result_json=args.presence_result_json,
        contest_result_json=args.contest_result_json,
        cmd_command=args.cmd_command, presence_command=args.presence_command,
        contest_command=args.contest_command,
        cer_mean_max=args.cer_mean_max, cer0_drop_max=args.cer0_drop_max,
        extract_sep_run_id=getattr(args, "extract_sep_run_id", None),
        asr_model_hash=getattr(args, "asr_model_hash", None),
        asr_context_mode=getattr(args, "asr_context_mode", None),
        mossformer_model_hash=getattr(args, "mossformer_model_hash", None),
        speaker_encoder_hash=getattr(args, "speaker_encoder_hash", None),
        qkw_calibrator_hash=getattr(args, "qkw_calibrator_hash", None),
        inference_signature=getattr(args, "inference_signature", None),
    )
    return run(cfg)
