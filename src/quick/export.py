from __future__ import annotations

import hashlib
import os
import re
import shutil
from pathlib import Path
from typing import Any

from .audio import file_sha256
from .io import write_json, write_jsonl
from .route import STREAM_ORDER, finite, route_uid


def safe_slug(value: str, limit: int = 48) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "empty"
    return text[:limit]


def natural_key(value: str) -> tuple[Any, ...]:
    return tuple(int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(value)))


def group_orders(rows: list[dict[str, Any]]) -> dict[str, int]:
    keys = sorted({str(r["group_key"]) for r in rows}, key=lambda x: (0 if x.split("\0", 1)[0] == "pos" else 1, natural_key(x.split("\0", 1)[1])))
    return {key: i for i, key in enumerate(keys, 1)}


def uid_hash(split: str, uid: str) -> str:
    return hashlib.sha256(f"{split}\0{uid}".encode("utf-8")).hexdigest()[:8]


def stream_slot(stream: str) -> int:
    if stream in STREAM_ORDER:
        return STREAM_ORDER[stream]
    if str(stream).startswith("spk"):
        try:
            return int(str(stream)[3:]) + 1
        except ValueError:
            pass
    return 900


def _kind(row: dict[str, Any]) -> tuple[int, str, str]:
    if row.get("role") == "s1" and row.get("view") == "raw":
        return 1000, "s1", "raw"
    if row.get("role") == "s1":
        return 2000, "s1", "moss48k"
    if row.get("view") == "raw":
        return 3000, "s7", "raw"
    return 4000, "s7", "moss48k"


def _materialize(source: Path, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if file_sha256(dest) != file_sha256(source):
            raise RuntimeError(f"export collision with different bytes: {dest}")
        return "existing"
    try:
        os.link(source, dest)
        return "hardlink"
    except OSError:
        shutil.copy2(source, dest)
        return "copy"


def _candidate_json(row: dict[str, Any], export_name: str, slot: int, selected: bool, loss: str | None) -> dict[str, Any]:
    result = {
        "candidate_id": row["candidate_id"], "export_name": export_name, "slot": slot, "is_selected": selected,
        "role": row.get("role"), "arm": row.get("arm"), "stream": row.get("stream"), "view": row.get("view"),
        "source_wav": row.get("source_wav"), "file_sha256": row.get("file_sha256"), "pcm_sha256": row.get("pcm_sha256"),
        "raw_parent_candidate_id": row.get("raw_parent_candidate_id"), "raw_parent_pcm_sha256": row.get("raw_parent_pcm_sha256"),
        "validity": row.get("validity"), "content_class": row.get("content_class"), "loss_reason": loss,
        "asr": {k: row.get(k) for k in ("hyp", "cer_route", "cer_char", "cer_py", "cer_alias", "alias_hit", "wake_coverage", "extra_ratio", "core_hit", "nll", "q_kw", "score_kind")},
        "se": {k: row.get(k) for k in ("se_backend", "delta_cer", "cos_se_raw", "raw_parent_pcm_sha256")},
        "audio_quality": row.get("audio_quality") or {},
        "speaker": {k: row.get(k) for k in ("speaker_ref_score", "cos_se_raw", "gain_ref")},
    }
    return result


def export_flat(
    rows: list[dict[str, Any]],
    *,
    out_dir: str | Path,
    policy: Any,
    selected_only_dir: str | Path | None = None,
    selected_only_index: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    orders = group_orders(rows)
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_group.setdefault(str(row["group_key"]), []).append(row)
    all_index: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    failures: list[str] = []
    for group_key in sorted(by_group, key=lambda x: orders[x]):
        group_rows = by_group[group_key]
        split, uid = group_key.split("\0", 1)
        result = route_uid(group_rows, policy)
        if not result.get("ok"):
            fallback = min((r for r in group_rows if r.get("role") == "s1" and r.get("view") == "raw" and finite(r.get("cer_route")) is not None), key=lambda r: float(r["cer_route"]), default=None)
            if fallback is None:
                failures.append(group_key)
                continue
            result.update({"ok": True, "selected": fallback, "reason_code": "AUDIT_FALLBACK_S1_RAW", "selection_mode": "audit_fallback", "switched_s7": False})
        selected = result["selected"]
        selected_id = selected["candidate_id"]
        order = orders[group_key]
        prefix = f"G{order:06d}__{split}__U-{safe_slug(uid)}-{uid_hash(split, uid)}"
        cand_json: list[dict[str, Any]] = []
        for row in sorted(group_rows, key=lambda r: (_kind(r)[0] + stream_slot(str(r.get("stream"))), str(r["candidate_id"]))):
            is_selected = row["candidate_id"] == selected_id
            if is_selected:
                slot = 0
                marker = "SELECTED"
            else:
                base, role, view = _kind(row)
                slot = base + stream_slot(str(row.get("stream")))
                marker = "CAND"
            base, role, view = _kind(row)
            name = f"{prefix}__{slot:04d}__{marker}__{role}__{view}__{safe_slug(str(row.get('stream') or 'unknown'), 24)}__P-{str(row['pcm_sha256'])[:8]}.wav"
            mode = _materialize(Path(row["source_wav"]), root / name)
            loss = None if is_selected else _loss_reason(row, selected, result)
            cand_json.append(_candidate_json(row, name, slot, is_selected, loss) | {"materialize_mode": mode})
            all_index.append({"group_order": order, "group_prefix": prefix, "uid": uid, "split": split, "candidate_id": row["candidate_id"], "export_name": name, "slot": slot, "is_selected": is_selected, "role": row.get("role"), "arm": row.get("arm"), "stream": row.get("stream"), "view": row.get("view"), "pcm_sha256": row.get("pcm_sha256"), "cer_route": row.get("cer_route"), "q_kw_or_nll": row.get("q_kw") if row.get("qkw_calibrated") else row.get("nll"), "content_class": row.get("content_class"), "validity": row.get("validity"), "loss_reason": loss})
        reason_name = f"{prefix}__9000__ROUTE_REASON.json"
        decision = {
            "schema": "kws_s1_s7_moss_route_reason/v1", "group_order": order, "group_key": group_key, "group_prefix": prefix,
            "uid": uid, "split": split, "wake_text": group_rows[0].get("wake_text"), "lang": group_rows[0].get("lang"),
            "policy": metadata or {},
            "availability": {"s1": any(r.get("role") == "s1" for r in group_rows), "s7": any(r.get("role") == "s7" for r in group_rows), "expected_candidate_refs": len(group_rows), "exported_candidate_refs": len(cand_json)},
            "decision": {"status": "selected", "selection_mode": result.get("selection_mode", "normal"), "selected_candidate_id": selected_id, "selected_export_name": next(x["export_name"] for x in cand_json if x["is_selected"]), "selected_role": selected.get("role"), "selected_view": selected.get("view"), "selected_stream": selected.get("stream"), "triggered_s7": result.get("triggered_s7", False), "switched_s7": result.get("switched_s7", False), "reason_code": result.get("reason_code"), "reason_text": _reason_text(result), "low_cer_status": result.get("low_cer_status", "min_nonzero"), "production_eligible": False, "requires_manual_review": result.get("selection_mode") == "audit_fallback"},
            "stage_winners": {"s1_candidate_id": result.get("s1_winner", {}).get("candidate_id") if result.get("s1_winner") else None, "s7_candidate_id": result.get("s7_winner", {}).get("candidate_id") if result.get("s7_winner") else None},
            "decision_trace": result.get("trace", []), "candidates": cand_json,
            "audit": {"all_candidates_listed": True, "exactly_one_selected": sum(x["is_selected"] for x in cand_json) == 1, "selected_is_first_by_filename": True, "feature_scored_once_per_signature": True, "filename_collision": False},
        }
        write_json(root / reason_name, decision)
        reasons[result.get("reason_code", "UNKNOWN")] = reasons.get(result.get("reason_code", "UNKNOWN"), 0) + 1
        selected_rows.append({"group_order": order, "group_prefix": prefix, "uid": uid, "split": split, "selected": selected, "selected_name": decision["decision"]["selected_export_name"], "reason_name": reason_name})
    write_jsonl(root / "ZZZZZZ__EXPORT_INDEX.jsonl", all_index)
    summary = {"schema": "kws_s1_s7_flat_export/v1", "n_groups": len(by_group), "n_exported_groups": len(selected_rows), "n_failures": len(failures), "failed_groups": failures, "n_candidate_refs": len(all_index), "reason_counts": dict(sorted(reasons.items())), "selected_counts": {}, "production_approved": False, "flat_dir": str(root.resolve())}
    for item in selected_rows:
        key = f"{item['selected'].get('role')}:{item['selected'].get('view')}"
        summary["selected_counts"][key] = summary["selected_counts"].get(key, 0) + 1
    write_json(root / "ZZZZZZ__EXPORT_SUMMARY.json", summary)
    if selected_only_dir is not None:
        dest_root = Path(selected_only_dir)
        selected_index: list[dict[str, Any]] = []
        for item in selected_rows:
            selected = item["selected"]
            dest = dest_root / item["split"] / f"{item['uid']}.wav"
            mode = _materialize(Path(selected["source_wav"]), dest)
            selected_index.append({"uid": item["uid"], "split": item["split"], "dest_rel": str(dest.relative_to(dest_root)), "ok": True, "chosen_role": selected.get("role"), "chosen_arm": selected.get("arm"), "chosen_stream": selected.get("stream"), "chosen_view": selected.get("view"), "cer_route": selected.get("cer_route"), "nll": selected.get("nll"), "q_kw": selected.get("q_kw"), "pcm_sha256": selected.get("pcm_sha256"), "review_group_prefix": item["group_prefix"], "route_reason_json": item["reason_name"], "materialize_mode": mode})
        write_jsonl(selected_only_index or (dest_root / "index.jsonl"), selected_index)
        summary["selected_only_dir"] = str(dest_root.resolve())
        write_json(root / "ZZZZZZ__EXPORT_SUMMARY.json", summary)
    return summary


def _loss_reason(row: dict[str, Any], selected: dict[str, Any], result: dict[str, Any]) -> str:
    if row.get("validity") != "rankable":
        return str(row.get("validity"))
    if row.get("pcm_sha256") == selected.get("pcm_sha256") and row.get("candidate_id") != selected.get("candidate_id"):
        return "same_pcm_prefer_s1"
    if finite(row.get("cer_route")) is not None and finite(selected.get("cer_route")) is not None and float(row["cer_route"]) > float(selected["cer_route"]) + 1e-12:
        return "higher_cer"
    if row.get("qkw_calibrated") and selected.get("qkw_calibrated") and finite(row.get("q_kw")) is not None and finite(selected.get("q_kw")) is not None and row["q_kw"] < selected["q_kw"]:
        return "equal_cer_lower_qkw"
    return "deterministic_tie_or_stage_policy"


def _reason_text(result: dict[str, Any]) -> str:
    return {"KEEP_S1_CER0_TRUSTED": "s1 CER0 且语义可信，保持 s1", "SWITCH_S7_LOWER_CER": "s7 严格降低选路 CER", "KEEP_S1_LOWER_CER": "s7 CER 更高，低 CER 硬规则保持 s1", "KEEP_S1_TIE_NO_GAIN": "同 CER 但没有达到冻结的 q_kw/NLL/声纹增益", "SAME_AUDIO_KEEP_S1": "s1 与 s7 winner 为同一 PCM，保持 s1"}.get(str(result.get("reason_code")), str(result.get("reason_code")))
