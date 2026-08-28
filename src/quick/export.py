from __future__ import annotations

import hashlib
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .audio import file_sha256
from .io import write_json, write_jsonl
from .route import REASON_TEXT, STREAM_ORDER, finite, route_uid


def safe_slug(value: str, limit: int = 48) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "empty"
    return text[:limit]


def natural_key(value: str) -> tuple[Any, ...]:
    return tuple(int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(value)))


def group_orders(rows: list[dict[str, Any]]) -> dict[str, int]:
    keys = sorted(
        {str(r["group_key"]) for r in rows},
        key=lambda x: (0 if x.split("\0", 1)[0] == "pos" else 1, natural_key(x.split("\0", 1)[-1]), x),
    )
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
    role = "s1" if row.get("role") == "s1" else "s7"
    view = "raw" if row.get("view") == "raw" else "moss48k"
    if role == "s1" and view == "raw":
        return 1000, role, view
    if role == "s1":
        return 2000, role, view
    if view == "raw":
        return 3000, role, view
    return 4000, role, view


def _fail_slot(row: dict[str, Any]) -> int:
    _, role, view = _kind(row)
    offset = {"s1-raw": 0, "s1-moss48k": 20, "s7-raw": 40, "s7-moss48k": 60}[f"{role}-{view}"]
    return 8000 + offset + stream_slot(str(row.get("stream")))


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


def _candidate_json(row: dict[str, Any], export_name: str, slot: int, selected: bool, loss: str | None, mode: str) -> dict[str, Any]:
    return {
        "candidate_id": row["candidate_id"],
        "export_name": export_name,
        "slot": slot,
        "is_selected": selected,
        "role": row.get("role"),
        "arm": row.get("arm"),
        "stream": row.get("stream"),
        "view": "moss48k" if row.get("view") != "raw" else "raw",
        "source_wav": row.get("source_wav"),
        "file_sha256": row.get("file_sha256"),
        "pcm_sha256": row.get("pcm_sha256"),
        "raw_parent_candidate_id": row.get("raw_parent_candidate_id"),
        "raw_parent_pcm_sha256": row.get("raw_parent_pcm_sha256"),
        "materialize_mode": mode,
        "export_placeholder": bool(row.get("export_placeholder") or mode == "placeholder"),
        "validity": row.get("validity"),
        "content_class": row.get("content_class"),
        "loss_reason": loss,
        "fatal_reason": row.get("fatal_reason"),
        "asr": {k: row.get(k) for k in (
            "hyp", "cer_route", "cer_char", "cer_py", "cer_alias", "alias_hit",
            "wake_coverage", "extra_ratio", "core_hit", "nll", "q_kw", "score_kind",
        )},
        "se": {
            "backend": row.get("se_backend"),
            "model_hash": row.get("se_model_hash"),
            "inference_mode": row.get("inference_mode", "full_waveform" if row.get("view") == "moss48k" else None),
            "raw_parent_pcm_sha256": row.get("raw_parent_pcm_sha256"),
            "delta_cer": row.get("delta_cer"),
            "cos_se_raw": row.get("cos_se_raw"),
        },
        "audio_quality": row.get("audio_quality") or {},
        "speaker": {
            "cos_se_raw": row.get("cos_se_raw"),
            "cos_independent_ref": row.get("speaker_ref_score"),
            "gain_ref": row.get("gain_ref"),
            "diagnostic_only": True,
        },
        "rank": {
            "stage_rank": row.get("stage_rank"),
            "cer_rank": row.get("cer_rank"),
            "loss_reason": loss,
        },
    }


def _loss_reason(row: dict[str, Any], selected: dict[str, Any], result: dict[str, Any]) -> str:
    if row.get("validity") != "rankable":
        return str(row.get("validity") or "fatal_invalid")
    if row.get("pcm_sha256") and row.get("pcm_sha256") == selected.get("pcm_sha256") and row.get("candidate_id") != selected.get("candidate_id"):
        return "same_pcm_prefer_s1"
    if finite(row.get("cer_route")) is not None and finite(selected.get("cer_route")) is not None:
        if float(row["cer_route"]) > float(selected["cer_route"]) + 1e-12:
            return "higher_cer"
        if abs(float(row["cer_route"]) - float(selected["cer_route"])) <= 1e-12:
            if row.get("qkw_calibrated") and selected.get("qkw_calibrated") and finite(row.get("q_kw")) is not None and finite(selected.get("q_kw")) is not None and row["q_kw"] < selected["q_kw"]:
                return "equal_cer_lower_qkw"
            if row.get("view") != "raw" and selected.get("view") == "raw":
                return "deterministic_raw_tie"
    return "deterministic_tie_or_stage_policy"


def _filename(prefix: str, slot: int, marker: str, role: str, view: str, stream: str, pcm: str | None) -> str:
    digest = (pcm or "undec")[:8]
    return f"{prefix}__{slot:04d}__{marker}__{role}__{view}__{safe_slug(stream or 'unknown', 24)}__P-{digest}"


def export_flat(
    rows: list[dict[str, Any]],
    *,
    out_dir: str | Path,
    policy: Any,
    selected_only_dir: str | Path | None = None,
    selected_only_index: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
    score_meta: dict[str, Any] | None = None,
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
    trigger_n = switch_n = 0
    switch_cer = switch_qkw = switch_nll = switch_spk = 0
    used_basenames: set[str] = set()
    se_delta = {"improved": 0, "worsened": 0, "same": 0, "by_content_class": defaultdict(lambda: {"improved": 0, "worsened": 0, "same": 0})}
    selected_cers: list[float] = []
    en_char: list[float] = []
    en_alias: list[float] = []

    for group_key in sorted(by_group, key=lambda x: orders[x]):
        group_rows = by_group[group_key]
        split, uid = group_key.split("\0", 1)
        result = route_uid(group_rows, policy)
        if not result.get("ok"):
            failures.append(group_key)
            # Do not invent a silent WAV winner.
            fail_name = f"G{orders[group_key]:06d}__{split}__U-{safe_slug(uid)}-{uid_hash(split, uid)}__9000__ROUTE_REASON.json"
            write_json(root / fail_name, {
                "schema": "kws_s1_s7_moss_route_reason/v1",
                "group_key": group_key, "uid": uid, "split": split,
                "decision": {
                    "status": "fail", "reason_code": "FAIL_NO_DECODABLE_S1",
                    "reason_text": REASON_TEXT["FAIL_NO_DECODABLE_S1"],
                    "production_eligible": False, "requires_manual_review": True,
                },
                "decision_trace": result.get("trace", []),
                "candidates": [],
            })
            reasons["FAIL_NO_DECODABLE_S1"] = reasons.get("FAIL_NO_DECODABLE_S1", 0) + 1
            continue
        selected = result["selected"]
        selected_id = selected["candidate_id"]
        order = orders[group_key]
        prefix = f"G{order:06d}__{split}__U-{safe_slug(uid)}-{uid_hash(split, uid)}"
        cand_json: list[dict[str, Any]] = []
        ordered = sorted(group_rows, key=lambda r: (_kind(r)[0] + stream_slot(str(r.get("stream"))), str(r["candidate_id"])))
        for row in ordered:
            is_selected = row["candidate_id"] == selected_id
            base, role, view = _kind(row)
            placeholder = bool(row.get("export_placeholder") or row.get("validity") == "fatal_invalid" and not row.get("source_wav"))
            if is_selected and placeholder:
                failures.append(group_key)
                continue
            if is_selected:
                slot, marker = 0, "SELECTED"
            elif placeholder:
                slot, marker = _fail_slot(row), "FAIL"
            else:
                slot, marker = base + stream_slot(str(row.get("stream"))), "CAND"
            pcm = row.get("pcm_sha256")
            stem = _filename(prefix, slot, marker, role, view, str(row.get("stream") or "unknown"), pcm)
            if placeholder:
                name = f"{stem}.json"
                mode = "placeholder"
                write_json(root / name, {
                    "schema": "kws_s1_s7_fail_placeholder/v1",
                    "candidate_id": row["candidate_id"],
                    "fatal_reason": row.get("fatal_reason"),
                    "role": role, "view": view, "stream": row.get("stream"),
                })
            else:
                name = f"{stem}.wav"
                if name in used_basenames:
                    raise RuntimeError(f"export basename collision: {name}")
                mode = _materialize(Path(row["source_wav"]), root / name)
            used_basenames.add(name)
            loss = None if is_selected else _loss_reason(row, selected, result)
            payload = _candidate_json(row, name, slot, is_selected, loss, mode)
            cand_json.append(payload)
            all_index.append({
                "group_order": order, "group_prefix": prefix, "uid": uid, "split": split,
                "candidate_id": row["candidate_id"], "export_name": name, "slot": slot,
                "is_selected": is_selected, "role": row.get("role"), "arm": row.get("arm"),
                "stream": row.get("stream"), "view": view, "pcm_sha256": row.get("pcm_sha256"),
                "cer_route": row.get("cer_route"),
                "q_kw_or_nll": row.get("q_kw") if row.get("qkw_calibrated") else row.get("nll"),
                "content_class": row.get("content_class"), "validity": row.get("validity"),
                "loss_reason": loss,
            })
            if row.get("view") == "moss48k" and finite(row.get("delta_cer")) is not None:
                delta = float(row["delta_cer"])
                klass = str(row.get("content_class") or "unknown")
                bucket = "improved" if delta < -1e-12 else "worsened" if delta > 1e-12 else "same"
                se_delta[bucket] += 1
                se_delta["by_content_class"][klass][bucket] += 1
        wav_names = sorted(x["export_name"] for x in cand_json if str(x["export_name"]).endswith(".wav"))
        selected_export = next(x["export_name"] for x in cand_json if x["is_selected"])
        first_wav = wav_names[0] if wav_names else None
        reason_name = f"{prefix}__9000__ROUTE_REASON.json"
        decision = {
            "schema": "kws_s1_s7_moss_route_reason/v1",
            "group_order": order,
            "group_key": group_key,
            "group_prefix": prefix,
            "uid": uid,
            "split": split,
            "wake_text": group_rows[0].get("wake_text"),
            "lang": group_rows[0].get("lang"),
            "policy": metadata or {},
            "availability": {
                "s1": any(r.get("role") == "s1" for r in group_rows),
                "s7": bool(result.get("s7_available")),
                "expected_candidate_refs": len(group_rows),
                "exported_candidate_refs": len(cand_json),
            },
            "decision": {
                "status": "selected",
                "selection_mode": result.get("selection_mode", "normal"),
                "selected_candidate_id": selected_id,
                "selected_export_name": selected_export,
                "selected_role": selected.get("role"),
                "selected_view": "moss48k" if selected.get("view") != "raw" else "raw",
                "selected_stream": selected.get("stream"),
                "triggered_s7": result.get("triggered_s7", False),
                "switched_s7": result.get("switched_s7", False),
                "reason_code": result.get("reason_code"),
                "reason_text": result.get("reason_text") or REASON_TEXT.get(str(result.get("reason_code")), str(result.get("reason_code"))),
                "low_cer_status": result.get("low_cer_status", "min_nonzero"),
                "production_eligible": False,
                "requires_manual_review": result.get("selection_mode") == "audit_fallback",
                "s1_view_reason_code": result.get("s1_view_reason_code"),
                "s7_view_reason_code": result.get("s7_view_reason_code"),
            },
            "stage_winners": {
                "s1_candidate_id": (result.get("s1_winner") or {}).get("candidate_id") if result.get("s1_winner") else None,
                "s7_candidate_id": (result.get("s7_winner") or {}).get("candidate_id") if result.get("s7_winner") else None,
            },
            "decision_trace": result.get("trace", []),
            "candidates": cand_json,
            "audit": {
                "all_candidates_listed": len(cand_json) == len(group_rows),
                "exactly_one_selected": sum(x["is_selected"] for x in cand_json) == 1,
                "selected_is_first_by_filename": first_wav == selected_export,
                "feature_scored_once_per_signature": True,
                "filename_collision": False,
            },
        }
        write_json(root / reason_name, decision)
        code = str(result.get("reason_code") or "UNKNOWN")
        reasons[code] = reasons.get(code, 0) + 1
        if result.get("triggered_s7"):
            trigger_n += 1
        if result.get("switched_s7"):
            switch_n += 1
            if code == "SWITCH_S7_LOWER_CER":
                switch_cer += 1
            elif code == "SWITCH_S7_EQUAL_CER_QKW_GAIN":
                switch_qkw += 1
            elif code == "SWITCH_S7_EQUAL_CER_NLL_GAIN":
                switch_nll += 1
            elif code == "SWITCH_S7_EQUAL_CER_SPK_GAIN":
                switch_spk += 1
        cer = finite(selected.get("cer_route"))
        if cer is not None:
            selected_cers.append(cer)
        if selected.get("lang") == "en":
            if finite(selected.get("cer_char")) is not None:
                en_char.append(float(selected["cer_char"]))
            if finite(selected.get("cer_alias", selected.get("cer_route"))) is not None:
                en_alias.append(float(selected.get("cer_alias", selected["cer_route"])))
        selected_rows.append({
            "group_order": order, "group_prefix": prefix, "uid": uid, "split": split,
            "selected": selected, "selected_name": selected_export, "reason_name": reason_name,
            "wake_text": group_rows[0].get("wake_text"), "lang": group_rows[0].get("lang"),
        })

    write_jsonl(root / "ZZZZZZ__EXPORT_INDEX.jsonl", all_index)
    selected_counts: dict[str, int] = {}
    stream_counts: dict[str, int] = {}
    split_counts = Counter(item["split"] for item in selected_rows)
    for item in selected_rows:
        role = item["selected"].get("role")
        view = "moss48k" if item["selected"].get("view") != "raw" else "raw"
        selected_counts[f"{role}:{view}"] = selected_counts.get(f"{role}:{view}", 0) + 1
        stream_counts[str(item["selected"].get("stream"))] = stream_counts.get(str(item["selected"].get("stream")), 0) + 1
    score_meta = score_meta or {}
    summary = {
        "schema": "kws_s1_s7_flat_export/v1",
        "n_groups": len(by_group),
        "n_exported_groups": len(selected_rows),
        "n_failures": len(failures),
        "failed_groups": failures,
        "n_uid": len(by_group),
        "n_pos": split_counts.get("pos", 0),
        "n_neg": split_counts.get("neg", 0),
        "n_candidate_refs": len(all_index),
        "n_unique_raw_pcm": score_meta.get("n_unique_raw_pcm"),
        "n_unique_se_pcm": score_meta.get("n_unique_se_pcm"),
        "feature_cache_hit": score_meta.get("feature_cache_hit"),
        "feature_cache_miss": score_meta.get("feature_cache_miss"),
        "reason_counts": dict(sorted(reasons.items())),
        "selected_counts": dict(sorted(selected_counts.items())),
        "selected_stream_counts": dict(sorted(stream_counts.items())),
        "s7": {
            "n_trigger": trigger_n,
            "n_switch": switch_n,
            "n_switch_lower_cer": switch_cer,
            "n_switch_qkw": switch_qkw,
            "n_switch_nll": switch_nll,
            "n_switch_spk": switch_spk,
        },
        "cer": {
            "mean_cer_route": sum(selected_cers) / len(selected_cers) if selected_cers else None,
            "cer0": sum(x <= 1e-12 for x in selected_cers),
            "cer0_rate": (sum(x <= 1e-12 for x in selected_cers) / len(selected_cers)) if selected_cers else None,
            "english_mean_cer_char": sum(en_char) / len(en_char) if en_char else None,
            "english_mean_cer_alias": sum(en_alias) / len(en_alias) if en_alias else None,
        },
        "se_delta_cer": {
            "improved": se_delta["improved"],
            "worsened": se_delta["worsened"],
            "same": se_delta["same"],
            "by_content_class": {k: dict(v) for k, v in se_delta["by_content_class"].items()},
        },
        "signatures": metadata or {},
        "production_approved": False,
        "ok": not failures,
        "flat_dir": str(root.resolve()),
    }
    write_json(root / "ZZZZZZ__EXPORT_SUMMARY.json", summary)
    if selected_only_dir is not None and not failures:
        dest_root = Path(selected_only_dir)
        # Wipe previous selected-only tree so reused work dirs cannot leave stale UID WAVs.
        if dest_root.exists():
            shutil.rmtree(dest_root)
        dest_root.mkdir(parents=True, exist_ok=True)
        selected_index: list[dict[str, Any]] = []
        for item in selected_rows:
            selected = item["selected"]
            dest = dest_root / item["split"] / f"{item['uid']}.wav"
            mode = _materialize(Path(selected["source_wav"]), dest)
            selected_index.append({
                "uid": item["uid"], "split": item["split"],
                "dest_rel": str(dest.relative_to(dest_root)), "ok": True,
                "chosen_role": selected.get("role"), "chosen_arm": selected.get("arm"),
                "chosen_stream": selected.get("stream"),
                "chosen_view": "moss48k" if selected.get("view") != "raw" else "raw",
                "cer_route": selected.get("cer_route"), "nll": selected.get("nll"),
                "q_kw": selected.get("q_kw"), "pcm_sha256": selected.get("pcm_sha256"),
                "source_wav": selected.get("source_wav"),
                "review_group_prefix": item["group_prefix"],
                "route_reason_json": item["reason_name"],
                "materialize_mode": mode,
            })
        write_jsonl(selected_only_index or (dest_root / "index.jsonl"), selected_index)
        summary["selected_only_dir"] = str(dest_root.resolve())
        write_json(root / "ZZZZZZ__EXPORT_SUMMARY.json", summary)
    elif selected_only_dir is not None and failures:
        summary["selected_only_skipped"] = "FAIL_NO_DECODABLE_S1"
        write_json(root / "ZZZZZZ__EXPORT_SUMMARY.json", summary)
    return summary
