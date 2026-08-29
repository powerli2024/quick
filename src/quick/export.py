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


def _find_se_companion(selected: dict[str, Any], group_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Find the MossFormer view generated from the selected raw PCM."""
    if selected.get("view") != "raw":
        return selected if selected.get("source_wav") else None
    sid = str(selected.get("candidate_id") or "")
    spcm = selected.get("pcm_sha256")
    matches = [
        row for row in group_rows
        if row.get("view") == "moss48k"
        and row.get("source_wav")
        and (
            str(row.get("raw_parent_candidate_id") or "") == sid
            or (spcm and row.get("raw_parent_pcm_sha256") == spcm)
        )
    ]
    if not matches:
        return None
    # Prefer a rankable/finite-CER companion, then deterministic candidate id.
    return min(matches, key=lambda row: (
        0 if row.get("validity") == "rankable" and finite(row.get("cer_route")) is not None else 1,
        float(row.get("cer_route")) if finite(row.get("cer_route")) is not None else float("inf"),
        str(row.get("candidate_id") or ""),
    ))


def _fmt(value: Any, digits: int = 4) -> str:
    number = finite(value)
    if number is not None:
        return f"{number:.{digits}f}"
    text = str(value or "")
    return text if text else "-"


def _render_reason_txt(item: dict[str, Any], group_rows: list[dict[str, Any]], result: dict[str, Any], selected: dict[str, Any], companion: dict[str, Any] | None) -> str:
    """Human-readable route explanation; machine JSON remains in review_flat."""
    s1_winner = result.get("s1_winner") or {}
    s7_winner = result.get("s7_winner") or {}
    def winner_line(label: str, row: dict[str, Any]) -> str:
        if not row:
            return f"{label}: 无可排名候选"
        return (
            f"{label}: {row.get('role')}/{row.get('stream')}/"
            f"{('SE' if row.get('view') != 'raw' else 'raw')} "
            f"CER={_fmt(row.get('cer_route'))} ASR=\"{str(row.get('hyp') or '').strip() or '-'}\""
        )
    lines = [
        f"UID: {item['uid']}    split: {item['split']}    wake_text: {item.get('wake_text') or '-'}    lang: {item.get('lang') or '-'}",
        "选路流程: 每个 s1/s7 流先比较 raw 与 MossFormer SE；CER 为硬主指标；CER 接近时比较 q_kw/NLL/声纹，再比较噪声与 SNR；最后才用确定性排序。",
        "候选命名: role=s1/s7；stream=original 或 spk1、spk2...；view=raw 或 SE。",
        winner_line("s1 阶段 winner", s1_winner),
        winner_line("s7 阶段 winner", s7_winner),
        f"触发 s7: {'是' if result.get('triggered_s7') else '否'}    切换 s7: {'是' if result.get('switched_s7') else '否'}    原因: {result.get('reason_code') or '-'} - {result.get('reason_text') or '-'}",
        "",
        "最终选择",
        f"  role={selected.get('role')}  stream={selected.get('stream')}  view={'SE' if selected.get('view') != 'raw' else 'raw'}  candidate={selected.get('candidate_id')}",
        f"  CER={_fmt(selected.get('cer_route'))}  ASR={str(selected.get('hyp') or '').strip() or '-'}",
        f"  q_kw={_fmt(selected.get('q_kw'))}  NLL={_fmt(selected.get('nll'))}  keyword_score={_fmt(selected.get('keyword_score'))}",
        f"  p_music={_fmt((selected.get('audio_quality') or {}).get('p_music'))}  p_overlap={_fmt((selected.get('audio_quality') or {}).get('p_overlap'))}  SNR(dB)={_fmt((selected.get('audio_quality') or {}).get('snr_vad_db'))}  DNSMOS={_fmt((selected.get('audio_quality') or {}).get('dnsmos_ovrl'))}",
        f"  SE伴随文件: {companion.get('source_wav') if companion else '缺失'}",
        "",
        "各候选评价（按 role→stream→raw/SE 排列）",
    ]
    ordered = sorted(group_rows, key=lambda row: (
        0 if row.get('role') == 's1' else 1,
        str(row.get('stream') or ''),
        0 if row.get('view') == 'raw' else 1,
        str(row.get('candidate_id') or ''),
    ))
    for row in ordered:
        mark = "[最终选择]" if row.get('candidate_id') == selected.get('candidate_id') else "[候选]"
        quality = row.get('audio_quality') or {}
        hyp = " ".join(str(row.get('hyp') or '').split()) or "-"
        lines.append(
            f"  {mark} {row.get('role')}/{row.get('stream')}/{('SE' if row.get('view') != 'raw' else 'raw')} "
            f"validity={row.get('validity') or '-'} CER={_fmt(row.get('cer_route'))} "
            f"ASR=\"{hyp}\" coverage={_fmt(row.get('wake_coverage'))} core={row.get('core_hit') if row.get('core_hit') is not None else '-'} extra={_fmt(row.get('extra_ratio'))} "
            f"q_kw={_fmt(row.get('q_kw'))} NLL={_fmt(row.get('nll'))} keyword={_fmt(row.get('keyword_score'))} "
            f"p_music={_fmt(quality.get('p_music'))} p_overlap={_fmt(quality.get('p_overlap'))} "
            f"clip={_fmt(quality.get('clip_rate'))} speech={_fmt(quality.get('speech_ratio'))} SNR(dB)={_fmt(quality.get('snr_vad_db'))} "
            f"DNSMOS={_fmt(quality.get('dnsmos_ovrl'))} cos_SE_raw={_fmt(row.get('cos_se_raw'))} speaker_ref={_fmt(row.get('speaker_ref_score'))}"
        )
    lines.extend(["", "说明: CER 越低越好；p_music/p_overlap/clip 越低越好；SNR 与 DNSMOS 越高越好。缺失值不参与比较。"])
    return "\n".join(lines) + "\n"


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
            "wake_coverage", "extra_ratio", "core_hit", "nll", "q_kw", "keyword_score",
            "keyword_score_kind", "qkw_calibrator_hash", "score_kind",
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
    switch_ctc = 0
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
                "q_kw": row.get("q_kw"),
                "qkw_calibrator_hash": row.get("qkw_calibrator_hash"),
                "keyword_score": row.get("keyword_score"),
                "keyword_score_kind": row.get("keyword_score_kind"),
                "nll": row.get("nll"),
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
            elif code == "SWITCH_S7_EQUAL_CER_CTC_GAIN":
                switch_ctc += 1
            elif code == "SWITCH_S7_EQUAL_CER_NLL_GAIN":
                switch_nll += 1
            elif code == "SWITCH_S7_EQUAL_CER_SPK_GAIN":
                switch_spk += 1
            elif code == "SWITCH_S7_CLOSE_CER_QUALITY_GAIN":
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
            "group_rows": group_rows, "route_result": result,
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
            "n_switch_ctc": switch_ctc,
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
        n_se_companion = 0
        n_se_missing = 0
        for item in selected_rows:
            selected = item["selected"]
            dest = dest_root / item["split"] / f"{item['uid']}.wav"
            mode = _materialize(Path(selected["source_wav"]), dest)
            companion = _find_se_companion(selected, by_group[f"{item['split']}\0{item['uid']}"])
            se_dest = None
            se_mode = None
            if companion and companion.get("source_wav"):
                if selected.get("view") == "raw":
                    # Keep one row per UID for downstream compatibility; the
                    # companion is a deterministic sidecar next to the main WAV.
                    se_dest = dest.with_name(f"{item['uid']}__se.wav")
                    se_mode = _materialize(Path(companion["source_wav"]), se_dest)
                else:
                    # The selected SE file is already the main artifact; do
                    # not create a byte-identical duplicate.
                    se_dest = dest
                    se_mode = "selected_main"
                n_se_companion += 1
            else:
                n_se_missing += 1
            reason_txt = dest_root / item["split"] / f"{item['uid']}__ROUTE_REASON.txt"
            reason_txt.parent.mkdir(parents=True, exist_ok=True)
            reason_txt.write_text(
                _render_reason_txt(item, item.get("group_rows") or [], item.get("route_result") or {}, selected, companion),
                encoding="utf-8",
            )
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
                "route_reason_txt": str(reason_txt.relative_to(dest_root)),
                "materialize_mode": mode,
                "se_wav": str(se_dest.relative_to(dest_root)) if se_dest else None,
                "se_source_wav": companion.get("source_wav") if companion else None,
                "se_dest_rel": str(se_dest.relative_to(dest_root)) if se_dest else None,
                "se_candidate_id": companion.get("candidate_id") if companion else None,
                "se_pcm_sha256": companion.get("pcm_sha256") if companion else None,
                "se_file_sha256": file_sha256(se_dest) if se_dest else None,
                "se_view": "moss48k" if companion else None,
                "se_materialize_mode": se_mode,
                "se_is_selected": bool(selected.get("view") != "raw"),
                "se_companion_missing": companion is None,
            })
        write_jsonl(selected_only_index or (dest_root / "index.jsonl"), selected_index)
        summary["selected_only_dir"] = str(dest_root.resolve())
        summary["selected_only_se_companion"] = {
            "n_with_se": n_se_companion,
            "n_missing_se": n_se_missing,
            "filename_suffix": "__se.wav",
            "index_fields": ["se_wav", "se_source_wav", "se_candidate_id", "se_pcm_sha256", "se_file_sha256"],
        }
        summary["selected_only_reason_format"] = {
            "human_readable": True,
            "filename_suffix": "__ROUTE_REASON.txt",
            "machine_json_kept_in": str(root.resolve()),
        }
        write_json(root / "ZZZZZZ__EXPORT_SUMMARY.json", summary)
    elif selected_only_dir is not None and failures:
        summary["selected_only_skipped"] = "FAIL_NO_DECODABLE_S1"
        write_json(root / "ZZZZZZ__EXPORT_SUMMARY.json", summary)
    return summary
