from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .audio import pcm_sha256, read_wav
from .io import read_json, read_jsonl


class ExportValidationError(RuntimeError):
    pass


def _group_files(root: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(root.iterdir(), key=lambda p: p.name):
        if not path.is_file() or path.name.startswith("ZZZZZZ__"):
            continue
        parts = path.name.split("__")
        if len(parts) < 3:
            continue
        prefix = "__".join(parts[:3])
        groups[prefix].append(path)
    return groups


def validate_review_flat(
    root: str | Path,
    *,
    expected_groups: int | None = None,
    expected_uids: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    root = Path(root)
    errors: list[str] = []
    if not root.is_dir():
        raise ExportValidationError(f"review_flat missing: {root}")
    names = sorted(p.name for p in root.iterdir() if p.is_file())
    groups = _group_files(root)
    seen_prefixes: list[str] = []
    for name in names:
        if name.startswith("ZZZZZZ__"):
            continue
        parts = name.split("__")
        if len(parts) < 3:
            errors.append(f"unexpected filename {name}")
            continue
        prefix = "__".join(parts[:3])
        if not seen_prefixes or seen_prefixes[-1] != prefix:
            if prefix in seen_prefixes:
                errors.append(f"group_prefix not contiguous: {prefix}")
            seen_prefixes.append(prefix)

    index_path = root / "ZZZZZZ__EXPORT_INDEX.jsonl"
    summary_path = root / "ZZZZZZ__EXPORT_SUMMARY.json"
    if not index_path.is_file():
        errors.append("missing ZZZZZZ__EXPORT_INDEX.jsonl")
    if not summary_path.is_file():
        errors.append("missing ZZZZZZ__EXPORT_SUMMARY.json")
    index = read_jsonl(index_path) if index_path.is_file() else []
    summary = read_json(summary_path) if summary_path.is_file() else {}

    for prefix, files in groups.items():
        wavs = [p for p in files if p.suffix.lower() == ".wav"]
        reasons = [p for p in files if p.name.endswith("9000__ROUTE_REASON.json")]
        selected = [p for p in wavs if "__0000__SELECTED__" in p.name]
        if len(selected) != 1:
            errors.append(f"{prefix}: expected exactly one 0000__SELECTED wav, got {len(selected)}")
        elif wavs and wavs[0] != selected[0]:
            errors.append(f"{prefix}: 0000__SELECTED is not the first wav by filename")
        if len(reasons) != 1:
            errors.append(f"{prefix}: expected exactly one 9000__ROUTE_REASON.json, got {len(reasons)}")
            continue
        payload = read_json(reasons[0])
        decision = payload.get("decision") or {}
        selected_name = decision.get("selected_export_name")
        if selected and selected_name != selected[0].name:
            errors.append(f"{prefix}: JSON selected_export_name {selected_name!r} != {selected[0].name}")
        candidates = payload.get("candidates") or []
        audio_cands = [c for c in candidates if not c.get("export_placeholder") and str(c.get("export_name", "")).endswith(".wav")]
        if len(audio_cands) != len(wavs):
            errors.append(f"{prefix}: JSON audio candidates {len(audio_cands)} != wav count {len(wavs)}")
        if sum(bool(c.get("is_selected")) for c in candidates) != 1:
            errors.append(f"{prefix}: expected exactly one selected candidate")
        used = set()
        for cand in audio_cands:
            name = cand.get("export_name")
            path = root / str(name)
            if name in used:
                errors.append(f"{prefix}: basename collision {name}")
            used.add(name)
            if not path.is_file():
                errors.append(f"{prefix}: missing exported wav {name}")
                continue
            expected_hash = cand.get("pcm_sha256")
            if expected_hash:
                x, sr = read_wav(path)
                actual = pcm_sha256(x, sr)
                if actual != expected_hash:
                    errors.append(f"{prefix}: pcm hash mismatch {name}")
        audit = payload.get("audit") or {}
        if wavs:
            first_is_selected = "__0000__SELECTED__" in wavs[0].name
            if audit.get("selected_is_first_by_filename") is not True and first_is_selected:
                pass
            if not first_is_selected:
                errors.append(f"{prefix}: selected_is_first_by_filename failed")

    if expected_groups is not None and len(groups) != expected_groups:
        errors.append(f"group count {len(groups)} != expected {expected_groups}")
    if expected_uids is not None:
        got = set()
        for prefix, files in groups.items():
            reasons = [p for p in files if p.name.endswith("9000__ROUTE_REASON.json")]
            if not reasons:
                continue
            payload = read_json(reasons[0])
            got.add((str(payload.get("split")), str(payload.get("uid"))))
        missing = expected_uids - got
        if missing:
            errors.append(f"missing UID groups: {sorted(missing)[:10]}")
    if summary.get("n_failures"):
        errors.append(f"export summary reports failures={summary.get('failed_groups')}")
    result = {
        "ok": not errors,
        "n_groups": len(groups),
        "n_index_rows": len(index),
        "errors": errors,
    }
    if errors:
        raise ExportValidationError("review_flat validation failed: " + "; ".join(errors[:12]))
    return result
