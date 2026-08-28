from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .audio import file_sha256, pcm_sha256, quality_metrics, read_wav

_CJK = re.compile(r"[\u4e00-\u9fff]")
SKIP_STAGE_DIRS = {"meta", "reports", "best_sep", "packs", "review_flat"}


@dataclass
class Arm:
    split: str
    label: str
    index_path: Path
    rows: list[dict[str, Any]]


@dataclass
class Canonical:
    pcm_sha256: str
    file_sha256: str
    source_wav: str
    sample_rate: int
    metrics: dict[str, Any]
    wav: Any = field(default=None, repr=False)


def _label(split_root: Path, index: Path) -> str:
    return index.parent.relative_to(split_root).as_posix()


def discover_arms(pos_neg: str | Path, split: str) -> dict[str, Arm]:
    """Match kws stage discovery: one index per stage, plus thr_* children."""
    root = Path(pos_neg) / split
    if not root.is_dir():
        raise FileNotFoundError(f"missing split directory: {root}")
    from .io import read_jsonl

    out: dict[str, Arm] = {}
    for stage in sorted(root.iterdir(), key=lambda p: p.name):
        if not stage.is_dir() or stage.name in SKIP_STAGE_DIRS or stage.name.startswith("."):
            continue
        candidates = [(stage.name, stage / "index.jsonl")]
        candidates.extend((f"{stage.name}/{thr.name}", thr / "index.jsonl") for thr in sorted(stage.glob("thr_*")))
        for label, path in candidates:
            if not path.is_file():
                continue
            if label in out:
                raise ValueError(f"duplicate arm label {split}/{label}")
            out[label] = Arm(split, label, path, read_jsonl(path))
    if not out:
        raise ValueError(f"no index.jsonl found below {root}")
    return out


def _text(row: dict[str, Any]) -> str:
    return str(row.get("wake_text") or row.get("唤醒文本") or row.get("wake") or row.get("text") or "").strip()


def _lang(row: dict[str, Any], wake: str) -> tuple[str, str]:
    value = str(row.get("lang") or row.get("language") or "").strip().lower()
    aliases = {"zh": "zh", "cn": "zh", "chinese": "zh", "中文": "zh", "mandarin": "zh", "en": "en", "english": "en", "英文": "en"}
    if value in aliases:
        return aliases[value], "index_lang"
    if _CJK.search(wake):
        return "zh", "inferred_from_wake_text"
    if wake:
        return "en", "inferred_from_wake_text"
    return "", "missing_wake_text"


def _semantic_stream(name: str) -> str:
    tag = str(name or "").strip()
    if tag in {"original", "peak"}:
        return "original"
    return tag


def stream_names(row: dict[str, Any], wav_dir: Path, uid: str) -> list[str]:
    streams = row.get("streams")
    names: list[str] = []
    if isinstance(streams, dict) and streams:
        names = [_semantic_stream(k) for k in streams]
    elif isinstance(streams, list) and streams:
        for item in streams:
            if isinstance(item, dict):
                names.append(_semantic_stream(item.get("stream") or item.get("name") or item.get("id") or ""))
            else:
                names.append(_semantic_stream(item))
        names = [x for x in names if x]
    else:
        for wav in sorted(wav_dir.glob(f"{uid}_*.wav")):
            tag = wav.stem[len(uid) + 1 :]
            names.append(_semantic_stream(tag))
    # Preserve extract-sep identity: peak and original are one stream.
    return list(dict.fromkeys(x for x in names if x))


def wav_for(arm: Arm, uid: str, stream: str) -> Path:
    tags = ["peak", "original"] if stream == "original" else [stream]
    parent = arm.index_path.parent / "wav"
    for tag in tags:
        path = parent / f"{uid}_{tag}.wav"
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"missing WAV split={arm.split} arm={arm.label} uid={uid} stream={stream}: {parent / f'{uid}_{tags[0]}.wav'}")


def choose_arm(arms: dict[str, Arm], label: str, prefix: str) -> Arm:
    if label == "auto":
        raise ValueError(f"{prefix}_arm=auto is forbidden for strict runs; lock an exact arm label")
    if label not in arms:
        matches = [x for x in arms if x.startswith(prefix)]
        raise ValueError(f"unknown {prefix} arm={label!r}; available={matches}")
    return arms[label]


def _load_canonical(path: Path, registry: dict[str, Canonical]) -> tuple[str, str, str | None]:
    fsha = file_sha256(path)
    try:
        raw_x, raw_sr = read_wav(path)
    except Exception as exc:
        return f"undecodable:{fsha}", fsha, f"{type(exc).__name__}: {exc}"
    psha = pcm_sha256(raw_x, raw_sr)
    if psha not in registry:
        registry[psha] = Canonical(psha, fsha, str(path), raw_sr, quality_metrics(raw_x, raw_sr), wav=None)
    elif registry[psha].file_sha256 != fsha and not registry[psha].metrics.get("fatal_invalid"):
        # Same PCM, different container: keep first source_wav, record alias.
        pass
    return psha, fsha, None


def build_inventory(
    pos_neg: str | Path,
    *,
    s1_arm: str,
    s7_arm: str,
    expected_uids: int = 0,
    splits: tuple[str, ...] = ("pos", "neg"),
) -> tuple[list[dict[str, Any]], dict[str, Canonical], dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    registry: dict[str, Canonical] = {}
    selected: dict[str, Any] = {"splits": {}, "s1_arm": s1_arm, "s7_arm": s7_arm}
    s1_uids: dict[str, set[str]] = {}
    availability: dict[str, dict[str, Any]] = {}
    for split in splits:
        arms = discover_arms(pos_neg, split)
        a1 = choose_arm(arms, s1_arm, "s1")
        a7 = choose_arm(arms, s7_arm, "s7")
        if a1.label.split("/")[0].startswith("s7") or a7.label.split("/")[0].startswith("s1"):
            # Soft check only; labels are frozen by the caller.
            pass
        selected["splits"][split] = {
            "s1": a1.label, "s1_index": str(a1.index_path.resolve()),
            "s7": a7.label, "s7_index": str(a7.index_path.resolve()),
        }
        seen_by_arm: dict[str, set[str]] = {"s1": set(), "s7": set()}
        rows_by_arm: dict[str, dict[str, dict[str, Any]]] = {"s1": {}, "s7": {}}
        for role, arm in (("s1", a1), ("s7", a7)):
            for row in arm.rows:
                uid = str(row.get("uid") or row.get("id") or "").strip()
                if not uid:
                    raise ValueError(f"missing uid split={split} arm={arm.label}")
                if uid in seen_by_arm[role]:
                    raise ValueError(f"duplicate uid split={split} arm={arm.label} uid={uid}")
                seen_by_arm[role].add(uid)
                rows_by_arm[role][uid] = row
            if role == "s1":
                s1_uids[split] = set(seen_by_arm[role])
        extra = seen_by_arm["s7"] - seen_by_arm["s1"]
        if extra:
            raise ValueError(f"s7 has UIDs outside s1 split={split}: {sorted(extra)[:10]}")
        for uid in sorted(seen_by_arm["s1"]):
            s1_row = rows_by_arm["s1"][uid]
            wake, (lang, meta_src) = _text(s1_row), _lang(s1_row, _text(s1_row))
            if not wake or lang not in {"zh", "en"}:
                raise ValueError(f"missing/invalid wake_text or lang split={split} uid={uid}")
            s7_row = rows_by_arm["s7"].get(uid)
            s7_ok = s7_row is not None
            key = f"{split}\0{uid}"
            availability[key] = {
                "s1": True,
                "s7": s7_ok,
                "s1_streams": stream_names(s1_row, a1.index_path.parent / "wav", uid),
                "s7_streams": stream_names(s7_row, a7.index_path.parent / "wav", uid) if s7_ok else [],
            }
            for role, arm, row, present in (("s1", a1, s1_row, True), ("s7", a7, s7_row, s7_ok)):
                if not present or row is None:
                    continue
                wav_dir = arm.index_path.parent / "wav"
                for stream in stream_names(row, wav_dir, uid):
                    try:
                        path = wav_for(arm, uid, stream)
                    except FileNotFoundError as exc:
                        refs.append({
                            "candidate_id": f"C-{split}-{uid}-{role}-raw-{stream}",
                            "group_key": key, "uid": uid, "split": split,
                            "wake_text": wake, "lang": lang, "role": role, "arm": arm.label,
                            "stream": stream, "view": "raw", "source_wav": None,
                            "file_sha256": None, "pcm_sha256": None, "canonical_id": None,
                            "metadata_source": meta_src, "s7_available": s7_ok,
                            "validity": "fatal_invalid", "fatal_reason": str(exc),
                            "decode_ok": False,
                        })
                        continue
                    psha, fsha, err = _load_canonical(path, registry)
                    refs.append({
                        "candidate_id": f"C-{split}-{uid}-{role}-raw-{stream}",
                        "group_key": key, "uid": uid, "split": split,
                        "wake_text": wake, "lang": lang, "role": role, "arm": arm.label,
                        "stream": stream, "view": "raw", "source_wav": str(path),
                        "file_sha256": fsha, "pcm_sha256": psha, "canonical_id": psha,
                        "metadata_source": meta_src, "s7_available": s7_ok,
                        "validity": "fatal_invalid" if err else "pending",
                        "fatal_reason": err, "decode_ok": err is None,
                    })
    if expected_uids:
        total = sum(len(values) for values in s1_uids.values())
        if total != expected_uids:
            raise ValueError(f"s1 UID coverage total: {total} != {expected_uids}")
    selected["n_uid_by_split"] = {k: len(v) for k, v in sorted(s1_uids.items())}
    selected["n_uid"] = sum(selected["n_uid_by_split"].values())
    selected["n_candidate_refs_raw"] = len(refs)
    selected["n_unique_raw_pcm"] = len(registry)
    selected["n_s7_unavailable"] = sum(1 for v in availability.values() if not v["s7"])
    selected["availability"] = availability
    selected["n_undecodable_raw"] = sum(1 for r in refs if not r.get("decode_ok"))
    return refs, registry, selected
