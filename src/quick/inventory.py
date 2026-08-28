from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audio import file_sha256, pcm_sha256, quality_metrics, read_wav

_CJK = re.compile(r"[\u4e00-\u9fff]")


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
    wav: Any
    sample_rate: int
    metrics: dict[str, Any]


def _label(split_root: Path, index: Path) -> str:
    return index.parent.relative_to(split_root).as_posix()


def discover_arms(pos_neg: str | Path, split: str) -> dict[str, Arm]:
    root = Path(pos_neg) / split
    if not root.is_dir():
        raise FileNotFoundError(f"missing split directory: {root}")
    from .io import read_jsonl

    out: dict[str, Arm] = {}
    for index in sorted(root.rglob("index.jsonl")):
        label = _label(root, index)
        if label in out:
            raise ValueError(f"duplicate arm label {split}/{label}")
        out[label] = Arm(split, label, index, read_jsonl(index))
    if not out:
        raise ValueError(f"no index.jsonl found below {root}")
    return out


def _text(row: dict[str, Any]) -> str:
    return str(row.get("wake_text") or row.get("唤醒文本") or row.get("wake") or row.get("text") or "").strip()


def _lang(row: dict[str, Any], wake: str) -> str:
    value = str(row.get("lang") or row.get("language") or "").strip().lower()
    if value in {"zh", "cn", "chinese", "中文", "mandarin"}:
        return "zh"
    if value in {"en", "english", "英文"}:
        return "en"
    return "zh" if _CJK.search(wake) else "en" if wake else ""


def stream_names(row: dict[str, Any], wav_dir: Path, uid: str) -> list[str]:
    streams = row.get("streams")
    if isinstance(streams, dict) and streams:
        return sorted(str(k) for k in streams)
    if isinstance(streams, list) and streams:
        names = []
        for item in streams:
            if isinstance(item, dict):
                names.append(str(item.get("stream") or item.get("name") or item.get("id") or ""))
            else:
                names.append(str(item))
        return sorted(x for x in names if x)
    names: list[str] = []
    for wav in sorted(wav_dir.glob(f"{uid}_*.wav")):
        tag = wav.stem[len(uid) + 1 :]
        names.append("original" if tag == "peak" else tag)
    return sorted(set(names))


def wav_for(arm: Arm, uid: str, stream: str) -> Path:
    tag = "peak" if stream == "original" else stream
    path = arm.index_path.parent / "wav" / f"{uid}_{tag}.wav"
    if not path.is_file():
        raise FileNotFoundError(f"missing WAV split={arm.split} arm={arm.label} uid={uid} stream={stream}: {path}")
    return path.resolve()


def choose_arm(arms: dict[str, Arm], label: str, prefix: str) -> Arm:
    if label == "auto":
        raise ValueError(f"{prefix}_arm=auto is forbidden for strict runs; lock an exact arm label")
    if label not in arms:
        matches = [x for x in arms if x.startswith(prefix)]
        raise ValueError(f"unknown {prefix} arm={label!r}; available={matches}")
    return arms[label]


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
    for split in splits:
        arms = discover_arms(pos_neg, split)
        a1 = choose_arm(arms, s1_arm, "s1")
        a7 = choose_arm(arms, s7_arm, "s7")
        selected["splits"][split] = {
            "s1": a1.label, "s1_index": str(a1.index_path.resolve()),
            "s7": a7.label, "s7_index": str(a7.index_path.resolve()),
        }
        seen_by_arm: dict[str, set[str]] = {"s1": set(), "s7": set()}
        for role, arm in (("s1", a1), ("s7", a7)):
            for row in arm.rows:
                uid = str(row.get("uid") or row.get("id") or "").strip()
                if not uid:
                    raise ValueError(f"missing uid split={split} arm={arm.label}")
                if uid in seen_by_arm[role]:
                    raise ValueError(f"duplicate uid split={split} arm={arm.label} uid={uid}")
                seen_by_arm[role].add(uid)
                wake, lang = _text(row), _lang(row, _text(row))
                if not wake or lang not in {"zh", "en"}:
                    raise ValueError(f"missing/invalid wake_text or lang split={split} uid={uid}")
                wav_dir = arm.index_path.parent / "wav"
                for stream in stream_names(row, wav_dir, uid):
                    path = wav_for(arm, uid, stream)
                    raw_x, raw_sr = read_wav(path)
                    psha = pcm_sha256(raw_x, raw_sr)
                    fsha = file_sha256(path)
                    if psha not in registry:
                        registry[psha] = Canonical(psha, fsha, str(path), raw_x, raw_sr, quality_metrics(raw_x, raw_sr))
                    refs.append({
                        "candidate_id": f"C-{split}-{uid}-{role}-raw-{stream}",
                        "group_key": f"{split}\0{uid}", "uid": uid, "split": split,
                        "wake_text": wake, "lang": lang, "role": role, "arm": arm.label,
                        "stream": stream, "view": "raw", "source_wav": str(path),
                        "file_sha256": fsha, "pcm_sha256": psha, "canonical_id": psha,
                        "metadata_source": "index",
                    })
            if role == "s1":
                s1_uids[split] = set(seen_by_arm[role])
        extra = seen_by_arm["s7"] - seen_by_arm["s1"]
        if extra:
            raise ValueError(f"s7 has UIDs outside s1 split={split}: {sorted(extra)[:10]}")
    if expected_uids:
        total = sum(len(values) for values in s1_uids.values())
        if total != expected_uids:
            raise ValueError(f"s1 UID coverage total: {total} != {expected_uids}")
    selected["n_uid_by_split"] = {k: len(v) for k, v in sorted(s1_uids.items())}
    selected["n_candidate_refs_raw"] = len(refs)
    selected["n_unique_raw_pcm"] = len(registry)
    return refs, registry, selected
