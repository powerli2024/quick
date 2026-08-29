from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from .audio import file_sha256, pcm_sha256, read_wav
from .inventory import Canonical
from .io import write_jsonl


def default_audio_cache_root(work_dir: str | Path) -> Path:
    """Return a stable cache beside run-specific work directories."""
    return Path(work_dir).resolve().parent / "quick_audio_cache"


def _copy_atomic(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_name(f".{dest.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, dest)
    finally:
        if temporary.exists():
            temporary.unlink()


def _assert_pcm(path: Path, expected: str) -> None:
    wav, sr = read_wav(path)
    actual = pcm_sha256(wav, sr)
    if actual != expected:
        raise RuntimeError(f"audio cache PCM mismatch: {path} expected={expected} actual={actual}")


def materialize_sep_audio(
    refs: list[dict[str, Any]],
    registry: dict[str, Canonical],
    *,
    cache_root: str | Path,
    run_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Copy each unique separated PCM to a stable, content-addressed registry.

    The source files remain untouched.  Candidate references are rebound to the
    cached copies so all later SE/scoring/export stages survive a moved or
    removed upstream extract-sep run.
    """
    root = Path(cache_root).resolve() / "sep_pcm"
    root.mkdir(parents=True, exist_ok=True)
    hits = misses = 0
    records: list[dict[str, Any]] = []
    cached: dict[str, tuple[str, str]] = {}
    for pcm, can in sorted(registry.items()):
        if str(pcm).startswith("undecodable:"):
            continue
        source = Path(can.source_wav).resolve()
        dest = root / str(pcm)[:2] / f"{pcm}.wav"
        if dest.is_file():
            _assert_pcm(dest, str(pcm))
            hits += 1
            status = "hit"
        else:
            if not source.is_file():
                raise FileNotFoundError(f"separated source disappeared before cache materialization: {source}")
            _copy_atomic(source, dest)
            _assert_pcm(dest, str(pcm))
            misses += 1
            status = "fresh"
        fsha = file_sha256(dest)
        can.source_wav = str(dest)
        can.file_sha256 = fsha
        cached[str(pcm)] = (str(dest), fsha)
        records.append({
            "schema": "quick_sep_audio_cache/v1",
            "pcm_sha256": str(pcm),
            "file_sha256": fsha,
            "cached_wav": str(dest),
            "source_wav": str(source),
            "status": status,
        })

    for ref in refs:
        pcm = ref.get("canonical_id") or ref.get("pcm_sha256")
        if pcm not in cached:
            continue
        cached_wav, fsha = cached[str(pcm)]
        ref["origin_wav"] = ref.get("source_wav")
        ref["source_wav"] = cached_wav
        ref["file_sha256"] = fsha
        ref["audio_cache_kind"] = "sep_pcm"

    if run_manifest is not None:
        write_jsonl(run_manifest, records)
    return {
        "schema": "quick_sep_audio_cache/v1",
        "root": str(root),
        "n_unique": len(records),
        "n_cache_hit": hits,
        "n_cache_miss": misses,
        "n_fresh": misses,
        "manifest": str(Path(run_manifest).resolve()) if run_manifest is not None else None,
    }
