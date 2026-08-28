from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from .audio import file_sha256, pcm_sha256, quality_metrics, read_wav, write_wav
from .inventory import Canonical as InventoryCanonical
from .io import write_json, write_jsonl


def _safe(value: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _spectral(x: np.ndarray, sr: int) -> np.ndarray:
    """Small deterministic control SE; not a substitute for MossFormer."""
    y = np.asarray(x, dtype=np.float32).reshape(-1)
    if y.size < max(16, sr // 20):
        return y.copy()
    nfft = 512
    hop = 128
    win = np.hanning(nfft).astype(np.float32)
    pad = nfft // 2
    z = np.pad(y, (pad, pad))
    frames = 1 + max(0, (len(z) - nfft) // hop)
    noise_frames = max(1, min(frames, int(round(0.25 * sr / hop))))
    noise_power = np.zeros(nfft // 2 + 1, dtype=np.float32)
    for i in range(noise_frames):
        frame = z[i * hop : i * hop + nfft] * win
        noise_power += np.abs(np.fft.rfft(frame)) ** 2
    noise_power /= noise_frames
    out = np.zeros_like(z, dtype=np.float32)
    norm = np.zeros_like(z, dtype=np.float32)
    for i in range(frames):
        frame = z[i * hop : i * hop + nfft] * win
        spec = np.fft.rfft(frame)
        power = np.abs(spec) ** 2
        gain = np.sqrt(np.maximum(power - 1.2 * noise_power, 0.05 * power + 1e-10) / (power + 1e-10))
        enhanced = np.fft.irfft(spec * gain, nfft).astype(np.float32) * win
        out[i * hop : i * hop + nfft] += enhanced
        norm[i * hop : i * hop + nfft] += win * win
    result = out / np.maximum(norm, 1e-6)
    return np.clip(result[pad : pad + len(y)], -1, 1).astype(np.float32)


def _validate(src: Path, dest: Path) -> tuple[np.ndarray, int]:
    before, sr0 = read_wav(src)
    after, sr1 = read_wav(dest)
    if after.size == 0 or not np.isfinite(after).all():
        raise RuntimeError(f"SE output is empty/nonfinite: {dest}")
    tol = max(int(round(0.1 * sr0)), int(round(before.size * 0.02)))
    if abs(after.size / max(sr1, 1) - before.size / max(sr0, 1)) > 0.1 or abs(after.size - before.size * sr1 / sr0) > tol:
        raise RuntimeError(f"SE changed duration beyond tolerance: {src} -> {dest}")
    return after, sr1


def add_se_views(
    refs: list[dict[str, Any]],
    registry: dict[str, InventoryCanonical],
    *,
    work_dir: str | Path,
    backend: str,
    command: str | None = None,
    batch_command: str | None = None,
    precomputed_dir: str | Path | None = None,
    resume: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    backend = str(backend).lower().strip()
    if backend not in {"none", "spectral", "command", "precomputed"}:
        raise ValueError(f"unsupported SE backend {backend!r}")
    if backend == "none":
        return list(refs), {"backend": backend, "n_unique_raw": len(registry), "n_views": 0, "status": "audit_only"}
    root = Path(work_dir) / "se_wav"
    root.mkdir(parents=True, exist_ok=True)
    raw_items = sorted(registry.items())
    jobs: list[dict[str, Any]] = []
    output_by_raw: dict[str, Path] = {}
    for raw_hash, can in raw_items:
        dest = root / raw_hash[:2] / f"{raw_hash}.wav"
        output_by_raw[raw_hash] = dest
        if dest.is_file() and resume:
            continue
        if backend == "spectral":
            write_wav(dest, _spectral(can.wav, can.sample_rate), can.sample_rate)
        elif backend == "precomputed":
            if precomputed_dir is None:
                raise ValueError("precomputed backend requires --precomputed-dir")
            candidates = [Path(precomputed_dir) / f"{raw_hash}.wav"]
            for ref in refs:
                if ref["pcm_sha256"] == raw_hash:
                    candidates.extend([
                        Path(precomputed_dir) / ref["split"] / f"{ref['uid']}__{ref['role']}__{_safe(ref['arm'])}__{ref['stream']}.wav",
                        Path(precomputed_dir) / ref["split"] / f"{ref['uid']}_{ref['stream']}.wav",
                    ])
                    break
            source = next((p for p in candidates if p.is_file()), None)
            if source is None:
                raise FileNotFoundError(f"missing precomputed SE for raw_pcm={raw_hash}: {candidates}")
            data, sr = read_wav(source)
            write_wav(dest, data, sr)
        elif backend == "command":
            if batch_command:
                jobs.append({"raw_pcm_sha256": raw_hash, "input": can.source_wav, "output": str(dest), "sample_rate": can.sample_rate, "length_policy": "full_waveform"})
            else:
                if not command or "{input}" not in command or "{output}" not in command:
                    raise ValueError("command backend requires --se-command containing {input} and {output}, or --se-batch-command")
                argv = shlex.split(command.format(input=can.source_wav, output=str(dest)), posix=False)
                subprocess.run(argv, check=True)
        else:
            raise AssertionError(backend)
    if jobs:
        if "{manifest}" not in str(batch_command):
            raise ValueError("--se-batch-command must contain {manifest}")
        manifest = Path(work_dir) / "se_manifest.jsonl"
        write_jsonl(manifest, jobs)
        argv = shlex.split(str(batch_command).format(manifest=str(manifest)), posix=False)
        subprocess.run(argv, check=True)
    views: list[dict[str, Any]] = list(refs)
    se_hashes: set[str] = set()
    for raw_hash, can in raw_items:
        dest = output_by_raw[raw_hash]
        if not dest.is_file():
            raise RuntimeError(f"SE backend did not produce output: {dest}")
        after, sr = _validate(Path(can.source_wav), dest)
        shash = pcm_sha256(after, sr)
        se_hashes.add(shash)
        if shash not in registry:
            registry[shash] = InventoryCanonical(shash, file_sha256(dest), str(dest), after, sr, quality_metrics(after, sr))
        for raw in refs:
            if raw["pcm_sha256"] != raw_hash:
                continue
            row = dict(raw)
            row.update({
                "candidate_id": raw["candidate_id"].replace("-raw-", "-moss48k-"),
                "view": "moss48k", "source_wav": str(dest.resolve()),
                "raw_parent_candidate_id": raw["candidate_id"],
                "raw_parent_pcm_sha256": raw_hash,
                "file_sha256": file_sha256(dest), "pcm_sha256": shash, "canonical_id": shash,
                "se_backend": backend,
            })
            views.append(row)
    meta = {"backend": backend, "status": "complete", "n_unique_raw": len(raw_items), "n_views": len(views) - len(refs), "n_unique_se_pcm": len(se_hashes), "manifest": str(Path(work_dir) / "se_manifest.jsonl") if jobs else None}
    write_json(Path(work_dir) / "se_meta.json", meta)
    return views, meta
