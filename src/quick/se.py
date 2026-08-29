from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from .audio import file_sha256, pcm_sha256, quality_metrics, read_wav, write_wav
from .inventory import Canonical as InventoryCanonical
from .io import json_hash, read_json, write_json, write_jsonl


def _safe(value: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _split_cmd(template: str) -> list[str]:
    return shlex.split(template, posix=os.name != "nt")


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
    dur0 = before.size / max(sr0, 1)
    dur1 = after.size / max(sr1, 1)
    tol = max(0.1, dur0 * 0.02)
    sample_tol = max(int(round(0.1 * sr0)), int(round(before.size * 0.02)))
    if abs(dur1 - dur0) > tol or abs(after.size - before.size * sr1 / max(sr0, 1)) > sample_tol:
        raise RuntimeError(f"SE changed duration beyond tolerance: {src} -> {dest}")
    return after, sr1


def _load_raw(can: InventoryCanonical) -> tuple[np.ndarray, int]:
    if can.wav is not None:
        return np.asarray(can.wav, dtype=np.float32), int(can.sample_rate)
    return read_wav(can.source_wav)


def add_se_views(
    refs: list[dict[str, Any]],
    registry: dict[str, InventoryCanonical],
    *,
    work_dir: str | Path,
    cache_root: str | Path | None = None,
    backend: str,
    command: str | None = None,
    batch_command: str | None = None,
    precomputed_dir: str | Path | None = None,
    resume: bool = True,
    mossformer_model_hash: str | None = None,
    inference_signature: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    backend = str(backend).lower().strip()
    if backend not in {"none", "spectral", "command", "precomputed"}:
        raise ValueError(f"unsupported SE backend {backend!r}")
    if backend == "none":
        return list(refs), {"backend": backend, "n_unique_raw": len(registry), "n_views": 0, "status": "audit_only", "n_se_failures": 0}
    transform_signature = {
        "backend": backend,
        "command": command,
        "batch_command": batch_command,
        "precomputed_dir": str(Path(precomputed_dir).resolve()) if precomputed_dir else None,
        "mossformer_model_hash": mossformer_model_hash,
        "inference_signature": inference_signature or backend,
        "output_contract": "full_waveform_wav/v1",
    }
    transform_hash = json_hash(transform_signature)
    base = Path(cache_root).resolve() if cache_root is not None else Path(work_dir).resolve()
    root = base / "se48k" / transform_hash
    root.mkdir(parents=True, exist_ok=True)
    signature = {
        "transform": transform_signature,
        "transform_hash": transform_hash,
        "raw_pcm": sorted(registry),
    }
    meta_path = Path(work_dir) / "se_cache_binding.json"
    if meta_path.is_file():
        old = read_json(meta_path)
        if old.get("signature_hash") and old["signature_hash"] != json_hash(signature):
            raise RuntimeError("SE binding signature mismatch; use a new --work-dir so datasets are not mixed")
    write_json(meta_path, {"status": "running", "signature": signature, "signature_hash": json_hash(signature)})

    raw_ok = [(h, c) for h, c in sorted(registry.items()) if not str(h).startswith("undecodable:")]
    jobs: list[dict[str, Any]] = []
    output_by_raw: dict[str, Path] = {}
    failures: dict[str, str] = {}
    precomputed_index: dict[str, Path] = {}
    cache_hits = 0
    cache_misses = 0
    if backend == "precomputed" and precomputed_dir is not None:
        pre_root = Path(precomputed_dir)
        if pre_root.is_dir():
            # Previous quick runs use se_wav/<hash[:2]>/<hash>.wav; older
            # extract-main runs commonly flatten or nest the same hash.  Build
            # one bounded index so a full run does not perform O(N) rglob per
            # candidate.
            for p in pre_root.rglob("*.wav"):
                stem = p.stem.lower()
                if len(stem) >= 32 and all(ch in "0123456789abcdef" for ch in stem):
                    precomputed_index.setdefault(stem, p)
    for raw_hash, can in raw_ok:
        dest = root / raw_hash[:2] / f"{raw_hash}.wav"
        output_by_raw[raw_hash] = dest
        if dest.is_file() and resume:
            cache_hits += 1
            continue
        cache_misses += 1
        try:
            if backend == "spectral":
                wav, sr = _load_raw(can)
                write_wav(dest, _spectral(wav, sr), sr)
            elif backend == "precomputed":
                if precomputed_dir is None:
                    raise ValueError("precomputed backend requires --precomputed-dir")
                candidates = [Path(precomputed_dir) / f"{raw_hash}.wav"]
                indexed = precomputed_index.get(str(raw_hash).lower())
                if indexed is not None:
                    candidates.insert(0, indexed)
                for ref in refs:
                    if ref.get("pcm_sha256") == raw_hash and ref.get("source_wav"):
                        candidates.extend([
                            Path(precomputed_dir) / ref["split"] / f"{ref['uid']}__{ref['role']}__{_safe(ref['arm'])}__{ref['stream']}.wav",
                            Path(precomputed_dir) / ref["split"] / f"{ref['uid']}_{ref['stream']}.wav",
                        ])
                        break
                source = next((p for p in candidates if p.is_file()), None)
                if source is None:
                    raise FileNotFoundError(f"missing precomputed SE for raw_pcm={raw_hash}")
                data, sr = read_wav(source)
                write_wav(dest, data, sr)
            elif backend == "command":
                dest.parent.mkdir(parents=True, exist_ok=True)
                if batch_command:
                    jobs.append({
                        "raw_pcm_sha256": raw_hash, "input": can.source_wav, "output": str(dest),
                        "sample_rate": can.sample_rate, "length_policy": "full_waveform",
                    })
                else:
                    if not command or "{input}" not in command or "{output}" not in command:
                        raise ValueError("command backend requires --se-command containing {input} and {output}, or --se-batch-command")
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    argv = _split_cmd(command.format(input=can.source_wav, output=str(dest)))
                    subprocess.run(argv, check=True)
            else:
                raise AssertionError(backend)
        except Exception as exc:
            failures[raw_hash] = f"{type(exc).__name__}: {exc}"
    if jobs:
        if "{manifest}" not in str(batch_command):
            raise ValueError("--se-batch-command must contain {manifest}")
        pending = [j for j in jobs if j["raw_pcm_sha256"] not in failures]
        if pending:
            manifest = Path(work_dir) / "se_manifest.jsonl"
            write_jsonl(manifest, pending)
            argv = _split_cmd(str(batch_command).format(manifest=str(manifest)))
            try:
                subprocess.run(argv, check=True)
            except Exception as exc:
                for job in pending:
                    failures[job["raw_pcm_sha256"]] = f"batch_failed: {type(exc).__name__}: {exc}"

    views: list[dict[str, Any]] = list(refs)
    se_hashes: set[str] = set()
    n_fail_refs = 0
    refs_by_pcm: dict[str, list[dict[str, Any]]] = {}
    for raw in refs:
        if raw.get("view") != "raw":
            continue
        pcm = raw.get("pcm_sha256")
        if not pcm:
            continue
        refs_by_pcm.setdefault(str(pcm), []).append(raw)
    for raw_hash, can in raw_ok:
        dest = output_by_raw[raw_hash]
        se_error = failures.get(raw_hash)
        after = None
        sr = can.sample_rate
        shash = None
        if se_error is None:
            if not dest.is_file():
                se_error = f"SE backend did not produce output: {dest}"
            else:
                try:
                    after, sr = _validate(Path(can.source_wav), dest)
                    shash = pcm_sha256(after, sr)
                    se_hashes.add(shash)
                    if shash not in registry:
                        registry[shash] = InventoryCanonical(shash, file_sha256(dest), str(dest.resolve()), sr, quality_metrics(after, sr), wav=None)
                except Exception as exc:
                    se_error = f"{type(exc).__name__}: {exc}"
        for raw in refs_by_pcm.get(raw_hash, []):
            row = dict(raw)
            row.update({
                "candidate_id": str(raw["candidate_id"]).replace("-raw-", "-moss48k-"),
                "view": "moss48k",
                "raw_parent_candidate_id": raw["candidate_id"],
                "raw_parent_pcm_sha256": raw_hash,
                "se_backend": backend,
                "se_model_hash": mossformer_model_hash,
                "inference_mode": "full_waveform",
            })
            if se_error:
                row.update({
                    "source_wav": None, "file_sha256": None, "pcm_sha256": None,
                    "canonical_id": None, "validity": "fatal_invalid",
                    "fatal_reason": se_error, "decode_ok": False, "export_placeholder": True,
                })
                n_fail_refs += 1
            else:
                row.update({
                    "source_wav": str(dest.resolve()),
                    "file_sha256": file_sha256(dest),
                    "pcm_sha256": shash,
                    "canonical_id": shash,
                    "validity": "pending",
                    "decode_ok": True,
                    "fatal_reason": None,
                })
            views.append(row)
    for raw in refs:
        if raw.get("view") != "raw":
            continue
        pcm = raw.get("pcm_sha256")
        if pcm in output_by_raw:
            continue
        row = dict(raw)
        row.update({
            "candidate_id": str(raw["candidate_id"]).replace("-raw-", "-moss48k-"),
            "view": "moss48k",
            "raw_parent_candidate_id": raw["candidate_id"],
            "raw_parent_pcm_sha256": pcm,
            "se_backend": backend,
            "source_wav": None, "file_sha256": None, "pcm_sha256": None,
            "canonical_id": None, "validity": "fatal_invalid",
            "fatal_reason": raw.get("fatal_reason") or "raw_undecodable_skip_se",
            "decode_ok": False, "export_placeholder": True,
            "se_model_hash": mossformer_model_hash,
            "inference_mode": "full_waveform",
        })
        views.append(row)
        n_fail_refs += 1
    meta = {
        "backend": backend, "status": "complete",
        "n_unique_raw": len(raw_ok), "n_views": len(views) - len(refs),
        "n_unique_se_pcm": len(se_hashes), "n_se_failures": len(failures),
        "n_failed_se_refs": n_fail_refs,
        "failures": failures,
        "manifest": str(Path(work_dir) / "se_manifest.jsonl") if jobs else None,
        "signature_hash": json_hash(signature),
        "transform_hash": transform_hash,
        "cache_root": str(root),
        "n_cache_hit": cache_hits,
        "n_cache_miss": cache_misses,
        "n_fresh": cache_misses,
        "mossformer_model_hash": mossformer_model_hash,
        "inference_signature": inference_signature or backend,
    }
    write_json(Path(work_dir) / "se_meta.json", meta)
    write_json(meta_path, {"status": "complete", "signature": signature, "signature_hash": json_hash(signature), **{k: meta[k] for k in ("n_unique_raw", "n_unique_se_pcm", "n_se_failures")}})
    return views, meta
