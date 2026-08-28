from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .io import json_hash, read_json

WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".onnx", ".ckpt", ".params", ".index"}
CONFIG_SUFFIXES = {".json", ".txt", ".model", ".tokenizer"}


def file_sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_model_dir(path: str | Path | None) -> str | None:
    """Bind config + tokenizer + weight bytes. Missing dir returns None."""
    if path is None:
        return None
    root = Path(path)
    if not root.is_dir():
        return None
    files = [
        p for p in sorted(root.rglob("*"))
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in (WEIGHT_SUFFIXES | CONFIG_SUFFIXES)
    ]
    if not files:
        files = [p for p in sorted(root.glob("*")) if p.is_file() and not p.name.startswith(".")]
    digest = hashlib.sha256()
    for p in files:
        rel = p.relative_to(root).as_posix().encode("utf-8")
        digest.update(rel)
        digest.update(b"\0")
        digest.update(str(p.stat().st_size).encode())
        digest.update(b"\0")
        with p.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\n")
    return digest.hexdigest()


def freeze_signatures(
    *,
    s1_arm: str,
    s7_arm: str,
    extract_sep_run_id: str | None = None,
    asr_model_hash: str | None = None,
    asr_context_mode: str | None = None,
    asr_sidecar: str | Path | None = None,
    nll_sidecar: str | Path | None = None,
    qkw_sidecar: str | Path | None = None,
    embedding_sidecar: str | Path | None = None,
    noise_sidecar: str | Path | None = None,
    asr_command: str | None = None,
    nll_command: str | None = None,
    qkw_command: str | None = None,
    embedding_command: str | None = None,
    english_alias_table_hash: str | None = None,
    qkw_calibrator_hash: str | None = None,
    qkw_calibrated: bool = False,
    mossformer_model_hash: str | None = None,
    inference_signature: str | None = None,
    speaker_encoder_hash: str | None = None,
    noise_model_hashes: Any = None,
    route_policy: dict[str, Any] | None = None,
    policy_json: str | Path | None = None,
    se_backend: str | None = None,
    se_command: str | None = None,
    se_batch_command: str | None = None,
) -> dict[str, Any]:
    payload = {
        "extract_sep_run_id": extract_sep_run_id,
        "s1_arm": s1_arm,
        "s7_arm": s7_arm,
        "asr_model_hash": asr_model_hash,
        "asr_context_mode": asr_context_mode,
        "asr_sidecar_hash": file_sha256(asr_sidecar),
        "nll_sidecar_hash": file_sha256(nll_sidecar),
        "qkw_sidecar_hash": file_sha256(qkw_sidecar),
        "embedding_sidecar_hash": file_sha256(embedding_sidecar),
        "noise_sidecar_hash": file_sha256(noise_sidecar),
        "asr_command": asr_command,
        "nll_command": nll_command,
        "qkw_command": qkw_command,
        "embedding_command": embedding_command,
        "english_alias_table_hash": english_alias_table_hash,
        "qkw_calibrator_hash": qkw_calibrator_hash,
        "qkw_calibrated": bool(qkw_calibrated),
        "mossformer_model_hash": mossformer_model_hash,
        "inference_signature": inference_signature or se_backend,
        "speaker_encoder_hash": speaker_encoder_hash,
        "noise_model_hashes": noise_model_hashes,
        "se_backend": se_backend,
        "se_command": se_command,
        "se_batch_command": se_batch_command,
        "policy_json": str(Path(policy_json).resolve()) if policy_json else None,
        "policy_json_hash": file_sha256(policy_json),
        "route_policy": route_policy or {},
    }
    payload["route_policy_hash"] = json_hash(payload["route_policy"])
    payload["signature_hash"] = json_hash({k: payload[k] for k in payload if k != "signature_hash"})
    return payload


def assert_work_dir_signature(work_dir: str | Path, signatures: dict[str, Any]) -> None:
    path = Path(work_dir) / "signatures.json"
    if not path.is_file():
        return
    old = read_json(path)
    if old.get("signature_hash") != signatures.get("signature_hash"):
        raise RuntimeError(
            "work-dir signatures.json does not match this run; choose a new --work-dir "
            "so ASR/SE/embedding/noise results are not mixed"
        )
