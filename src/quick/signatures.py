from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .io import json_hash, read_json


def _file_hash(path: Path | None) -> str | None:
    if path is None or not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_signatures(
    *,
    s1_arm: str,
    s7_arm: str,
    extract_sep_run_id: str | None = None,
    asr_model_hash: str | None = None,
    asr_context_mode: str | None = None,
    asr_sidecar: str | Path | None = None,
    english_alias_table_hash: str | None = None,
    qkw_calibrator_hash: str | None = None,
    mossformer_model_hash: str | None = None,
    inference_signature: str | None = None,
    speaker_encoder_hash: str | None = None,
    noise_model_hashes: Any = None,
    route_policy: dict[str, Any] | None = None,
    se_backend: str | None = None,
    se_command: str | None = None,
    se_batch_command: str | None = None,
) -> dict[str, Any]:
    payload = {
        "extract_sep_run_id": extract_sep_run_id,
        "s1_arm": s1_arm,
        "s7_arm": s7_arm,
        "asr_model_hash": asr_model_hash or _file_hash(Path(asr_sidecar) if asr_sidecar else None),
        "asr_context_mode": asr_context_mode,
        "english_alias_table_hash": english_alias_table_hash,
        "qkw_calibrator_hash": qkw_calibrator_hash,
        "mossformer_model_hash": mossformer_model_hash,
        "inference_signature": inference_signature or se_backend,
        "speaker_encoder_hash": speaker_encoder_hash,
        "noise_model_hashes": noise_model_hashes,
        "se_backend": se_backend,
        "se_command": se_command,
        "se_batch_command": se_batch_command,
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
