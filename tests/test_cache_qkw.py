from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from helpers import make_pos_neg
from quick.audio import quality_metrics
from quick.inventory import Canonical
from quick.io import write_json, write_jsonl
from quick.pipeline import RunConfig, run
from quick.qkw import SCHEMA, apply_model, calibrator_hash, fit_logistic, load_calibrator
from quick.scoring import score_key, score_rows


def test_audio_cache_reused_across_work_dirs(tmp_path: Path):
    pos_neg = make_pos_neg(tmp_path / "sep")
    cache = tmp_path / "fixed_audio_cache"
    fake = Path(__file__).resolve().parent / "fake_asr.py"

    reports = []
    for name in ("run1", "run2"):
        reports.append(run(RunConfig(
            pos_neg=pos_neg,
            s1_arm="s1_onnx_full",
            s7_arm="s7_cv_then_onnx_gate/thr_a",
            work_dir=tmp_path / name,
            audio_cache_root=cache,
            expected_uids=3,
            se_backend="spectral",
            asr_command=f"{sys.executable} {fake} --manifest {{manifest}} --output {{output}}",
        )))

    first_inventory = reports[0]["phases"]["I1_inventory"]["audio_cache"]
    second_inventory = reports[1]["phases"]["I1_inventory"]["audio_cache"]
    first_se = reports[0]["phases"]["I2_se"]
    second_se = reports[1]["phases"]["I2_se"]
    assert first_inventory["n_cache_miss"] > 0
    assert second_inventory["n_cache_hit"] == second_inventory["n_unique"]
    assert first_se["n_cache_miss"] > 0
    assert second_se["n_cache_hit"] == second_se["n_unique_raw"]
    assert reports[0]["paths"]["audio_cache_root"] == reports[1]["paths"]["audio_cache_root"] == str(cache.resolve())
    assert Path(reports[1]["paths"]["sep_audio_cache"]).is_dir()
    assert Path(reports[1]["paths"]["se_audio_cache"]).is_dir()


def test_same_work_dir_resume_uses_input_signature(tmp_path: Path):
    pos_neg = make_pos_neg(tmp_path / "sep")
    fake = Path(__file__).resolve().parent / "fake_asr.py"
    cfg = RunConfig(
        pos_neg=pos_neg,
        s1_arm="s1_onnx_full",
        s7_arm="s7_cv_then_onnx_gate/thr_a",
        work_dir=tmp_path / "work",
        audio_cache_root=tmp_path / "cache",
        expected_uids=3,
        se_backend="spectral",
        asr_command=f"{sys.executable} {fake} --manifest {{manifest}} --output {{output}}",
    )
    run(cfg)
    report = run(cfg)
    assert report["phases"]["I1_inventory"]["audio_cache"]["n_cache_hit"] > 0
    assert report["phases"]["I2_se"]["n_cache_hit"] > 0


def _one_candidate(tmp_path: Path):
    pcm = "pcm-one"
    wav = np.zeros(1600, dtype=np.float32)
    wav[100:500] = 0.1
    registry = {pcm: Canonical(pcm, "file", str(tmp_path / "x.wav"), 16000, quality_metrics(wav, 16000))}
    row = {
        "candidate_id": "C-pos-u-s1-raw-original", "group_key": "pos\0u", "uid": "u", "split": "pos",
        "wake_text": "hicolmo", "lang": "en", "role": "s1", "arm": "s1", "stream": "original",
        "view": "raw", "pcm_sha256": pcm, "canonical_id": pcm, "validity": "pending",
    }
    return registry, row


def test_calibrated_qkw_requires_full_signed_coverage(tmp_path: Path):
    registry, row = _one_candidate(tmp_path)
    sk = score_key(row)
    asr = tmp_path / "asr.jsonl"
    write_jsonl(asr, [{"score_key": sk, "pcm_sha256": row["pcm_sha256"], "wake_text": "hicolmo", "lang": "en", "hyp": "hicolmo"}])

    bad = tmp_path / "bad_qkw.jsonl"
    write_jsonl(bad, [{"score_key": sk, "pcm_sha256": row["pcm_sha256"], "wake_text": "hicolmo", "lang": "en", "q_kw": 0.9, "score_kind": "calibrated_qkw", "qkw_calibrator_hash": "wrong"}])
    try:
        score_rows([row], registry, asr_sidecar=asr, qkw_sidecar=bad, qkw_calibrated=True, qkw_calibrator_hash="frozen")
        assert False, "expected signed q_kw rejection"
    except RuntimeError as exc:
        assert "calibrated q_kw contract failed" in str(exc)

    good = tmp_path / "good_qkw.jsonl"
    write_jsonl(good, [{"score_key": sk, "pcm_sha256": row["pcm_sha256"], "wake_text": "hicolmo", "lang": "en", "q_kw": 0.9, "score_kind": "calibrated_qkw", "qkw_calibrator_hash": "frozen"}])
    scored, meta = score_rows([row], registry, asr_sidecar=asr, qkw_sidecar=good, qkw_calibrated=True, qkw_calibrator_hash="frozen")
    assert scored[0]["q_kw"] == 0.9
    assert scored[0]["qkw_calibrated"] is True
    assert meta["qkw_coverage"] == 1.0
    assert meta["qkw_valid"] == meta["qkw_expected"] == 1


def test_unsigned_qkw_never_enters_route(tmp_path: Path):
    registry, row = _one_candidate(tmp_path)
    sk = score_key(row)
    asr = tmp_path / "asr.jsonl"
    qkw = tmp_path / "qkw.jsonl"
    write_jsonl(asr, [{"score_key": sk, "hyp": "hicolmo"}])
    write_jsonl(qkw, [{"score_key": sk, "q_kw": 0.99, "score_kind": "calibrated_qkw", "qkw_calibrator_hash": "self-declared"}])
    scored, meta = score_rows([row], registry, asr_sidecar=asr, qkw_sidecar=qkw, qkw_calibrated=False)
    assert scored[0]["q_kw"] is None
    assert scored[0]["qkw_calibrated"] is False
    assert meta["qkw_coverage"] is None


def test_logistic_qkw_is_bounded_and_monotonic():
    model = fit_logistic([-2, -1, -0.5, 0.5, 1, 2], [0, 0, 0, 1, 1, 1])
    low = apply_model(model, -1.0)
    high = apply_model(model, 1.0)
    assert 0.0 <= low < high <= 1.0


def test_qkw_calibrator_artifact_hash_is_verified(tmp_path: Path):
    model = fit_logistic([-2, -1, 1, 2], [0, 0, 1, 1])
    payload = {
        "schema": SCHEMA,
        "score_field": "ctc_align_score",
        "score_direction": "higher_is_better",
        "source_sha256": "dev-set-hash",
        "independent_dev_required": True,
        "models": {"en": model},
    }
    payload["calibrator_hash"] = calibrator_hash(payload)
    path = tmp_path / "calibrator.json"
    write_json(path, payload)
    _, digest = load_calibrator(path)
    assert digest == payload["calibrator_hash"]
    payload["models"]["en"]["intercept"] += 1
    write_json(path, payload)
    try:
        load_calibrator(path)
        assert False, "expected tampered calibrator rejection"
    except ValueError as exc:
        assert "hash mismatch" in str(exc)
