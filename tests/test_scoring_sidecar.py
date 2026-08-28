from pathlib import Path

from quick.io import write_jsonl
from quick.scoring import find_sidecar, load_sidecar, score_rows


def test_duplicate_pcm_sidecar_rows_do_not_conflict(tmp_path: Path):
    pcm = "abc123"
    path = tmp_path / "asr.jsonl"
    write_jsonl(path, [
        {"candidate_id": "C-pos-u-s1-raw-spk1", "pcm_sha256": pcm, "wake_text": "hicolmo", "lang": "en", "hyp": "hicolmo"},
        {"candidate_id": "C-pos-u-s7-raw-spk1", "pcm_sha256": pcm, "wake_text": "hicolmo", "lang": "en", "hyp": "hicolmo"},
        {"candidate_id": "C-pos-u-s1-moss48k-spk1", "pcm_sha256": pcm, "wake_text": "hicolmo", "lang": "en", "hyp": "hicolmo"},
    ])
    side = load_sidecar(path)
    hit = find_sidecar(side, {
        "candidate_id": "C-pos-u-s7-moss48k-spk1",
        "pcm_sha256": pcm,
        "wake_text": "hicolmo",
        "lang": "en",
    })
    assert hit is not None
    assert hit["hyp"] == "hicolmo"


def test_sidecar_bind_rejects_pcm_mismatch(tmp_path: Path):
    path = tmp_path / "asr.jsonl"
    write_jsonl(path, [
        {"candidate_id": "C-a", "pcm_sha256": "pcmA", "wake_text": "hicolmo", "lang": "en", "hyp": "hicolmo"},
    ])
    side = load_sidecar(path)
    try:
        find_sidecar(side, {"candidate_id": "C-a", "pcm_sha256": "pcmB", "wake_text": "hicolmo", "lang": "en"})
        assert False, "expected pcm mismatch"
    except ValueError as exc:
        assert "pcm mismatch" in str(exc)


def test_score_rows_reuse_same_pcm_across_roles(tmp_path: Path):
    from quick.inventory import Canonical
    from quick.audio import quality_metrics
    import numpy as np

    pcm = "samepcm"
    wav = np.zeros(1600, dtype=np.float32)
    wav[100:400] = 0.2
    registry = {pcm: Canonical(pcm, "f", str(tmp_path / "x.wav"), 16000, quality_metrics(wav, 16000), wav=None)}
    asr = tmp_path / "asr.jsonl"
    write_jsonl(asr, [
        {"pcm_sha256": pcm, "wake_text": "hicolmo", "lang": "en", "hyp": "heycolmo"},
    ])
    rows = [
        {"candidate_id": "C-pos-u-s1-raw-spk1", "group_key": "pos\0u", "uid": "u", "split": "pos",
         "wake_text": "hicolmo", "lang": "en", "role": "s1", "arm": "s1", "stream": "spk1",
         "view": "raw", "pcm_sha256": pcm, "canonical_id": pcm, "validity": "pending"},
        {"candidate_id": "C-pos-u-s7-raw-spk1", "group_key": "pos\0u", "uid": "u", "split": "pos",
         "wake_text": "hicolmo", "lang": "en", "role": "s7", "arm": "s7", "stream": "spk1",
         "view": "raw", "pcm_sha256": pcm, "canonical_id": pcm, "validity": "pending"},
    ]
    scored, meta = score_rows(rows, registry, asr_sidecar=asr)
    assert len(scored) == 2
    assert meta["n_unique_score_key"] == 1
    assert scored[0]["cer_route"] == scored[1]["cer_route"] == 0.0
