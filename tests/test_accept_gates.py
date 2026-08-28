from quick.accept import i8_validate, local_evaluate
from quick.route import RoutePolicy


def _cand(uid: str, cer: float | None, *, role: str = "s1", view: str = "raw", stream: str = "original", pcm: str = "p"):
    row = {
        "candidate_id": f"C-pos-{uid}-{role}-{view}-{stream}",
        "group_key": f"pos\0{uid}",
        "uid": uid,
        "split": "pos",
        "role": role,
        "view": view,
        "stream": stream,
        "validity": "rankable" if cer is not None else "fatal_invalid",
        "cer_route": cer,
        "nll": 1.0,
        "q_kw": None,
        "qkw_calibrated": False,
        "pcm_sha256": pcm,
        "content_class": "target_wake",
        "audio_quality": {"clip_rate": 0.0, "speech_ratio": 0.8, "p_overlap": 0.0},
        "extra_ratio": 0.0,
        "s7_available": True,
        "decode_ok": cer is not None,
        "source_wav": f"/tmp/{uid}.wav",
        "core_hit": True,
        "lang": "en",
        "wake_text": "hicolmo",
    }
    return row


def test_missing_cer_blocks_local_pass():
    groups = {
        "pos\0a": [_cand("a", 0.0)],
        "pos\0b": [_cand("b", None)],
    }
    local, decisions = local_evaluate(groups, RoutePolicy(), expected=2, cer_mean_max=0.03, cer0_drop_max=0.02)
    assert local["n_selected_finite_cer"] == 1
    assert local["n_baseline_finite_cer"] == 1
    assert local["local_pass"] is False
    assert local["checks"]["finite_cer_coverage"] is False
    assert any(not d.get("ok") or d.get("has_finite_cer") is False or d.get("reason_code") == "FAIL_NO_DECODABLE_S1" for d in decisions)


def test_spectral_cannot_production_approve():
    local = {
        "local_pass": True,
        "expected_uid": 3,
        "uid_fingerprint": "fp",
        "n_audit_fallback": 0,
    }
    i8 = i8_validate(
        local,
        {"schema": "kws_i8_eval/v1", "kind": "cmd"},
        {"schema": "kws_i8_eval/v1", "kind": "presence"},
        {"schema": "kws_i8_eval/v1", "kind": "contest"},
        se_backend="spectral",
    )
    assert i8["production_approved"] is False
    assert i8["status"] != "PASS"
    assert "spectral" in str(i8.get("reason"))


def test_bare_frr_json_cannot_pass_without_bindings():
    local = {
        "local_pass": True,
        "expected_uid": 3,
        "uid_fingerprint": "fp",
        "n_audit_fallback": 0,
    }
    cmd = {"baseline": {"frr": 0.1, "far": 0.1}, "candidate": {"frr": 0.05, "far": 0.1}}
    presence = {"baseline": {"frr": 0.1, "far": 0.1}, "candidate": {"frr": 0.05, "far": 0.1}}
    contest = {"baseline": {"score": 0.8}, "candidate": {"score": 0.9}}
    i8 = i8_validate(
        local, cmd, presence, contest,
        bindings={
            "signature_hash": "sig", "selected_index_hash": "idx", "s1_arm": "s1",
            "s7_arm": "s7", "se_backend": "command", "asr_model_hash": "asr",
            "route_policy_hash": "pol", "mossformer_model_hash": "moss",
        },
        se_backend="command",
    )
    assert i8["production_approved"] is False
    assert i8["status"] == "NO_GO"


def test_qkw_calibrated_requires_hash():
    local = {"local_pass": True, "expected_uid": 1, "uid_fingerprint": "fp", "n_audit_fallback": 0}
    i8 = i8_validate(local, None, None, None, se_backend="command", qkw_calibrated=True, qkw_calibrator_hash=None)
    assert i8["status"] == "NO_GO"
    assert i8["reason"] == "qkw_calibrated_without_calibrator_hash"
