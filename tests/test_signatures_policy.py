import json
from pathlib import Path

from quick.policy import build_route_policy, load_policy_file
from quick.signatures import freeze_signatures, hash_model_dir


def test_policy_json_changes_margins(tmp_path: Path):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({
        "nll_switch_margin": 0.25,
        "qkw_switch_margin": 0.5,
        "acceptance": {"mean_cer_max": 0.01},
    }), encoding="utf-8")
    data = load_policy_file(path)
    policy = build_route_policy(data)
    assert policy.nll_switch_margin == 0.25
    assert policy.qkw_switch_margin == 0.5
    overridden = build_route_policy(data, overrides={"nll_switch_margin": 0.03})
    assert overridden.nll_switch_margin == 0.03


def test_hash_model_dir_includes_weight_bytes(tmp_path: Path):
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text('{"a":1}', encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"weight-a")
    h1 = hash_model_dir(root)
    (root / "model.safetensors").write_bytes(b"weight-b")
    h2 = hash_model_dir(root)
    assert h1 and h2 and h1 != h2


def test_signatures_include_sidecar_hashes(tmp_path: Path):
    asr = tmp_path / "asr.jsonl"
    asr.write_text('{"hyp":"x"}\n', encoding="utf-8")
    nll = tmp_path / "nll.jsonl"
    nll.write_text('{"nll":1.0}\n', encoding="utf-8")
    sig = freeze_signatures(
        s1_arm="s1_onnx_full",
        s7_arm="s7_x",
        asr_sidecar=asr,
        nll_sidecar=nll,
        qkw_calibrated=True,
        qkw_calibrator_hash="calhash",
        se_backend="command",
        mossformer_model_hash="moss",
        route_policy={"nll_switch_margin": 0.01},
    )
    assert sig["asr_sidecar_hash"]
    assert sig["nll_sidecar_hash"]
    assert sig["qkw_calibrator_hash"] == "calhash"
    assert sig["route_policy_hash"]
