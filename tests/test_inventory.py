from pathlib import Path

import pytest

from quick.inventory import build_inventory, discover_arms
from helpers import make_pos_neg


def test_auto_arm_forbidden(tmp_path: Path):
    root = make_pos_neg(tmp_path / "sep")
    arms = discover_arms(root, "pos")
    assert "s1_onnx_full" in arms
    assert "s7_cv_then_onnx_gate/thr_a" in arms
    with pytest.raises(ValueError, match="auto is forbidden"):
        build_inventory(root, s1_arm="s1_onnx_full", s7_arm="auto")


def test_peak_maps_to_original(tmp_path: Path):
    root = make_pos_neg(tmp_path / "sep")
    refs, registry, meta = build_inventory(root, s1_arm="s1_onnx_full", s7_arm="s7_cv_then_onnx_gate/thr_a")
    streams = {r["stream"] for r in refs if r["uid"] == "pos_0001" and r["role"] == "s1"}
    assert streams == {"original", "spk1"}
    assert meta["n_uid"] == 3
    assert all(r["s7_available"] for r in refs)
    assert len(registry) >= 1
