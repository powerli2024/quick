import json
import sys
from pathlib import Path

from quick.export import group_orders, safe_slug, uid_hash
from quick.pipeline import RunConfig, _render_command, run
from quick.validate import validate_review_flat

from helpers import make_pos_neg


def test_group_prefix_is_stable():
    rows = [
        {"group_key": "neg\0neg_2"},
        {"group_key": "pos\0pos_0001"},
        {"group_key": "pos\0pos_0002"},
    ]
    orders = group_orders(rows)
    assert orders["pos\0pos_0001"] == 1
    assert orders["pos\0pos_0002"] == 2
    assert orders["neg\0neg_2"] == 3
    assert uid_hash("pos", "pos_0001") == uid_hash("pos", "pos_0001")
    assert safe_slug("neg_1005cer") == "neg_1005cer"


def test_sidecar_template_preserves_literal_braces(tmp_path: Path):
    template = "python -c \"print('{literal}')\" --manifest {manifest} --output {output}"
    rendered = _render_command(
        template,
        manifest=tmp_path / "manifest.jsonl",
        output=tmp_path / "out.jsonl",
    )
    assert "{literal}" in rendered
    assert "{manifest}" not in rendered
    assert "{output}" not in rendered


def test_spectral_pipeline_never_production_approved(tmp_path: Path):
    """Spectral is plumbing/control only; local export may succeed, production must not."""
    pos_neg = make_pos_neg(tmp_path / "sep")
    work = tmp_path / "work"
    fake = Path(__file__).resolve().parent / "fake_asr.py"
    # Deliberately weak external JSON without provenance — must not unlock PASS.
    cmd = json.dumps({"baseline": {"frr": 0.10, "far": 0.10}, "candidate": {"frr": 0.08, "far": 0.10}})
    presence = json.dumps({"baseline": {"frr": 0.12, "far": 0.09}, "candidate": {"frr": 0.11, "far": 0.09}})
    contest = json.dumps({"baseline": {"score": 0.80}, "candidate": {"score": 0.81}})
    (tmp_path / "cmd.json").write_text(cmd, encoding="utf-8")
    (tmp_path / "presence.json").write_text(presence, encoding="utf-8")
    (tmp_path / "contest.json").write_text(contest, encoding="utf-8")
    report = run(RunConfig(
        pos_neg=pos_neg,
        s1_arm="s1_onnx_full",
        s7_arm="s7_cv_then_onnx_gate/thr_a",
        work_dir=work,
        expected_uids=3,
        se_backend="spectral",
        asr_command=f"{sys.executable} {fake} --manifest {{manifest}} --output {{output}}",
        selected_only_dir=work / "best_sep_selected",
        cmd_result_json=tmp_path / "cmd.json",
        presence_result_json=tmp_path / "presence.json",
        contest_result_json=tmp_path / "contest.json",
        asr_context_mode="wake",
        mossformer_model_hash=None,
        policy_json=Path(__file__).resolve().parents[1] / "configs" / "route_policy.json",
    ))
    assert report["production_approved"] is False
    assert report["status"] != "PASS"
    local = report["phases"]["I4_I5_local"]
    assert local["local_pass"] is True
    assert local["n_selected_finite_cer"] == local["n_baseline_finite_cer"] == local["n_paired"] == 3
    flat = Path(report["paths"]["flat_dir"])
    validate_review_flat(flat, expected_groups=3)
    names = sorted(p.name for p in flat.iterdir() if p.suffix == ".wav")
    groups = {}
    for name in names:
        prefix = "__".join(name.split("__")[:3])
        groups.setdefault(prefix, []).append(name)
    for files in groups.values():
        assert "__0000__SELECTED__" in files[0]
    reasons = list(flat.glob("*__9000__ROUTE_REASON.json"))
    assert len(reasons) == 3
    selected = work / "best_sep_selected"
    assert (selected / "pos" / "pos_0001.wav").is_file()
    assert (selected / "index.jsonl").is_file()
    codes = {json.loads(p.read_text(encoding="utf-8"))["decision"]["reason_code"] for p in reasons}
    assert "KEEP_S1_CER0_TRUSTED" in codes
    assert "SWITCH_S7_LOWER_CER" in codes
    assert report["phases"]["I8"]["reason"] in {
        "spectral_or_non_production_se_forbidden",
        "cmd_schema_invalid",
        "i8_coverage_missing",
    }


def test_qkw_calibrated_without_hash_raises(tmp_path: Path):
    pos_neg = make_pos_neg(tmp_path / "sep")
    fake = Path(__file__).resolve().parent / "fake_asr.py"
    try:
        run(RunConfig(
            pos_neg=pos_neg,
            s1_arm="s1_onnx_full",
            s7_arm="s7_cv_then_onnx_gate/thr_a",
            work_dir=tmp_path / "work2",
            expected_uids=3,
            se_backend="spectral",
            asr_command=f"{sys.executable} {fake} --manifest {{manifest}} --output {{output}}",
            qkw_calibrated=True,
            qkw_calibrator_hash=None,
        ))
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "qkw-calibrator-hash" in str(exc)
