import json
import sys
from pathlib import Path

from quick.export import group_orders, safe_slug, uid_hash
from quick.pipeline import RunConfig, run
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


def test_full_spectral_pipeline_flat_export(tmp_path: Path):
    pos_neg = make_pos_neg(tmp_path / "sep")
    work = tmp_path / "work"
    fake = Path(__file__).resolve().parent / "fake_asr.py"
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
    ))
    assert report["status"] == "PASS"
    assert report["production_approved"] is True
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
    payload = json.loads(reasons[0].read_text(encoding="utf-8"))
    assert payload["schema"] == "kws_s1_s7_moss_route_reason/v1"
    assert payload["audit"]["exactly_one_selected"] is True
    selected = work / "best_sep_selected"
    assert (selected / "pos" / "pos_0001.wav").is_file()
    assert (selected / "index.jsonl").is_file()
    local = report["phases"]["I4_I5_local"]
    assert local["n_uid"] == 3
    assert local["n_switched_s7"] >= 1
    codes = {json.loads(p.read_text(encoding="utf-8"))["decision"]["reason_code"] for p in reasons}
    assert "KEEP_S1_CER0_TRUSTED" in codes
    assert "SWITCH_S7_LOWER_CER" in codes
    summary = json.loads((flat / "ZZZZZZ__EXPORT_SUMMARY.json").read_text(encoding="utf-8"))
    assert summary["n_groups"] == 3
    assert summary["s7"]["n_switch"] >= 1
