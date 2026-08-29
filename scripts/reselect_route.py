#!/usr/bin/env python3
"""Re-run only scoring/routing/export after all sidecars are available.

This command never reruns separation, SE or ASR.  It rebuilds the lightweight
canonical registry from ``candidate_refs_with_se.jsonl`` and applies the
current sidecars and route policy into a new output directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quick.accept import local_evaluate  # noqa: E402
from quick.audio import quality_metrics, read_wav  # noqa: E402
from quick.export import export_flat  # noqa: E402
from quick.inventory import Canonical  # noqa: E402
from quick.io import read_json, read_jsonl, write_json, write_jsonl  # noqa: E402
from quick.cer import load_alias_table  # noqa: E402
from quick.policy import build_route_policy, load_policy_file  # noqa: E402
from quick.scoring import score_rows  # noqa: E402
from quick.signatures import file_sha256, hash_model_dir  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="reselect/export from completed sidecars")
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--asr-jsonl", type=Path, default=None)
    p.add_argument("--nll-jsonl", type=Path, default=None)
    p.add_argument("--qkw-jsonl", type=Path, default=None)
    p.add_argument("--embedding-jsonl", type=Path, default=None)
    p.add_argument("--noise-jsonl", type=Path, default=None)
    p.add_argument("--alias-json", type=Path, default=None)
    p.add_argument("--policy-json", type=Path, default=None)
    p.add_argument("--qkw-calibrated", action="store_true")
    p.add_argument("--qkw-calibrator-json", type=Path, default=None)
    p.add_argument("--expected-uids", type=int, default=0)
    args = p.parse_args()
    work, out = args.work_dir.resolve(), args.output_dir.resolve()
    refs_path = work / "candidate_refs_with_se.jsonl"
    if not refs_path.is_file():
        raise SystemExit(f"missing completed candidate refs: {refs_path}")
    rows = read_jsonl(refs_path)
    registry: dict[str, Canonical] = {}
    for row in rows:
        pcm = str(row.get("canonical_id") or row.get("pcm_sha256") or "")
        source = row.get("source_wav")
        if not pcm or not source or not Path(source).is_file() or pcm in registry:
            continue
        wav, sr = read_wav(source)
        registry[pcm] = Canonical(pcm, file_sha256(source), str(Path(source).resolve()), sr, quality_metrics(wav, sr), wav=None)
    aliases = load_alias_table(args.alias_json)
    policy_file = load_policy_file(args.policy_json)
    policy = build_route_policy(policy_file)
    out.mkdir(parents=True, exist_ok=True)
    asr = args.asr_jsonl or work / "asr.jsonl"
    nll = args.nll_jsonl or work / "nll.jsonl"
    qkw = args.qkw_jsonl or work / "qkw.jsonl"
    embed = args.embedding_jsonl or work / "embed.jsonl"
    noise = args.noise_jsonl or work / "noise.jsonl"
    qhash = None
    if args.qkw_calibrated:
        if not args.qkw_calibrator_json or not args.qkw_calibrator_json.is_file():
            raise ValueError("--qkw-calibrated requires --qkw-calibrator-json")
        from quick.qkw import load_calibrator

        _, qhash = load_calibrator(args.qkw_calibrator_json)
    scored, score_meta = score_rows(
        rows, registry, asr_sidecar=asr if Path(asr).is_file() else None,
        nll_sidecar=nll if Path(nll).is_file() else None,
        qkw_sidecar=qkw if Path(qkw).is_file() else None,
        embedding_sidecar=embed if Path(embed).is_file() else None,
        noise_sidecar=noise if Path(noise).is_file() else None,
        aliases=aliases, qkw_calibrated=args.qkw_calibrated,
        qkw_calibrator_hash=qhash, feature_dir=out, policy=policy,
    )
    write_jsonl(out / "scored_candidates.jsonl", scored)
    groups = {}
    for row in scored:
        groups.setdefault(row["group_key"], []).append(row)
    expected = args.expected_uids or len(groups)
    local, decisions = local_evaluate(groups, policy, expected, cer_mean_max=0.03, cer0_drop_max=0.02, n_missing_asr=int(score_meta.get("n_missing_asr") or 0), allow_missing_asr=False)
    write_jsonl(out / "route_decisions.jsonl", decisions)
    metadata = {"reselect_from": str(work), "asr_sidecar_hash": file_sha256(asr), "nll_sidecar_hash": file_sha256(nll), "qkw_sidecar_hash": file_sha256(qkw), "qkw_calibrated": args.qkw_calibrated}
    summary = export_flat(scored, out_dir=out / "review_flat", policy=policy, selected_only_dir=out / "best_sep_selected", metadata=metadata, score_meta=score_meta)
    report = {"schema": "quick_reselect/v1", "status": "PASS" if local.get("local_pass") and summary.get("ok") else "NO_GO", "production_approved": False, "local": local, "score_meta": score_meta, "export": summary, "source_work_dir": str(work), "output_dir": str(out)}
    write_json(out / "report.json", report)
    print(f"[OK] reselected {report['status']} groups={len(groups)} output={out}")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
