#!/usr/bin/env python3
"""Turn an independent WeNet CTC decode into A3 keyword evidence.

The decoder is intentionally an adapter: ``--decode-command`` may call the
official WeNet ``recognize.py`` recipe, a local ONNX runtime, or an existing
decode cache.  It must write either JSONL (``score_key``/``hyp`` or
``key``/``text``) or Kaldi-style ``key text`` lines to ``{output}``.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from quick.cer import detail, load_alias_table  # noqa: E402


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _decode_rows(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                out[parts[0]] = parts[1].strip()
            continue
        if not isinstance(row, dict):
            continue
        key = row.get("score_key") or row.get("key") or row.get("candidate_id")
        text = row.get("hyp") or row.get("text") or row.get("transcript") or ""
        if key:
            out[str(key)] = str(text)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--decode-command", default=None,
                    help="template creating {output}; may use {manifest} and {model_dir}")
    ap.add_argument("--decode-jsonl", type=Path, default=None,
                    help="reuse an already generated WeNet transcript sidecar")
    ap.add_argument("--model-dir", default="")
    ap.add_argument("--alias-json", type=Path, default=None)
    ap.add_argument("--enabled-langs", default="zh",
                    help="comma-separated languages scored by this checkpoint (default: zh)")
    args = ap.parse_args()

    source = args.decode_jsonl
    decoder_error = None
    if source is None and args.decode_command:
        rendered = str(args.decode_command)
        for name, value in {
            "manifest": str(args.manifest), "output": str(args.output),
            "model_dir": str(args.model_dir),
        }.items():
            rendered = rendered.replace("{" + name + "}", value)
        try:
            subprocess.run(shlex.split(rendered, posix=os.name != "nt"), check=True)
            source = args.output
        except subprocess.CalledProcessError as exc:
            # CTC is an optional tie-breaker.  A checkpoint/config mismatch
            # must not discard a completed SenseVoice/SE run.
            decoder_error = f"decoder_exit_{exc.returncode}"
            source = None
    if source is None and decoder_error is None:
        raise SystemExit("A3 CTC needs --decode-command or --decode-jsonl")
    if source is not None and not source.is_file():
        raise SystemExit(f"WeNet decoder output not found: {source}")

    decoded = _decode_rows(source) if source is not None else {}
    aliases = load_alias_table(args.alias_json)
    enabled_langs = {x.strip().lower() for x in str(args.enabled_langs).split(",") if x.strip()}
    unique = {}
    for row in read_jsonl(args.manifest):
        sk = str(row.get("score_key") or "")
        if not sk or sk in unique:
            continue
        unique[sk] = row

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_missing = 0
    with args.output.open("w", encoding="utf-8") as f:
        for sk, row in unique.items():
            lang = str(row.get("lang") or "").lower()
            enabled = not enabled_langs or lang in enabled_langs
            hyp = decoded.get(sk, decoded.get(str(row.get("candidate_id")), "")) if enabled else ""
            if not hyp:
                n_missing += 1
            metrics = detail(hyp, str(row.get("wake_text") or ""), aliases)
            # 1-CER is deliberately a within-UID CTC alignment score, not a
            # cross-recording probability.  The route uses it only after CER
            # ties; calibration can later populate q_kw/qkw_calibrated.
            align = max(0.0, min(1.0, 1.0 - float(metrics["cer_route"]))) if enabled and decoder_error is None else None
            out = {
                "score_key": sk,
                "pcm_sha256": row.get("pcm_sha256"),
                "wake_text": row.get("wake_text"),
                "lang": row.get("lang"),
                "hyp": hyp,
                "ctc_hyp": hyp,
                "ctc_align_score": align,
                "target_coverage": metrics["wake_coverage"] if enabled and decoder_error is None else None,
                "ctc_cer_route": metrics["cer_route"],
                "score_kind": "ctc_align_uncalibrated" if decoder_error is None else "ctc_unavailable",
                "backend": "wenet_ctc",
                "error": decoder_error,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"[A3][WeNet-CTC] unique={len(unique)} missing={n_missing} output={args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
