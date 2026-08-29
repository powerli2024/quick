#!/usr/bin/env python3
"""Score cached CTC log-probability matrices with the frozen DP core.

The acoustic model is intentionally external.  A manifest row must provide
``log_probs_path`` (.npy or .npz with ``log_probs``) and either ``token_ids``
or a tokenizer JSON mapping text symbols to integer IDs.  Results are stable,
deduplicated and reusable across route runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from quick.ctc_align import forced_align_subsequence


def _rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_matrix(path: Path) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if isinstance(value, np.lib.npyio.NpzFile):
        if "log_probs" not in value:
            raise ValueError(f"{path} must contain npz key log_probs")
        value = value["log_probs"]
    return np.asarray(value, dtype=np.float64)


def _key(row: dict, tokenizer_hash: str, blank_id: int) -> str:
    payload = {"log_probs_path": str(row.get("log_probs_path")), "log_probs_sha256": row.get("log_probs_sha256"), "token_ids": row.get("token_ids"), "wake_text": row.get("wake_text"), "lang": row.get("lang"), "tokenizer_hash": tokenizer_hash, "blank_id": blank_id}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokenizer-json", type=Path, default=None, help="JSON symbol->id mapping; rows may instead provide token_ids")
    p.add_argument("--blank-id", type=int, default=0)
    p.add_argument("--sample-rate", type=float, default=50.0, help="CTC frames per second for returned start/end seconds")
    p.add_argument("--cache-jsonl", type=Path, default=None, help="optional content-keyed alignment cache")
    args = p.parse_args()
    mapping = json.loads(args.tokenizer_json.read_text(encoding="utf-8")) if args.tokenizer_json else {}
    if not isinstance(mapping, dict):
        raise ValueError("tokenizer JSON must be a symbol->integer object")
    tokenizer_hash = hashlib.sha256(json.dumps(mapping, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    rows = _rows(args.manifest)
    cache: dict[str, dict] = {}
    if args.cache_jsonl and args.cache_jsonl.is_file():
        for item in _rows(args.cache_jsonl):
            if item.get("ctc_key"):
                cache[str(item["ctc_key"])] = item
    out = []
    for row in rows:
        path = row.get("log_probs_path")
        if not path:
            raise ValueError(f"missing log_probs_path for {row.get('candidate_id')}")
        token_ids = row.get("token_ids")
        if token_ids is None:
            text = str(row.get("align_text") or row.get("wake_text") or "")
            try:
                token_ids = [int(mapping[ch]) for ch in text]
            except KeyError as exc:
                raise ValueError(f"tokenizer has no symbol {exc.args[0]!r} for {row.get('candidate_id')}") from exc
        token_ids = [int(x) for x in token_ids]
        key = _key({**row, "token_ids": token_ids}, tokenizer_hash, args.blank_id)
        if key not in cache:
            matrix = _load_matrix(Path(path))
            result = forced_align_subsequence(matrix, token_ids, blank_id=args.blank_id)
            item = {"schema": "quick_ctc_qkw/v2", "ctc_key": key, "score_key": row.get("score_key"), "candidate_id": row.get("candidate_id"), "pcm_sha256": row.get("pcm_sha256"), "wake_text": row.get("wake_text"), "lang": row.get("lang"), "backend": row.get("backend") or "external_ctc_logits", "tokens": token_ids, "best_alias": row.get("best_alias") or row.get("wake_text"), **result.as_dict(), "start_sec": result.start_frame / args.sample_rate, "end_sec": (result.end_frame + 1) / args.sample_rate, "occurrence_count": 1, "q_kw": None, "qkw_calibrated": False, "score_kind": "ctc_forced_alignment_uncalibrated", "tokenizer_hash": tokenizer_hash, "blank_id": args.blank_id}
            cache[key] = item
        item = dict(cache[key])
        item.update({"score_key": row.get("score_key"), "candidate_id": row.get("candidate_id"), "pcm_sha256": row.get("pcm_sha256"), "wake_text": row.get("wake_text"), "lang": row.get("lang")})
        out.append(item)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out), encoding="utf-8")
    if args.cache_jsonl:
        args.cache_jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.cache_jsonl.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in cache.values()), encoding="utf-8")
    print(json.dumps({"ok": True, "schema": "quick_ctc_qkw/v2", "n": len(out), "unique": len(cache), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
