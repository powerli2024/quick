from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from quick.audio import write_wav


def make_tone(path: Path, freq: float, sr: int = 16000, sec: float = 0.6) -> None:
    n = int(sr * sec)
    t = np.arange(n, dtype=np.float32) / sr
    tone = np.sin(2 * math.pi * freq * t)
    # Burst envelope so VAD speech_ratio is not near-zero on a constant sinusoid.
    frame = int(sr * 0.02)
    env = np.ones(n, dtype=np.float32)
    for i in range(0, n, frame * 4):
        env[i : i + frame] *= 0.02
    write_wav(path, (0.25 * tone * env).astype(np.float32), sr)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def make_arm(root: Path, split: str, arm: str, uids: list[str], *, freq_base: float, wake: str = "hicolmo") -> None:
    base = root / split / arm
    wav_dir = base / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, uid in enumerate(uids):
        make_tone(wav_dir / f"{uid}_peak.wav", freq_base + 15 * i)
        make_tone(wav_dir / f"{uid}_spk1.wav", freq_base + 90 + 15 * i)
        rows.append({
            "uid": uid,
            "wake_text": wake,
            "lang": "en",
            "streams": {"original": {}, "spk1": {}},
        })
    write_jsonl(base / "index.jsonl", rows)


def make_pos_neg(root: Path) -> Path:
    make_arm(root, "pos", "s1_onnx_full", ["pos_0001", "pos_0002"], freq_base=220)
    make_arm(root, "pos", "s7_cv_then_onnx_gate/thr_a", ["pos_0001", "pos_0002"], freq_base=480)
    make_arm(root, "neg", "s1_onnx_full", ["neg_0001"], freq_base=260)
    make_arm(root, "neg", "s7_cv_then_onnx_gate/thr_a", ["neg_0001"], freq_base=520)
    return root
