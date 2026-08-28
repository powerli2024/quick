from __future__ import annotations

import hashlib
import math
import wave
from pathlib import Path
from typing import Any

import numpy as np


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf

        x, sr = sf.read(str(path), always_2d=False)
        x = np.asarray(x, dtype=np.float32)
        if x.ndim > 1:
            x = x.mean(axis=-1)
        return x.reshape(-1), int(sr)
    except Exception:
        with wave.open(str(path), "rb") as handle:
            sr, channels, width, n = handle.getframerate(), handle.getnchannels(), handle.getsampwidth(), handle.getnframes()
            raw = handle.readframes(n)
        if width == 2:
            x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif width == 4:
            x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise RuntimeError(f"unsupported WAV sample width {width}: {path}")
        if channels > 1:
            x = x.reshape(-1, channels).mean(axis=1)
        return x.reshape(-1), int(sr)


def resample_linear(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if orig_sr == target_sr or x.size == 0:
        return x
    n = max(1, int(round(len(x) * target_sr / orig_sr)))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def write_wav(path: str | Path, x: np.ndarray, sr: int = 16000) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    x = np.clip(np.asarray(x, dtype=np.float32).reshape(-1), -1, 1)
    try:
        import soundfile as sf

        sf.write(str(p), x, sr)
        return
    except Exception:
        pcm = (x * 32767).astype(np.int16)
        with wave.open(str(p), "wb") as handle:
            handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(sr); handle.writeframes(pcm.tobytes())


def canonical_pcm(x: np.ndarray, sr: int, target_sr: int = 16000) -> np.ndarray:
    y = resample_linear(np.asarray(x, dtype=np.float32), sr, target_sr)
    return np.asarray(y, dtype="<f4")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pcm_sha256(x: np.ndarray, sr: int, target_sr: int = 16000) -> str:
    return hashlib.sha256(canonical_pcm(x, sr, target_sr).tobytes()).hexdigest()


def quality_metrics(x: np.ndarray, sr: int, *, p_music: float | None = None, p_overlap: float | None = None) -> dict[str, Any]:
    y = np.asarray(x, dtype=np.float32).reshape(-1)
    eps = 1e-8
    if y.size == 0 or not np.isfinite(y).all():
        return {"duration_sec": 0.0, "fatal_invalid": True, "reason": "empty_or_nonfinite"}
    rms = float(np.sqrt(np.mean(y * y)))
    peak = float(np.max(np.abs(y)))
    frame = max(1, int(round(sr * 0.02)))
    usable = y[: (len(y) // frame) * frame]
    if usable.size:
        powers = np.mean(usable.reshape(-1, frame) ** 2, axis=1)
        threshold = max(float(np.percentile(powers, 20)) * 2.0, 1e-7)
        active = powers > threshold
        speech_ratio = float(np.mean(active))
        p_active = float(np.mean(powers[active])) if active.any() else 0.0
        p_noise = float(np.mean(powers[~active])) if (~active).any() else 0.0
        snr_valid = bool(active.any() and (~active).any() and p_noise > eps)
        snr = 10 * math.log10(max(p_active - p_noise, eps) / (p_noise + eps)) if snr_valid else None
        longest = 0; current = 0
        for value in active:
            current = current + 1 if value else 0
            longest = max(longest, current)
    else:
        speech_ratio, snr, snr_valid, longest = 0.0, None, False, 0
    spec = np.abs(np.fft.rfft(y[: min(len(y), sr * 10)] * np.hanning(min(len(y), sr * 10)))) if len(y) > 1 else np.array([0.0])
    power = spec * spec + eps
    flatness = float(np.exp(np.mean(np.log(power))) / np.mean(power))
    flux = float(np.mean(np.abs(np.diff(spec)))) if len(spec) > 1 else 0.0
    return {
        "duration_sec": float(len(y) / sr),
        "rms_dbfs": float(20 * math.log10(rms + eps)),
        "peak": peak,
        "clip_rate": float(np.mean(np.abs(y) >= 0.999)),
        "dc_offset": float(abs(np.mean(y))),
        "zero_ratio": float(np.mean(np.abs(y) < 1e-6)),
        "speech_ratio": speech_ratio,
        "silence_ratio": 1.0 - speech_ratio,
        "longest_speech_sec": float(longest * frame / sr),
        "snr_vad_db": snr,
        "snr_valid": snr_valid,
        "spectral_flatness": flatness,
        "spectral_flux": flux,
        "p_music": p_music,
        "p_overlap": p_overlap,
        "fatal_invalid": bool(float(len(y) / sr) <= 0 or not np.isfinite(y).all()),
    }
