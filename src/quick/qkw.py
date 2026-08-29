from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .io import json_hash, read_json

SCHEMA = "quick_qkw_calibrator/v1"


def calibrator_hash(payload: dict[str, Any]) -> str:
    return json_hash({k: v for k, v in payload.items() if k != "calibrator_hash"})


def load_calibrator(path: str | Path) -> tuple[dict[str, Any], str]:
    payload = read_json(path)
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"invalid q_kw calibrator schema: {payload.get('schema')!r}")
    digest = calibrator_hash(payload)
    declared = payload.get("calibrator_hash")
    if declared and declared != digest:
        raise ValueError(f"q_kw calibrator hash mismatch declared={declared} actual={digest}")
    models = payload.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("q_kw calibrator has no language models")
    if payload.get("independent_dev_required") is not True or not payload.get("source_sha256"):
        raise ValueError("q_kw calibrator must bind an independent labeled development set")
    if payload.get("score_direction") not in {"higher_is_better", "lower_is_better"}:
        raise ValueError("q_kw calibrator has invalid score_direction")
    for lang, model in models.items():
        if lang not in {"zh", "en", "global"} or not isinstance(model, dict):
            raise ValueError(f"invalid q_kw language model {lang!r}")
        required = ("mean", "scale", "slope", "intercept", "n", "n_positive", "n_negative")
        if any(name not in model for name in required):
            raise ValueError(f"q_kw model {lang!r} is incomplete")
        if float(model["scale"]) <= 0 or min(int(model["n_positive"]), int(model["n_negative"])) <= 0:
            raise ValueError(f"q_kw model {lang!r} has invalid scale/class coverage")
    return payload, digest


def fit_logistic(scores: list[float], labels: list[int], *, l2: float = 1e-4) -> dict[str, Any]:
    x = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or x.size < 2:
        raise ValueError("q_kw calibration needs paired one-dimensional scores and labels")
    if not np.isfinite(x).all() or not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("q_kw calibration scores must be finite and labels must be 0/1")
    if np.unique(y).size != 2:
        raise ValueError("q_kw calibration requires both positive and negative labels")
    mean = float(np.mean(x))
    scale = float(np.std(x))
    if scale < 1e-8:
        raise ValueError("q_kw calibration score is constant")
    z = (x - mean) / scale
    design = np.column_stack([z, np.ones_like(z)])
    theta = np.zeros(2, dtype=np.float64)
    for _ in range(100):
        logits = np.clip(design @ theta, -35.0, 35.0)
        prob = 1.0 / (1.0 + np.exp(-logits))
        weight = np.maximum(prob * (1.0 - prob), 1e-8)
        grad = design.T @ (prob - y) + np.array([l2 * theta[0], 0.0])
        hess = design.T @ (weight[:, None] * design) + np.diag([l2, 1e-10])
        step = np.linalg.solve(hess, grad)
        theta -= step
        if float(np.max(np.abs(step))) < 1e-9:
            break
    return {
        "method": "logistic_platt",
        "mean": mean,
        "scale": scale,
        "slope": float(theta[0]),
        "intercept": float(theta[1]),
        "n": int(x.size),
        "n_positive": int(np.sum(y == 1)),
        "n_negative": int(np.sum(y == 0)),
    }


def apply_model(model: dict[str, Any], score: float) -> float:
    z = (float(score) - float(model["mean"])) / float(model["scale"])
    logit = float(model["slope"]) * z + float(model["intercept"])
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-min(logit, 700.0)))
    exp_value = math.exp(max(logit, -700.0))
    return exp_value / (1.0 + exp_value)


def apply_calibrator(payload: dict[str, Any], *, lang: str, score: float) -> float:
    models = payload["models"]
    model = models.get(str(lang)) or models.get("global")
    if model is None:
        raise ValueError(f"q_kw calibrator has no model for lang={lang!r}")
    return apply_model(model, score)
