from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

from .io import read_json
from .route import RoutePolicy

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_JSON = REPO_ROOT / "configs" / "route_policy.json"

POLICY_KEYS = {f.name for f in fields(RoutePolicy)}


def load_policy_file(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        path = DEFAULT_POLICY_JSON if DEFAULT_POLICY_JSON.is_file() else None
    if path is None:
        return {}
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"route policy must be a JSON object: {path}")
    return data


def build_route_policy(file_data: dict[str, Any], *, overrides: dict[str, Any] | None = None) -> RoutePolicy:
    kwargs: dict[str, Any] = {}
    for key in POLICY_KEYS:
        if key in file_data and file_data[key] is not None:
            kwargs[key] = file_data[key]
    for key, value in (overrides or {}).items():
        if value is not None and key in POLICY_KEYS:
            kwargs[key] = value
    return RoutePolicy(**kwargs)
