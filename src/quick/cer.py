from __future__ import annotations

import re
import string
import unicodedata
from pathlib import Path
from typing import Any

_CJK = re.compile(r"[\u4e00-\u9fff]")
_PUNCT = str.maketrans("", "", string.punctuation + "，。！？、；：‘’“”「」『』（）【】《》…·—–")


def normalize_text(text: Any) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).lower()
    return value.translate(_PUNCT).strip()


def normalize_en(text: Any) -> str:
    value = normalize_text(text)
    return " ".join(value.split())


def has_cjk(text: Any) -> bool:
    return bool(_CJK.search(str(text or "")))


def _distance(left: str, right: str) -> float:
    if not right:
        return 0.0 if not left else 1.0
    prev = list(range(len(left) + 1))
    for i, rch in enumerate(right, 1):
        cur = [i]
        for j, lch in enumerate(left, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (rch != lch)))
        prev = cur
    return float(prev[-1]) / len(right)


def to_pinyin(text: Any) -> str:
    try:
        from pypinyin import Style, lazy_pinyin
    except ModuleNotFoundError as exc:
        raise RuntimeError("Chinese CER requires pypinyin") from exc
    value = normalize_text(text).replace(" ", "")
    return "".join(lazy_pinyin(value, style=Style.NORMAL, errors=lambda x: list(x.lower())))


def load_alias_table(path: str | Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    import json

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("alias table must be a JSON object mapping reference to alias list")
    out: dict[str, list[str]] = {}
    for key, vals in raw.items():
        if isinstance(vals, str):
            vals = [vals]
        if not isinstance(vals, list) or not all(isinstance(v, str) for v in vals):
            raise ValueError(f"alias table entry {key!r} must be a string list")
        canonical = normalize_en(key)
        aliases = [normalize_en(key), *(normalize_en(v) for v in vals)]
        out[canonical] = list(dict.fromkeys(aliases))
    return out


def detail(hyp: str, ref: str, aliases: dict[str, list[str]] | None = None) -> dict[str, Any]:
    hyp_raw = str(hyp or "")
    ref_norm = normalize_text(ref)
    hyp_norm = normalize_text(hyp_raw)
    char = _distance(hyp_norm, ref_norm)
    py = _distance(to_pinyin(hyp_raw), to_pinyin(ref)) if has_cjk(ref) else char
    if has_cjk(ref):
        route = py
        metric = "toneless_pinyin_cer"
        alias_hit = None
        alias_cer = None
        ref_tokens = list(ref_norm)
        hyp_tokens = list(hyp_norm)
    else:
        ref_en = normalize_en(ref)
        hyp_en = normalize_en(hyp_raw)
        table = aliases or {}
        choices = table.get(ref_en, [ref_en])
        scored = [(a, _distance(hyp_en, a)) for a in choices]
        alias_hit, alias_cer = min(scored, key=lambda x: (x[1], x[0]))
        route = alias_cer
        metric = "english_frozen_alias_cer"
        ref_tokens = ref_en.split()
        hyp_tokens = hyp_en.split()
    if has_cjk(ref):
        core = ref_tokens[-1] if ref_tokens else ""
        coverage_tokens = ref_tokens
        core_hit = bool(core and core in hyp_norm)
    else:
        matched_reference = (alias_hit or normalize_en(ref)).split()
        coverage_tokens = matched_reference
        core = matched_reference[-1] if matched_reference else ""
        # ``hicolmo``/``heycolmo`` are one-token spellings whose business core
        # is the brand suffix.  A production alias table may provide richer
        # core_keywords; this conservative fallback only handles the frozen
        # Colmo family and never invents arbitrary aliases.
        if core.endswith("colmo") and core != "colmo":
            core = "colmo"
        core_hit = bool(core and core in normalize_en(hyp_raw).replace(" ", ""))
    if coverage_tokens:
        matched = sum(1 for token in coverage_tokens if token in hyp_tokens or token in hyp_norm)
        coverage = matched / len(coverage_tokens)
    else:
        coverage = 0.0
    extra = max(0, len(hyp_tokens) - len(ref_tokens)) / max(1, len(hyp_tokens)) if not has_cjk(ref) else max(0, len(hyp_norm)-len(ref_norm)) / max(1, len(hyp_norm))
    return {
        "hyp": hyp_raw,
        "ref": str(ref),
        "cer_route": float(route),
        "cer_char": float(char),
        "cer_py": float(py),
        "metric": metric,
        "alias_hit": alias_hit,
        "cer_alias": None if alias_cer is None else float(alias_cer),
        "wake_coverage": float(coverage),
        "extra_ratio": float(extra),
        "core_hit": bool(core_hit),
    }
