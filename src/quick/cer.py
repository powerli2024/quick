from __future__ import annotations

import re
import string
import unicodedata
from pathlib import Path
from typing import Any

_CJK = re.compile(r"[\u4e00-\u9fff]")
_PUNCT = str.maketrans("", "", string.punctuation + "，。！？、；：‘’“”「」『』（）【】《》…·—–")

# Frozen Colmo-family aliases. Extra keys may be supplied by a pre-test JSON
# table; aliases must never be appended after seeing final-test ASR.
DEFAULT_ENGLISH_ALIASES: dict[str, list[str]] = {
    "hicolmo": ["hicolmo", "heycolmo", "hi colmo", "hey colmo"],
    "heycolmo": ["heycolmo", "hicolmo", "hey colmo", "hi colmo"],
    "hi colmo": ["hicolmo", "heycolmo", "hi colmo", "hey colmo"],
    "hey colmo": ["heycolmo", "hicolmo", "hey colmo", "hi colmo"],
}


def normalize_text(text: Any) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).lower()
    return value.translate(_PUNCT).strip()


def normalize_chars(text: Any) -> str:
    """KWS-compatible strict character form: NFKC, no whitespace, no punctuation."""
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = "".join(ch for ch in value if not ch.isspace())
    return value.translate(_PUNCT).lower().strip()


def normalize_en(text: Any) -> str:
    return " ".join(normalize_text(text).split())


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
    value = normalize_chars(text)
    return "".join(lazy_pinyin(value, style=Style.NORMAL, errors=lambda x: list(x.lower())))


def load_alias_table(path: str | Path | None, *, include_defaults: bool = True) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if include_defaults:
        for key, vals in DEFAULT_ENGLISH_ALIASES.items():
            out[normalize_en(key)] = list(dict.fromkeys(normalize_en(v) for v in vals))
    if path is None:
        return out
    import json

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("alias table must be a JSON object mapping reference to alias list")
    for key, vals in raw.items():
        if isinstance(vals, str):
            vals = [vals]
        if not isinstance(vals, list) or not all(isinstance(v, str) for v in vals):
            raise ValueError(f"alias table entry {key!r} must be a string list")
        canonical = normalize_en(key)
        aliases = [canonical, *(normalize_en(v) for v in vals)]
        merged = list(dict.fromkeys([*(out.get(canonical, [])), *aliases]))
        out[canonical] = merged
    return out


def alias_table_hash(aliases: dict[str, list[str]]) -> str:
    from .io import json_hash

    return json_hash({k: list(v) for k, v in sorted(aliases.items())})


def detail(hyp: str, ref: str, aliases: dict[str, list[str]] | None = None) -> dict[str, Any]:
    hyp_raw = str(hyp or "")
    ref_chars = normalize_chars(ref)
    hyp_chars = normalize_chars(hyp_raw)
    char = _distance(hyp_chars, ref_chars)
    if has_cjk(ref):
        py = _distance(to_pinyin(hyp_raw), to_pinyin(ref))
        route = py
        metric = "toneless_pinyin_cer"
        alias_hit = None
        alias_cer = None
        ref_tokens = list(normalize_text(ref).replace(" ", ""))
        hyp_tokens = list(normalize_text(hyp_raw).replace(" ", ""))
        core = ref_tokens[-1] if ref_tokens else ""
        core_hit = bool(core and core in hyp_chars)
        coverage_tokens = ref_tokens
    else:
        py = char
        ref_en = normalize_en(ref)
        hyp_en = normalize_en(hyp_raw)
        table = aliases or load_alias_table(None)
        choices = list(dict.fromkeys([ref_en, *(table.get(ref_en, [])), normalize_chars(ref) or ref_en]))
        scored = [(a, _distance(hyp_en, a)) for a in choices]
        alias_hit, alias_cer = min(scored, key=lambda x: (x[1], len(x[0]), x[0]))
        route = float(alias_cer)
        metric = "english_frozen_alias_cer"
        matched_reference = (alias_hit or ref_en).split()
        coverage_tokens = matched_reference
        core = matched_reference[-1] if matched_reference else ""
        if core.endswith("colmo") and core != "colmo":
            core = "colmo"
        core_hit = bool(core and core in hyp_chars)
        ref_tokens = ref_en.split()
        hyp_tokens = hyp_en.split()
    if coverage_tokens:
        hyp_blob = hyp_chars
        matched = sum(1 for token in coverage_tokens if token in hyp_tokens or normalize_chars(token) in hyp_blob)
        coverage = matched / len(coverage_tokens)
    else:
        coverage = 0.0
    extra = max(0, len(hyp_tokens) - len(ref_tokens)) / max(1, len(hyp_tokens))
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
