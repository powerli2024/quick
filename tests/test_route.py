from quick.route import RoutePolicy, route_uid


def _row(**kwargs):
    base = {
        "candidate_id": "C-pos-u-s1-raw-original",
        "role": "s1",
        "view": "raw",
        "stream": "original",
        "validity": "rankable",
        "cer_route": 0.0,
        "q_kw": None,
        "nll": 1.0,
        "qkw_calibrated": False,
        "pcm_sha256": "aaa",
        "content_class": "target_wake",
        "audio_quality": {"clip_rate": 0.0, "speech_ratio": 0.7, "p_overlap": 0.0},
        "extra_ratio": 0.0,
        "s7_available": True,
        "decode_ok": True,
        "source_wav": "/tmp/x.wav",
        "core_hit": True,
        "lang": "en",
    }
    base.update(kwargs)
    return base


def test_cer0_trusted_keeps_s1():
    s1 = _row()
    s7 = _row(candidate_id="C-pos-u-s7-raw-original", role="s7", cer_route=0.0, pcm_sha256="bbb")
    result = route_uid([s1, s7], RoutePolicy())
    assert result["reason_code"] == "KEEP_S1_CER0_TRUSTED"
    assert result["selected"]["role"] == "s1"
    assert result["triggered_s7"] is False


def test_s7_switches_on_strict_lower_cer():
    s1 = _row(cer_route=0.25)
    s7 = _row(candidate_id="C-pos-u-s7-moss48k-spk1", role="s7", view="moss48k", stream="spk1", cer_route=0.0, pcm_sha256="ccc")
    result = route_uid([s1, s7], RoutePolicy())
    assert result["triggered_s7"] is True
    assert result["switched_s7"] is True
    assert result["reason_code"] == "SWITCH_S7_LOWER_CER"
    assert result["s7_view_reason_code"] in {"MOSS_ONLY", "MOSS_LOWER_CER", "RAW_ONLY", "RAW_CONSERVATIVE_TIE"}


def test_same_pcm_keeps_s1_alias():
    s1 = _row(cer_route=0.2, pcm_sha256="same")
    s7 = _row(candidate_id="C-pos-u-s7-raw-original", role="s7", cer_route=0.2, pcm_sha256="same")
    result = route_uid([s1, s7], RoutePolicy())
    assert result["reason_code"] == "SAME_AUDIO_KEEP_S1"
    assert result["selected"]["role"] == "s1"


def test_s7_unavailable_reason():
    s1 = _row(cer_route=0.3, s7_available=False)
    result = route_uid([s1], RoutePolicy())
    assert result["reason_code"] == "KEEP_S1_S7_UNAVAILABLE"


def test_keep_s1_when_s7_cer_worse():
    s1 = _row(cer_route=0.1)
    s7 = _row(candidate_id="C-pos-u-s7-raw-spk1", role="s7", stream="spk1", cer_route=0.4, pcm_sha256="ddd")
    result = route_uid([s1, s7], RoutePolicy())
    assert result["reason_code"] == "KEEP_S1_LOWER_CER"
    assert result["switched_s7"] is False


def test_equal_cer_nll_gain_switches():
    s1 = _row(cer_route=0.2, nll=1.0, pcm_sha256="e1")
    s7 = _row(candidate_id="C-pos-u-s7-raw-original", role="s7", cer_route=0.2, nll=0.2, pcm_sha256="e2")
    result = route_uid([s1, s7], RoutePolicy(nll_switch_margin=0.01))
    assert result["reason_code"] == "SWITCH_S7_EQUAL_CER_NLL_GAIN"


def test_raw_preferred_on_stage_tie():
    raw = _row(cer_route=0.0, view="raw")
    moss = _row(candidate_id="C-pos-u-s1-moss48k-original", view="moss48k", cer_route=0.0, pcm_sha256="zzz")
    result = route_uid([moss, raw], RoutePolicy())
    assert result["selected"]["view"] == "raw"
    assert result["s1_view_reason_code"] == "RAW_CONSERVATIVE_TIE"
