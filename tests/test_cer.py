from quick.cer import detail, load_alias_table


def test_english_alias_cer_zero_for_heycolmo():
    aliases = load_alias_table(None)
    scored = detail("heycolmo", "hicolmo", aliases)
    assert scored["cer_route"] == 0.0
    assert scored["cer_alias"] == 0.0
    assert scored["core_hit"] is True
    assert scored["alias_hit"] in {"heycolmo", "hicolmo"}
    assert scored["metric"] == "english_frozen_alias_cer"


def test_english_core_hit_requires_colmo():
    scored = detail("hello world", "hicolmo")
    assert scored["core_hit"] is False
    assert scored["cer_route"] > 0


def test_english_partial_colmo_is_nonzero_but_rankable():
    scored = detail("colmo", "hicolmo")
    assert scored["core_hit"] is True
    assert scored["cer_route"] > 0


def test_chinese_toneless_pinyin():
    scored = detail("科目", "科慕")
    assert scored["cer_route"] == 0.0
    assert scored["cer_char"] > 0.0
    assert scored["metric"] == "toneless_pinyin_cer"
