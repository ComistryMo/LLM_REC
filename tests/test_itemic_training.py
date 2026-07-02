import pytest

from llmrec.itemic_training import build_variant, split_itemic


TOKEN = "<|video_begin|><s_a_12><s_b_34><s_c_56>"
MESSAGES = [
    {"role": "system", "content": "system"},
    {"role": "user", "content": "caption/no_think"},
    {"role": "assistant", "content": TOKEN},
]


def test_split_itemic():
    assert split_itemic(TOKEN) == ("<|video_begin|>", "<s_a_12>", "<s_b_34>", "<s_c_56>")
    with pytest.raises(ValueError):
        split_itemic("<s_a_12><s_b_34><s_c_56>")


def test_build_sid_and_hierarchy_variants():
    assert build_variant(MESSAGES, "sid")["output"] == "<s_a_12><s_b_34><s_c_56>"
    assert build_variant(MESSAGES, "a")["output"] == "<s_a_12>"
    assert "<s_a_12>" in build_variant(MESSAGES, "b")["input"]
    assert "<s_a_12><s_b_34>" in build_variant(MESSAGES, "c")["input"]
    assert build_variant(MESSAGES, "full")["output"] == TOKEN
