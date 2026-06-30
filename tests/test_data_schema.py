from llmrec.data_schema import find_itemic_tokens, infer_domain_from_text, is_valid_itemic_token, normalize_itemic_text


def test_itemic_normalize_and_validate():
    raw = "<|live_begin|><a_1><b_2><c_3><|live_end|>"
    normalized = normalize_itemic_text(raw)
    assert normalized == "<|living_begin|><s_a_1><s_b_2><s_c_3>"
    assert is_valid_itemic_token(normalized)
    assert find_itemic_tokens(raw) == [normalized]


def test_infer_domain():
    assert infer_domain_from_text("<|prod_begin|><s_a_1><s_b_2><s_c_3>") == "product"

