from llmrec.parse_sft import convert_record


def test_convert_alpaca_record():
    record = {
        "instruction": "请推荐一个商品",
        "input": "用户搜索露营装备",
        "output": "<|prod_begin|><s_a_1><s_b_2><s_c_3>",
    }
    converted = convert_record(record, 1)
    assert converted["messages"][-1]["role"] == "assistant"
    assert converted["task_type"] == "recommendation"
    assert converted["output_type"] == "itemic_pattern"
    assert converted["has_itemic"] is True

