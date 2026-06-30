from llmrec.parse_sft import convert_record
from llmrec.utils import read_jsonl


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


def test_read_jsonl_flattens_single_item_arrays(tmp_path):
    path = tmp_path / "official.jsonl"
    path.write_text('[{"prompt":"问题","response":"答案"}]\n', encoding="utf-8")
    assert read_jsonl(path) == [{"prompt": "问题", "response": "答案"}]


def test_official_chinese_task_routing():
    record = {
        "source": "懂物料part1",
        "prompt": "请根据商品描述生成商品token",
        "response": "<|prod_begin|><s_a_1><s_b_2><s_c_3>",
    }
    assert convert_record(record, 1)["task_type"] == "material_understanding"
