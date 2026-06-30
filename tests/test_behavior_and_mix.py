from pathlib import Path

from llmrec.build_history import build_history_sft
from llmrec.mix_datasets import mix_datasets
from llmrec.parse_behavior import parse_behavior_files
from llmrec.parse_sft import parse_sft_files
from llmrec.utils import read_jsonl


def test_behavior_sort_and_history(tmp_path: Path):
    input_path = tmp_path / "behavior.jsonl"
    input_path.write_text(
        "\n".join(
            [
                '{"user_id":"u","timestamp":"2","action_type":"click","domain":"product","itemic_pattern":"<|prod_begin|><s_a_1><s_b_2><s_c_3>"}',
                '{"user_id":"u","timestamp":"1","action_type":"watch","domain":"video","itemic_pattern":"<|video_begin|><s_a_1><s_b_2><s_c_3>"}',
            ]
        ),
        encoding="utf-8",
    )
    seq_path = tmp_path / "seq.jsonl"
    parse_behavior_files([input_path], seq_path)
    seq = read_jsonl(seq_path)[0]["events"]
    assert [e["timestamp"] for e in seq] == ["1", "2"]
    user_out = tmp_path / "D_user.jsonl"
    rec_out = tmp_path / "D_rec.jsonl"
    build_history_sft(seq_path, user_out, rec_out, windows=[1])
    assert user_out.exists()
    assert rec_out.exists()


def test_mix_pipeline(tmp_path: Path):
    sft = tmp_path / "sft.jsonl"
    sft.write_text(
        '{"id":"a","instruction":"CAP 定理是什么意思？","output":"CAP 定理是分布式系统理论。"}\n'
        '{"id":"b","prompt":"请推荐商品","completion":"<|prod_begin|><s_a_1><s_b_2><s_c_3>"}\n',
        encoding="utf-8",
    )
    unified = tmp_path / "unified.jsonl"
    parse_sft_files([sft], unified)
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "valid.jsonl"
    train_count, valid_count = mix_datasets([unified], train, valid, valid_ratio=0.5)
    assert train_count + valid_count == 2

