from pathlib import Path

from llmrec.split_dataset import content_key, split_official_dataset
from llmrec.utils import read_jsonl, write_jsonl


def test_split_filters_long_and_prevents_duplicate_leakage(tmp_path: Path):
    records = []
    for i in range(20):
        records.append(
            {
                "id": str(i),
                "source": "懂物料part1" if i < 10 else "懂推荐1",
                "task_type": "material_understanding" if i < 10 else "recommendation",
                "input_len": 10,
                "output_len": 5,
                "messages": [
                    {"role": "user", "content": f"q{i}"},
                    {"role": "assistant", "content": f"a{i}"},
                ],
            }
        )
    records.append({**records[0], "id": "duplicate"})
    records.append({**records[1], "id": "long", "input_len": 1000})
    input_path = tmp_path / "input.jsonl"
    train_path = tmp_path / "train.jsonl"
    valid_path = tmp_path / "valid.jsonl"
    write_jsonl(records, input_path)

    report = split_official_dataset(
        input_path,
        train_path,
        valid_path,
        tmp_path / "report.json",
        valid_ratio=0.2,
        max_length=100,
    )
    train = read_jsonl(train_path)
    valid = read_jsonl(valid_path)
    assert report["removed_exact_duplicates"] == 1
    assert report["excluded_over_max_length"] == 1
    assert {content_key(r) for r in train}.isdisjoint(content_key(r) for r in valid)
