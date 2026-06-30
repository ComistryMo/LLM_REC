from pathlib import Path

from llmrec.format_check import check_records
from llmrec.metrics import exact_match, pass_at_k


def test_format_secret_detection(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"id":"x","messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"password: abc"}]}\n',
        encoding="utf-8",
    )
    report = check_records(path)
    assert report["ok"] is False
    assert report["error_counts"]["secret_like_text"] == 1


def test_itemic_token_label_is_not_a_secret(tmp_path: Path):
    path = tmp_path / "itemic.jsonl"
    path.write_text(
        '{"id":"x","messages":[{"role":"user","content":"商品token:<|prod_begin|><s_a_1><s_b_2><s_c_3>"},{"role":"assistant","content":"ok"}]}\n',
        encoding="utf-8",
    )
    assert check_records(path)["ok"] is True


def test_metrics():
    assert exact_match("a", "a")
    assert pass_at_k('["x","y"]', "y", k=2)
