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


def test_metrics():
    assert exact_match("a", "a")
    assert pass_at_k('["x","y"]', "y", k=2)

