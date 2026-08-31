"""Tests for error-analysis ranking and trace helpers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RANK_PATH = ROOT / "eval" / "error_analysis" / "rank_problems.py"
SAMPLE_PATH = ROOT / "eval" / "error_analysis" / "sample_traces.py"
CODED = ROOT / "eval" / "error_analysis" / "samples" / "coded_sample_seed.json"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_seed_coded_sample_has_notes_before_types():
    data = json.loads(CODED.read_text(encoding="utf-8"))
    failures = [i for i in data["items"] if i.get("ok") is False]
    assert len(failures) >= 5
    for item in failures:
        assert (item.get("open_code_note") or "").strip(), item.get("traceId")
        assert (item.get("problem_type") or "").strip(), item.get("traceId")
        assert item.get("severity") in (1, 2, 3)


def test_rank_problems_orders_by_impact(tmp_path):
    # Invoke ranking logic via subprocess-equivalent: import and run main pieces
    import subprocess
    import sys

    out = tmp_path / "ranked.json"
    pred = tmp_path / "pred.md"
    result = subprocess.run(
        [
            sys.executable,
            str(RANK_PATH),
            "--sample",
            str(CODED),
            "--out",
            str(out),
            "--prediction-out",
            str(pred),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert out.exists()
    report = json.loads(out.read_text(encoding="utf-8"))
    ranked = report["ranked_problems"]
    assert ranked, result.stdout
    assert ranked[0]["problem_type"] == "wrong_document_retrieved"
    assert ranked[0]["frequency"] == 3
    assert report["chosen_fix_target"] == "wrong_document_retrieved"
    assert pred.exists()
    assert "prediction" in pred.read_text(encoding="utf-8").lower() or "Chosen target" in pred.read_text(
        encoding="utf-8"
    )


def test_sample_compact_preserves_replay_fields():
    mod = _load("sample_traces", SAMPLE_PATH)
    compact = mod.compact_for_worksheet(
        {
            "traceId": "t1",
            "question": "Q?",
            "searchQuery": "Q rewritten",
            "answer": "A",
            "retrieved": [{"fileName": "a.txt", "content": "hello", "relevanceScore": 0.9}],
        }
    )
    assert compact["traceId"] == "t1"
    assert compact["retrieved"][0]["content"] == "hello"
    assert compact["open_code_note"] == ""
    assert compact["problem_type"] == ""
