"""Unit tests for answer eval assertions and offline suite scoring."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSERT_PATH = ROOT / "eval" / "answer_evals" / "assertions.py"
RUN_PATH = ROOT / "eval" / "answer_evals" / "run_evals.py"
VALIDATE_PATH = ROOT / "eval" / "answer_evals" / "validate_judge.py"
CASES = ROOT / "eval" / "answer_evals" / "cases.json"
BEFORE = ROOT / "eval" / "answer_evals" / "fixtures" / "before.json"
AFTER = ROOT / "eval" / "answer_evals" / "fixtures" / "after.json"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_assertions_catch_week5_failures():
    mod = _load("assertions", ASSERT_PATH)
    cases = {c["id"]: c for c in json.loads(CASES.read_text(encoding="utf-8"))["cases"]}
    before = {i["id"]: i for i in json.loads(BEFORE.read_text(encoding="utf-8"))["items"]}

    bad = mod.run_assertions(
        case=cases["reg-property-address"],
        answer=before["reg-property-address"]["answer"],
        retrieved=before["reg-property-address"]["retrieved"],
    )
    assert bad["passed"] is False

    hall = mod.run_assertions(
        case=cases["reg-investigator-no-hallucination"],
        answer=before["reg-investigator-no-hallucination"]["answer"],
        retrieved=before["reg-investigator-no-hallucination"]["retrieved"],
    )
    assert hall["passed"] is False


def test_assertions_pass_after_fixtures():
    mod = _load("assertions", ASSERT_PATH)
    cases = {c["id"]: c for c in json.loads(CASES.read_text(encoding="utf-8"))["cases"]}
    after = {i["id"]: i for i in json.loads(AFTER.read_text(encoding="utf-8"))["items"]}

    good = mod.run_assertions(
        case=cases["reg-net-settlement-exact"],
        answer=after["reg-net-settlement-exact"]["answer"],
        retrieved=after["reg-net-settlement-exact"]["retrieved"],
    )
    assert good["passed"] is True


def test_context_recall_precision():
    mod = _load("assertions", ASSERT_PATH)
    retrieved = [
        {"fileName": "claim-investigation-report.txt"},
        {"fileName": "policy-schedule.txt"},
    ]
    assert mod.context_recall(retrieved=retrieved, expected_document="claim-investigation-report") == 1.0
    assert mod.context_precision(retrieved=retrieved, expected_document="claim-investigation-report") == 0.5


def test_run_evals_before_after_improves(tmp_path: Path):
    results = tmp_path / "results"
    before = subprocess.run(
        [
            sys.executable,
            str(RUN_PATH),
            "--label",
            "before",
            "--fixtures",
            str(BEFORE),
            "--no-judge",
            "--results-dir",
            str(results),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    after = subprocess.run(
        [
            sys.executable,
            str(RUN_PATH),
            "--label",
            "after",
            "--fixtures",
            str(AFTER),
            "--no-judge",
            "--results-dir",
            str(results),
            "--compare-to",
            "before",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "pass_rate" in after.stdout.lower() or "PASS" in after.stdout or "Summary" in after.stdout
    b = json.loads((results / "before.json").read_text(encoding="utf-8"))
    a = json.loads((results / "after.json").read_text(encoding="utf-8"))
    assert a["summary"]["pass_rate"] > b["summary"]["pass_rate"]
    assert a["summary"]["mean_score"] > b["summary"]["mean_score"]
    assert "wrong_document_retrieved" in a["summary"]["by_problem_type"]
    assert before.returncode == 0


def test_validate_judge_offline_agreement():
    result = subprocess.run(
        [sys.executable, str(VALIDATE_PATH), "--dry-run-offline"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "PASS" in result.stdout
    assert "accuracy" in result.stdout.lower() or "accuracy:" in result.stdout
