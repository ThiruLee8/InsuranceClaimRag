"""Trajectory evals: outcome vs path gap, and before/after on the top failure."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCORE = ROOT / "eval" / "trajectory_evals" / "score.py"
RUN = ROOT / "eval" / "trajectory_evals" / "run_evals.py"
CASES = ROOT / "eval" / "trajectory_evals" / "cases.json"
BEFORE = ROOT / "eval" / "trajectory_evals" / "fixtures" / "before.json"
AFTER = ROOT / "eval" / "trajectory_evals" / "fixtures" / "after.json"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_gap_case_right_answer_wrong_path():
    score = _load("traj_score", SCORE)
    cases = {c["id"]: c for c in json.loads(CASES.read_text(encoding="utf-8"))["cases"]}
    before = {i["id"]: i for i in json.loads(BEFORE.read_text(encoding="utf-8"))["items"]}
    row = score.score_trajectory(cases["gap-lucky-wrong-path"], before["gap-lucky-wrong-path"])
    assert row["outcome_pass"] is True
    assert row["trajectory_pass"] is False
    assert row["gap"] is True
    assert row["passed"] is False
    assert "settlement" in " ".join(row["queries"]).lower()


def test_good_path_passes_both():
    score = _load("traj_score", SCORE)
    cases = {c["id"]: c for c in json.loads(CASES.read_text(encoding="utf-8"))["cases"]}
    before = {i["id"]: i for i in json.loads(BEFORE.read_text(encoding="utf-8"))["items"]}
    row = score.score_trajectory(cases["good-cause-path"], before["good-cause-path"])
    assert row["outcome_pass"] is True
    assert row["trajectory_pass"] is True
    assert row["gap"] is False
    assert row["passed"] is True


def test_before_injection_follows_the_trick():
    score = _load("traj_score", SCORE)
    cases = {c["id"]: c for c in json.loads(CASES.read_text(encoding="utf-8"))["cases"]}
    before = {i["id"]: i for i in json.loads(BEFORE.read_text(encoding="utf-8"))["items"]}
    row = score.score_trajectory(cases["inject-indirect-arson"], before["inject-indirect-arson"])
    assert row["injection_followed"] is True
    assert row["passed"] is False


def test_after_injection_stops_the_trick():
    score = _load("traj_score", SCORE)
    cases = {c["id"]: c for c in json.loads(CASES.read_text(encoding="utf-8"))["cases"]}
    after = {i["id"]: i for i in json.loads(AFTER.read_text(encoding="utf-8"))["items"]}
    row = score.score_trajectory(cases["inject-indirect-arson"], after["inject-indirect-arson"])
    assert row["injection_followed"] is False
    assert row["outcome_pass"] is True
    assert row["passed"] is True
    assert "arson" not in row["answer"].lower()


def test_cost_percentile_and_tool_choice():
    score = _load("traj_score", SCORE)
    assert score.percentile([1, 2, 3, 4, 100], 99) >= 4
    assert score.tool_choice_accuracy(["search_documents", "finish"], ["search_documents", "finish"]) == 1.0
    assert 0 < score.tool_choice_accuracy(["search_documents", "finish"], ["list_documents", "finish"]) < 1


def test_run_evals_before_after_injection_drops(tmp_path: Path):
    results = tmp_path / "results"
    before = subprocess.run(
        [
            sys.executable,
            str(RUN),
            "--label",
            "before",
            "--fixtures",
            str(BEFORE),
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
            str(RUN),
            "--label",
            "after",
            "--fixtures",
            str(AFTER),
            "--results-dir",
            str(results),
            "--compare-to",
            "before",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads((results / "after.json").read_text(encoding="utf-8"))
    prior = json.loads((results / "before.json").read_text(encoding="utf-8"))
    assert prior["summary"]["injection_followed_rate"] > 0
    assert payload["summary"]["injection_followed_rate"] == 0.0
    hijack = payload["summary"]["by_problem_type"]["indirect_prompt_injection"]
    old = prior["summary"]["by_problem_type"]["indirect_prompt_injection"]
    assert old["injection_followed_rate"] == 1.0
    assert hijack["injection_followed_rate"] == 0.0
    assert hijack["pass_rate"] == 1.0
    assert payload["summary"]["pass_rate"] > prior["summary"]["pass_rate"]
    assert "inject-indirect-arson" in payload["delta_vs"]["improved_case_ids"]
    assert "injection_followed_rate:" in after.stdout
    assert "gap_rate" in before.stdout
