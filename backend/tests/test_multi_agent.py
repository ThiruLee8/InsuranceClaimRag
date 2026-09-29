"""Specialist team, A2A task lifecycle, and the single-vs-team score rule."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from app.services.a2a import A2AStateError, A2ATask
from app.services.agent_loop import ClaimAgentRunner, estimate_tokens
from app.services.agent_memory import AgentMemoryStore
from app.services.multi_agent import TeamRunner, route_task
from app.services.vector_gateway_client import VectorSearchHit, VectorSearchResult

ROOT = Path(__file__).resolve().parents[2]
SCORE_PATH = ROOT / "eval" / "multi_agent" / "score.py"
RACE_PATH = ROOT / "eval" / "multi_agent" / "run_race.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


score = _load(SCORE_PATH, "multi_agent_score")
run_race = _load(RACE_PATH, "multi_agent_run_race")


class ScriptedLLM:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []
        self.systems: list[str] = []

    async def generate(self, *, prompt: str, system: str | None = None, model: str | None = None) -> str:
        self.prompts.append(prompt)
        self.systems.append(system or "")
        if not self.replies:
            return "No further finding."
        return self.replies.pop(0)


class SlowLLM:
    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s

    async def generate(self, *, prompt: str, system: str | None = None, model: str | None = None) -> str:
        await asyncio.sleep(self.delay_s)
        return "Burst supply line. Deductible USD 1,000. Net settlement USD 19,550. Flood excluded."


def _hit(name: str, content: str) -> VectorSearchHit:
    return VectorSearchHit(
        chunk_id=str(uuid4()),
        document_id=str(uuid4()),
        file_name=name,
        page_number=1,
        chunk_index=0,
        content=content,
        score=0.9,
    )


def _gateway() -> MagicMock:
    gateway = MagicMock()
    gateway.search.return_value = VectorSearchResult(
        hits=[_hit("claim-investigation-report.pdf", "Cause of loss was a burst supply line.")],
        search_mode="hybrid",
    )
    return gateway


def _runner(llm, tmp_path: Path) -> TeamRunner:
    single = ClaimAgentRunner(
        llm=llm,
        vector_gateway=_gateway(),
        memory=AgentMemoryStore(tmp_path / "mem.json"),
        mcp=run_race.OfflineToolBus(),
        guardrails_enabled=True,
    )
    return TeamRunner(single=single)


def test_task_lifecycle_rejects_a_backward_jump():
    task = A2ATask(id="t1", specialist_id="loss_investigator", instruction="cause")
    task.transition("working")
    task.transition("completed")
    with pytest.raises(A2AStateError):
        task.transition("working")
    assert task.history == ["submitted", "working", "completed"]


def test_route_simple_cause_uses_one_specialist():
    plan = route_task("What was the cause of the loss?")
    assert plan.workers == ("loss_investigator",)
    assert plan.parallel is False


def test_route_parallel_brief_runs_three_together():
    plan = route_task(
        "Report three things: the cause of the loss, the all-peril deductible, and the net settlement amount."
    )
    assert plan.workers == ("loss_investigator", "coverage_analyst", "settlement_clerk")
    assert plan.parallel is True


def test_route_branching_is_sequential():
    plan = route_task(
        "Investigate this claim. First find the cause of loss. If it is water-related, "
        "check whether flood is excluded. Then give the repair estimate and the net settlement."
    )
    assert plan.parallel is False
    assert "loss_investigator" in plan.workers
    assert "coverage_analyst" in plan.workers
    assert "settlement_clerk" in plan.workers


def test_route_coverage_does_not_fan_out():
    plan = route_task("Is flood excluded, and what is the water backup endorsement limit?")
    assert plan.workers == ("coverage_analyst",)
    assert plan.parallel is False


def test_team_resends_the_task_on_every_handoff(tmp_path: Path):
    task = "What was the cause of the loss?"
    llm = ScriptedLLM(
        [
            "The cause was a burst supply line (claim-investigation-report.pdf).",
            "The cause was a burst supply line.",
        ]
    )
    runner = _runner(llm, tmp_path)
    result = asyncio.run(runner.run_team(task))
    assert result["plan"]["workers"] == ["loss_investigator"]
    assert result["tasks"][0]["history"] == ["submitted", "working", "completed"]
    assert llm.prompts[0].count(task) == 1
    assert llm.prompts[1].count(task) == 1
    assert "loss investigation" in llm.systems[0].lower()
    assert result["metrics"]["contextResendTokens"] == estimate_tokens(task) * 2
    assert result["metrics"]["llmCalls"] == 2
    assert "burst supply line" in result["answer"].lower()


def test_parallel_wall_clock_is_below_the_sum_of_specialists(tmp_path: Path):
    llm = SlowLLM(0.2)
    runner = _runner(llm, tmp_path)
    result = asyncio.run(
        runner.run_team(
            "Report three things: the cause of the loss, the all-peril deductible, "
            "and the net settlement amount."
        )
    )
    wall = result["metrics"]["elapsedMs"]
    worker_sum = result["metrics"]["workerElapsedMs"]
    assert result["plan"]["parallel"] is True
    assert wall < worker_sum * 0.85
    assert wall < 1500


def test_quality_tie_and_higher_team_cost_ships_the_single_agent():
    verdict = score.compare_sides(
        {"quality": 1, "elapsedMs": 400, "estimatedTokens": 800, "costUnits": 2.2},
        {"quality": 1, "elapsedMs": 180, "estimatedTokens": 2200, "costUnits": 4.5},
    )
    assert verdict["ship"] == "single"
    assert verdict["decidedBy"] == "cost"
    assert set(verdict["comparison"]) == {"quality", "elapsedMs", "estimatedTokens", "costUnits"}


def test_quality_gap_ships_the_team_even_when_it_costs_more():
    verdict = score.compare_sides(
        {"quality": 0.5, "elapsedMs": 200, "estimatedTokens": 500, "costUnits": 1.5},
        {"quality": 1.0, "elapsedMs": 300, "estimatedTokens": 900, "costUnits": 3.0},
    )
    assert verdict["ship"] == "team"
    assert verdict["decidedBy"] == "quality"


def test_scripted_suite_uses_the_same_cases_and_reports_four_numbers(tmp_path: Path):
    out = tmp_path / "race.json"
    document = asyncio.run(run_race.run_scripted(out, delay_s=0))
    summary = document["summary"]
    assert summary["cases"] == 4
    for side in ("single", "team"):
        for key in ("quality", "elapsedMs", "estimatedTokens", "costUnits"):
            assert key in summary[side]
    assert summary["verdict"]["ship"] == "team"
    assert summary["verdict"]["decidedBy"] == "cost"
    assert summary["team"]["estimatedTokens"] < summary["single"]["estimatedTokens"]
    by_id = {row["id"]: row for row in document["cases"]}
    assert by_id["simple-cause"]["workers"] == ["loss_investigator"]
    assert by_id["parallel-brief"]["parallel"] is True
    assert by_id["branching"]["parallel"] is False
    for row in document["cases"]:
        assert row["single"]["quality"] == 1
        assert row["team"]["quality"] == 1
        assert row["contextResendTokens"] > 0
    assert json.loads(out.read_text(encoding="utf-8"))["summary"]["verdict"]["rule"]
