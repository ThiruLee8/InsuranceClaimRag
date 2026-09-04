"""Tests for the hand-rolled ReAct agent loop, budgets, workflow, race, and memory."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from app.services.agent_loop import ClaimAgentRunner, budget_stop, compare_runs, parse_decision
from app.services.agent_memory import AgentMemoryStore, summarise_run
from app.services.agent_tools import ToolContext, execute_tool, format_hits
from app.services.vector_gateway_client import VectorSearchHit, VectorSearchResult


class ScriptedLLM:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls = 0
        self.prompts: list[str] = []

    async def generate(self, *, prompt: str, system: str | None = None, model: str | None = None) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        if not self.replies:
            return json.dumps(
                {
                    "thought": "fallback finish",
                    "action": "finish",
                    "action_input": {"answer": "fallback"},
                }
            )
        return self.replies.pop(0)


def _hit(name: str, content: str, score: float = 0.9, page: int = 1) -> VectorSearchHit:
    return VectorSearchHit(
        chunk_id=str(uuid4()),
        document_id=str(uuid4()),
        file_name=name,
        page_number=page,
        chunk_index=0,
        content=content,
        score=score,
    )


def _gateway(hits: list[VectorSearchHit] | None = None) -> MagicMock:
    gw = MagicMock()
    gw.search.return_value = VectorSearchResult(
        hits=hits or [_hit("claim-investigation-report.pdf", "Cause of loss was a burst pipe.")],
        search_mode="hybrid",
    )
    return gw


def _repo(names: list[str] | None = None) -> MagicMock:
    repo = MagicMock()
    docs = []
    for name in names or ["policy-schedule.pdf", "claim-investigation-report.pdf"]:
        doc = MagicMock()
        doc.OriginalFileName = name
        doc.Status = MagicMock(value="Processed")
        doc.PageCount = 4
        docs.append(doc)
    repo.list.return_value = (docs, len(docs))
    return repo


def test_parse_decision_json_and_react_text():
    json_decision = parse_decision(
        '{"thought": "look up cause", "action": "search_documents", '
        '"action_input": {"query": "cause of loss"}}'
    )
    assert json_decision is not None
    assert json_decision.action == "search_documents"
    assert json_decision.action_input["query"] == "cause of loss"

    fenced = parse_decision(
        '```json\n{"thought": "done", "action": "finish", "action_input": {"answer": "Burst pipe."}}\n```'
    )
    assert fenced is not None
    assert fenced.action == "finish"
    assert "Burst" in fenced.answer

    react = parse_decision(
        "Thought: I should list files first\nAction: list_documents\nAction Input: {}"
    )
    assert react is not None
    assert react.action == "list_documents"

    assert parse_decision("sorry I cannot") is None


def test_budget_stop_reasons():
    assert (
        budget_stop(steps=8, llm_calls=1, elapsed_s=1, max_steps=8, max_llm_calls=10, max_seconds=90)
        == "max_steps"
    )
    assert (
        budget_stop(steps=1, llm_calls=10, elapsed_s=1, max_steps=8, max_llm_calls=10, max_seconds=90)
        == "max_llm_calls"
    )
    assert (
        budget_stop(steps=1, llm_calls=1, elapsed_s=91, max_steps=8, max_llm_calls=10, max_seconds=90)
        == "max_seconds"
    )
    assert budget_stop(steps=1, llm_calls=1, elapsed_s=1, max_steps=8, max_llm_calls=10, max_seconds=90) is None


def test_memory_save_and_recall(tmp_path: Path):
    store = AgentMemoryStore(tmp_path / "mem.json")
    store.add(
        session_id="s1",
        task="Investigate water claim",
        summary=summarise_run(
            "Investigate water claim", "Burst pipe; $8,400 estimate; $1,000 deductible."
        ),
        facts=["deductible is $1,000"],
    )
    hits = store.recall("what deductible applies", session_id="s1")
    assert hits
    assert "deductible" in hits[0][0].summary.lower() or hits[0][0].facts
    assert store.clear("s1") == 1
    assert store.recall("deductible", session_id="s1") == []


def test_execute_search_and_list_tools():
    ctx = ToolContext(
        session_id="s",
        task="cause?",
        document_repo=_repo(),
        vector_gateway=_gateway(),
    )
    listed = execute_tool("list_documents", {}, ctx)
    assert "policy-schedule.pdf" in listed
    found = execute_tool("search_documents", {"query": "cause of loss"}, ctx)
    assert "burst pipe" in found.lower()
    assert ctx.sources


@pytest.mark.asyncio
async def test_agent_completes_multistep_task(tmp_path: Path):
    llm = ScriptedLLM(
        [
            json.dumps(
                {
                    "thought": "See which files exist",
                    "action": "list_documents",
                    "action_input": {},
                }
            ),
            json.dumps(
                {
                    "thought": "Find the cause",
                    "action": "search_documents",
                    "action_input": {"query": "cause of loss"},
                }
            ),
            json.dumps(
                {
                    "thought": "Check water coverage because the cause is a burst pipe",
                    "action": "search_documents",
                    "action_input": {"query": "water damage coverage exclusions"},
                }
            ),
            json.dumps(
                {
                    "thought": "Enough to answer",
                    "action": "finish",
                    "action_input": {
                        "answer": "Burst pipe caused the loss. Policy covers sudden water damage."
                    },
                }
            ),
        ]
    )
    runner = ClaimAgentRunner(
        llm=llm,
        vector_gateway=_gateway(
            [
                _hit("claim-investigation-report.pdf", "Cause of loss was a burst pipe."),
                _hit("policy-schedule.pdf", "Sudden water damage from burst pipes is covered."),
            ]
        ),
        document_repo=_repo(),
        memory=AgentMemoryStore(tmp_path / "mem.json"),
    )
    result = await runner.run_agent(
        "Investigate whether this water-damage claim is covered and what caused it.",
        session_id="sess-1",
        persist_memory=True,
    )
    assert result.metrics.stop_reason == "finished"
    assert result.metrics.steps == 4
    assert [s.action for s in result.steps] == [
        "list_documents",
        "search_documents",
        "search_documents",
        "finish",
    ]
    assert "Burst pipe" in result.answer
    assert result.memory_saved
    assert llm.calls == 4


@pytest.mark.asyncio
async def test_agent_stops_on_max_steps(tmp_path: Path):
    llm = ScriptedLLM(
        [
            json.dumps(
                {
                    "thought": "keep searching",
                    "action": "search_documents",
                    "action_input": {"query": f"query {i}"},
                }
            )
            for i in range(12)
        ]
    )
    runner = ClaimAgentRunner(
        llm=llm,
        vector_gateway=_gateway(),
        document_repo=_repo(),
        memory=AgentMemoryStore(tmp_path / "mem.json"),
    )
    result = await runner.run_agent("Find everything", max_steps=3, persist_memory=False)
    assert result.metrics.stop_reason == "max_steps"
    assert result.metrics.steps == 3
    assert result.metrics.llm_calls == 3


@pytest.mark.asyncio
async def test_agent_stops_on_repeated_action(tmp_path: Path):
    same = json.dumps(
        {
            "thought": "search again",
            "action": "search_documents",
            "action_input": {"query": "cause of loss"},
        }
    )
    runner = ClaimAgentRunner(
        llm=ScriptedLLM([same, same, same, same]),
        vector_gateway=_gateway(),
        document_repo=_repo(),
        memory=AgentMemoryStore(tmp_path / "mem.json"),
    )
    result = await runner.run_agent("What caused the loss?", max_steps=8, persist_memory=False)
    assert result.metrics.stop_reason == "repeated_action"


@pytest.mark.asyncio
async def test_workflow_is_fixed_sequence(tmp_path: Path):
    llm = ScriptedLLM(["Coverage A, burst pipe, $8400 repairs, $7200 paid."])
    runner = ClaimAgentRunner(
        llm=llm,
        vector_gateway=_gateway(),
        document_repo=_repo(),
        memory=AgentMemoryStore(tmp_path / "mem.json"),
    )
    result = await runner.run_workflow(
        "Summarize coverage, cause, repair estimate, and settlement.",
        persist_memory=False,
    )
    assert result.metrics.stop_reason == "finished"
    assert result.metrics.llm_calls == 1
    assert result.steps[0].action == "list_documents"
    assert result.steps[1].action == "search_documents"
    assert result.steps[-1].action == "finish"
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_race_returns_numbers_and_ship_choice(tmp_path: Path):
    agent_llm = ScriptedLLM(
        [
            json.dumps(
                {
                    "thought": "search cause",
                    "action": "search_documents",
                    "action_input": {"query": "cause of loss"},
                }
            ),
            json.dumps(
                {
                    "thought": "done",
                    "action": "finish",
                    "action_input": {"answer": "Burst pipe."},
                }
            ),
            "Coverage, cause, estimate, settlement summarised.",
        ]
    )
    runner = ClaimAgentRunner(
        llm=agent_llm,
        vector_gateway=_gateway(),
        document_repo=_repo(),
        memory=AgentMemoryStore(tmp_path / "mem.json"),
    )
    payload = await runner.run_race("What was the cause of the loss?", persist_memory=False)
    assert payload["agent"]["metrics"]["llmCalls"] == 2
    assert payload["workflow"]["metrics"]["llmCalls"] == 1
    assert payload["comparison"]["llmCalls"]["agent"] == 2
    assert payload["comparison"]["llmCalls"]["workflow"] == 1
    assert payload["ship"] in {"agent", "workflow"}
    assert payload["rationale"]


def test_compare_runs_ships_workflow_when_cheaper():
    from app.services.agent_loop import AgentMetrics, AgentRunResult

    agent = AgentRunResult(
        mode="agent",
        session_id="x",
        task="t",
        answer="a",
        metrics=AgentMetrics(
            steps=6,
            llm_calls=6,
            tool_calls=5,
            estimated_tokens=2000,
            elapsed_ms=4000,
            stop_reason="finished",
            cost_units=6.5,
        ),
    )
    workflow = AgentRunResult(
        mode="workflow",
        session_id="x",
        task="t",
        answer="a",
        metrics=AgentMetrics(
            steps=6,
            llm_calls=1,
            tool_calls=5,
            estimated_tokens=800,
            elapsed_ms=1500,
            stop_reason="finished",
            cost_units=1.2,
        ),
    )
    judged = compare_runs(agent, workflow)
    assert judged["ship"] == "workflow"
    assert "fixed sequence" in judged["rationale"].lower() or "workflow" in judged["rationale"].lower()


def test_format_hits_truncates():
    text = format_hits([_hit("a.pdf", "word " * 200, page=3)])
    assert "a.pdf" in text
    assert "p.3" in text
