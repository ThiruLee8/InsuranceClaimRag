from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.repositories import DocumentRepository
from app.services.agent_memory import AgentMemoryStore, summarise_run
from app.services.agent_tools import (
    TOOL_NAMES,
    ToolContext,
    execute_tool,
    tools_prompt_block,
)
from app.services.llm_service import LLMProvider, OllamaLLMService
from app.services.vector_gateway_client import VectorGatewayClient, VectorSearchHit

logger = get_logger(__name__)

EventCallback = Callable[[dict[str, Any]], Awaitable[None] | None]

REACT_SYSTEM = (
    "You are a claims investigation agent. You work in a loop: think, pick ONE tool, "
    "read the observation, repeat until you can answer.\n\n"
    "Respond with a single JSON object and nothing else:\n"
    '{"thought": "why this tool", "action": "tool_name", "action_input": {}}\n\n'
    "Rules:\n"
    "- One tool per turn.\n"
    "- Prefer a focused search query, not the whole task.\n"
    "- Use list_documents once if you do not know which files exist.\n"
    "- Call finish when observations are enough to answer.\n"
    "- Do not invent document facts. If they are missing, finish and say so.\n"
)

WORKFLOW_SYNTH_SYSTEM = (
    "You are an insurance claims document assistant. "
    "Answer using only the provided observations. Cite document names and pages. "
    "If the answer cannot be found, say so clearly. Do not invent facts."
)

# Known investigation checklist — used when the steps do not depend on the input.
WORKFLOW_STEPS: list[tuple[str, dict[str, str]]] = [
    ("list_documents", {}),
    (
        "search_documents",
        {"query": "insurance policy coverage exclusions deductibles limits endorsements"},
    ),
    (
        "search_documents",
        {"query": "cause of loss investigation timeline damaged property"},
    ),
    (
        "search_documents",
        {"query": "damage assessment estimated repair cost replacement"},
    ),
    (
        "search_documents",
        {"query": "claim settlement amount paid outstanding"},
    ),
]

SAMPLE_TASKS = [
    {
        "id": "investigate",
        "label": "Full investigation (branching)",
        "task": (
            "Investigate this claim. First find the cause of loss. If it is water-related, "
            "check whether the policy covers that water damage and any exclusions. Then find "
            "the repair estimate and what was actually paid. Flag any gaps."
        ),
        "why": "The next search depends on the cause — a genuine agent path.",
    },
    {
        "id": "checklist",
        "label": "Known checklist",
        "task": (
            "Summarize coverage, cause of loss, repair estimate, and settlement from the claim file."
        ),
        "why": "Steps are known in advance — a fixed workflow should win.",
    },
    {
        "id": "simple",
        "label": "Single fact",
        "task": "What was the cause of the loss?",
        "why": "The agent may finish after one search; the workflow still runs the full checklist.",
    },
    {
        "id": "memory",
        "label": "Follow-up (memory)",
        "task": (
            "Using prior investigation notes, what deductible applies and was it subtracted "
            "from the settlement?"
        ),
        "why": "Tests long-term memory. Run an investigation first with the same session id.",
    },
]


@dataclass
class AgentDecision:
    thought: str
    action: str
    action_input: Any
    raw: str = ""

    @property
    def answer(self) -> str:
        data = self.action_input
        if isinstance(data, dict):
            return str(data.get("answer") or data.get("final") or "").strip()
        return str(data or "").strip()


@dataclass
class AgentStep:
    index: int
    thought: str = ""
    action: str = ""
    action_input: Any = None
    observation: str = ""
    elapsed_ms: float = 0.0
    llm_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "thought": self.thought,
            "action": self.action,
            "actionInput": self.action_input,
            "observation": self.observation,
            "elapsedMs": round(self.elapsed_ms, 1),
            "llmCalls": self.llm_calls,
        }


@dataclass
class AgentMetrics:
    steps: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    estimated_tokens: int = 0
    elapsed_ms: float = 0.0
    stop_reason: str = ""
    cost_units: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "llmCalls": self.llm_calls,
            "toolCalls": self.tool_calls,
            "estimatedTokens": self.estimated_tokens,
            "elapsedMs": round(self.elapsed_ms, 1),
            "stopReason": self.stop_reason,
            "costUnits": round(self.cost_units, 3),
        }


@dataclass
class AgentRunResult:
    mode: str
    session_id: str
    task: str
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    sources: list[VectorSearchHit] = field(default_factory=list)
    metrics: AgentMetrics = field(default_factory=AgentMetrics)
    memories_used: list[str] = field(default_factory=list)
    memory_saved: str | None = None
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        seen: set[tuple[str, int | None, str]] = set()
        sources: list[dict[str, Any]] = []
        for hit in self.sources:
            key = (hit.file_name, hit.page_number, hit.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                {
                    "documentId": hit.document_id or None,
                    "chunkId": hit.chunk_id or None,
                    "fileName": hit.file_name,
                    "pageNumber": hit.page_number,
                    "relevanceScore": hit.score,
                    "content": hit.content[:400] if hit.content else None,
                }
            )
        return {
            "mode": self.mode,
            "sessionId": self.session_id,
            "task": self.task,
            "answer": self.answer,
            "steps": [s.to_dict() for s in self.steps],
            "sources": sources,
            "metrics": self.metrics.to_dict(),
            "memoriesUsed": self.memories_used,
            "memorySaved": self.memory_saved,
            "model": self.model,
        }


def estimate_tokens(*parts: str) -> int:
    chars = sum(len(p or "") for p in parts)
    return max(1, chars // 4) if chars else 0


def parse_decision(text: str) -> AgentDecision | None:
    """Parse JSON (preferred) or classic ReAct 'Thought / Action / Action Input' text."""
    if not text or not text.strip():
        return None
    raw = text.strip()
    data = _extract_json(raw)
    if data and isinstance(data, dict) and data.get("action"):
        action = str(data.get("action") or "").strip().lower()
        return AgentDecision(
            thought=str(data.get("thought") or "").strip(),
            action=action,
            action_input=_normalize_input(action, data.get("action_input", data.get("actionInput"))),
            raw=raw,
        )
    thought_m = re.search(r"Thought:\s*(.+?)(?:\n\s*Action:|$)", raw, re.IGNORECASE | re.DOTALL)
    action_m = re.search(r"Action:\s*([A-Za-z_]+)", raw, re.IGNORECASE)
    input_m = re.search(r"Action Input:\s*(.+)$", raw, re.IGNORECASE | re.DOTALL)
    if not action_m:
        return None
    action = action_m.group(1).strip().lower()
    input_raw: Any = input_m.group(1).strip() if input_m else {}
    parsed_input = _extract_json(str(input_raw))
    if parsed_input is not None:
        input_raw = parsed_input
    return AgentDecision(
        thought=(thought_m.group(1).strip() if thought_m else ""),
        action=action,
        action_input=_normalize_input(action, input_raw),
        raw=raw,
    )


def _extract_json(text: str) -> Any | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None


def _normalize_input(action: str, raw: Any) -> Any:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    text = str(raw).strip()
    if action == "finish":
        return {"answer": text}
    if action in {"search_documents", "search_memory"}:
        return {"query": text}
    if action == "save_memory":
        return {"fact": text}
    parsed = _extract_json(text)
    return parsed if parsed is not None else text


def budget_stop(
    *,
    steps: int,
    llm_calls: int,
    elapsed_s: float,
    max_steps: int,
    max_llm_calls: int,
    max_seconds: float,
) -> str | None:
    if max_steps and steps >= max_steps:
        return "max_steps"
    if max_llm_calls and llm_calls >= max_llm_calls:
        return "max_llm_calls"
    if max_seconds > 0 and elapsed_s >= max_seconds:
        return "max_seconds"
    return None


def _compact_scratchpad(parts: list[str], limit: int = 8000) -> str:
    """Short-term memory compression: keep the first and last turns if the pad grows."""
    joined = "\n\n".join(parts)
    if len(joined) <= limit or len(parts) <= 3:
        return joined
    kept = [parts[0], "[earlier steps omitted — see log]", *parts[-3:]]
    return "\n\n".join(kept)


def _fallback_answer(steps: list[AgentStep]) -> str:
    observations = [s.observation for s in steps if s.observation and s.action != "finish"]
    if not observations:
        return (
            "Stopped before a final answer. I could not find sufficient information "
            "in the provided documents."
        )
    last = observations[-1]
    return (
        "Stopped by a safety budget before finish. Last observation:\n"
        f"{last[:1200]}"
    )


def compare_runs(agent: AgentRunResult, workflow: AgentRunResult) -> dict[str, Any]:
    a, w = agent.metrics, workflow.metrics
    agent_ok = a.stop_reason == "finished"
    workflow_ok = w.stop_reason == "finished"

    def pack(metric: str, av: float, wv: float) -> dict[str, float]:
        return {"agent": av, "workflow": wv, "delta": round(av - wv, 3)}

    comparison = {
        "elapsedMs": pack("elapsedMs", a.elapsed_ms, w.elapsed_ms),
        "llmCalls": pack("llmCalls", a.llm_calls, w.llm_calls),
        "toolCalls": pack("toolCalls", a.tool_calls, w.tool_calls),
        "estimatedTokens": pack("estimatedTokens", a.estimated_tokens, w.estimated_tokens),
        "steps": pack("steps", a.steps, w.steps),
        "costUnits": pack("costUnits", a.cost_units, w.cost_units),
        "stopReason": {"agent": a.stop_reason, "workflow": w.stop_reason},
    }

    if agent_ok and not workflow_ok:
        ship, winner = "agent", "agent"
        rationale = (
            "The agent finished and the workflow did not. Ship the agent for this task."
        )
    elif workflow_ok and not agent_ok:
        ship, winner = "workflow", "workflow"
        rationale = (
            "The fixed sequence finished and the agent hit a budget. "
            "Ship the workflow — it is more reliable when the steps are known."
        )
    elif workflow_ok and agent_ok:
        workflow_cheaper = w.llm_calls <= a.llm_calls and w.elapsed_ms <= a.elapsed_ms * 1.05
        agent_narrower = a.tool_calls < w.tool_calls and a.llm_calls <= w.llm_calls + 1
        if workflow_cheaper and not agent_narrower:
            ship, winner = "workflow", "workflow"
            rationale = (
                "Both finished. The fixed sequence used fewer (or equal) LLM calls and was as fast "
                "or faster. I would ship the workflow for this claims checklist — the path is known, "
                "so an agent loop only adds planning cost and failure modes."
            )
        elif agent_narrower and a.elapsed_ms <= w.elapsed_ms:
            ship, winner = "agent", "agent"
            rationale = (
                "Both finished. The agent skipped unnecessary checklist searches and was not slower. "
                "I would ship the agent when the next step depends on what the last one found "
                "(for example: cause of loss → then only the matching exclusion)."
            )
        elif w.elapsed_ms <= a.elapsed_ms:
            ship, winner = "workflow", "workflow"
            rationale = (
                "Both finished. The workflow was faster. I would ship the fixed sequence for this "
                "task and keep the agent for genuinely branching investigations."
            )
        else:
            ship, winner = "agent", "agent"
            rationale = (
                "Both finished. The agent was faster on this input. I would still default to a "
                "workflow in production unless the path truly changes with the findings."
            )
    else:
        ship, winner = "workflow", "neither"
        rationale = (
            "Neither run finished cleanly. I would still ship a fixed workflow first — "
            "it is easier to debug than an open loop."
        )

    return {
        "winner": winner,
        "ship": ship,
        "rationale": rationale,
        "comparison": comparison,
    }


class ClaimAgentRunner:
    """Hand-rolled ReAct loop + fixed claims workflow. No LangChain/LangGraph."""

    def __init__(
        self,
        *,
        llm: LLMProvider | None = None,
        vector_gateway: VectorGatewayClient | None = None,
        document_repo: DocumentRepository | None = None,
        memory: AgentMemoryStore | None = None,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        self.settings = get_settings()
        self.llm = llm or OllamaLLMService()
        self.vector_gateway = vector_gateway or VectorGatewayClient()
        self.document_repo = document_repo
        self.memory = memory or AgentMemoryStore()
        self.time_fn = time_fn or time.perf_counter

    def _ctx(self, session_id: str, task: str) -> ToolContext:
        return ToolContext(
            session_id=session_id,
            task=task,
            document_repo=self.document_repo,
            vector_gateway=self.vector_gateway,
            memory=self.memory,
        )

    def _finalize_metrics(
        self,
        *,
        steps: list[AgentStep],
        llm_calls: int,
        tool_calls: int,
        tokens: int,
        started: float,
        stop_reason: str,
    ) -> AgentMetrics:
        elapsed_ms = (self.time_fn() - started) * 1000
        return AgentMetrics(
            steps=len(steps),
            llm_calls=llm_calls,
            tool_calls=tool_calls,
            estimated_tokens=tokens,
            elapsed_ms=elapsed_ms,
            stop_reason=stop_reason,
            cost_units=llm_calls + tokens / 4000,
        )

    async def _emit(self, on_event: EventCallback | None, event: dict[str, Any]) -> None:
        logger.info("agent_event", **{k: event[k] for k in ("type", "mode") if k in event})
        if on_event is None:
            return
        result = on_event(event)
        if hasattr(result, "__await__"):
            await result  # type: ignore[misc]

    def _maybe_persist(
        self,
        *,
        persist: bool,
        session_id: str,
        task: str,
        answer: str,
        facts: list[str],
    ) -> str | None:
        if not persist or not answer:
            return None
        summary = summarise_run(task, answer, facts)
        entry = self.memory.add(
            session_id=session_id,
            task=task,
            summary=summary,
            facts=facts,
        )
        return entry.id

    async def run_agent(
        self,
        task: str,
        *,
        session_id: str | None = None,
        model: str | None = None,
        max_steps: int = 8,
        max_llm_calls: int = 10,
        max_seconds: float = 90.0,
        persist_memory: bool = True,
        on_event: EventCallback | None = None,
    ) -> AgentRunResult:
        session_id = session_id or str(uuid.uuid4())
        selected_model = (model or self.settings.ollama_model).strip() or self.settings.ollama_model
        ctx = self._ctx(session_id, task)
        recalled = self.memory.recall(task, session_id=session_id, limit=4)
        memories_used = [e.summary for e, _ in recalled]
        started = self.time_fn()
        steps: list[AgentStep] = []
        scratch: list[str] = []
        llm_calls = 0
        tool_calls = 0
        tokens = 0
        parse_fails = 0
        recent_sigs: list[str] = []
        facts: list[str] = []

        await self._emit(
            on_event,
            {"type": "run_start", "mode": "agent", "sessionId": session_id, "task": task},
        )

        memory_block = ""
        if recalled:
            memory_block = "Prior session memory:\n" + "\n".join(
                f"- {e.summary}" for e, _ in recalled
            )

        # The whole agent is this loop: plan → act → observe → repeat until done.
        while True:
            elapsed_s = self.time_fn() - started
            stop = budget_stop(
                steps=len(steps),
                llm_calls=llm_calls,
                elapsed_s=elapsed_s,
                max_steps=max_steps,
                max_llm_calls=max_llm_calls,
                max_seconds=max_seconds,
            )
            if stop:
                answer = _fallback_answer(steps)
                result = self._finish(
                    mode="agent",
                    session_id=session_id,
                    task=task,
                    answer=answer,
                    steps=steps,
                    ctx=ctx,
                    llm_calls=llm_calls,
                    tool_calls=tool_calls,
                    tokens=tokens,
                    started=started,
                    stop_reason=stop,
                    memories_used=memories_used,
                    persist_memory=persist_memory,
                    facts=facts,
                    model=selected_model,
                )
                await self._emit(on_event, {"type": "done", "mode": "agent", "result": result.to_dict()})
                return result

            prompt = self._build_react_prompt(task, scratch, memory_block)
            raw = await self.llm.generate(prompt=prompt, system=REACT_SYSTEM, model=selected_model)
            llm_calls += 1
            tokens += estimate_tokens(REACT_SYSTEM, prompt, raw)
            logger.info("agent_plan", step=len(steps) + 1, llm_calls=llm_calls, chars=len(raw or ""))

            decision = parse_decision(raw)
            if decision is None or decision.action not in TOOL_NAMES:
                parse_fails += 1
                hint = (
                    "Your last output was not valid. Reply with JSON only: "
                    '{"thought":"...","action":"search_documents","action_input":{"query":"..."}}'
                )
                scratch.append(hint + f" Got: {(raw or '')[:240]}")
                step = AgentStep(
                    index=len(steps) + 1,
                    thought="(parse failed)",
                    action="invalid",
                    action_input={"raw": (raw or "")[:400]},
                    observation=hint,
                    elapsed_ms=(self.time_fn() - started) * 1000,
                    llm_calls=llm_calls,
                )
                steps.append(step)
                await self._emit(on_event, {"type": "step", "mode": "agent", "step": step.to_dict()})
                if parse_fails >= 3:
                    answer = _fallback_answer(steps)
                    result = self._finish(
                        mode="agent",
                        session_id=session_id,
                        task=task,
                        answer=answer,
                        steps=steps,
                        ctx=ctx,
                        llm_calls=llm_calls,
                        tool_calls=tool_calls,
                        tokens=tokens,
                        started=started,
                        stop_reason="parse_failures",
                        memories_used=memories_used,
                        persist_memory=False,
                        facts=facts,
                        model=selected_model,
                    )
                    await self._emit(
                        on_event, {"type": "done", "mode": "agent", "result": result.to_dict()}
                    )
                    return result
                continue

            parse_fails = 0
            step = AgentStep(
                index=len(steps) + 1,
                thought=decision.thought,
                action=decision.action,
                action_input=decision.action_input,
                elapsed_ms=(self.time_fn() - started) * 1000,
                llm_calls=llm_calls,
            )
            await self._emit(on_event, {"type": "step", "mode": "agent", "step": step.to_dict()})

            if decision.action == "finish":
                step.observation = "(finished)"
                steps.append(step)
                answer = decision.answer or _fallback_answer(steps)
                result = self._finish(
                    mode="agent",
                    session_id=session_id,
                    task=task,
                    answer=answer,
                    steps=steps,
                    ctx=ctx,
                    llm_calls=llm_calls,
                    tool_calls=tool_calls,
                    tokens=tokens,
                    started=started,
                    stop_reason="finished",
                    memories_used=memories_used,
                    persist_memory=persist_memory,
                    facts=facts,
                    model=selected_model,
                )
                await self._emit(on_event, {"type": "done", "mode": "agent", "result": result.to_dict()})
                return result

            sig = f"{decision.action}:{json.dumps(decision.action_input, sort_keys=True, default=str)}"
            recent_sigs.append(sig)
            if len(recent_sigs) >= 3 and len(set(recent_sigs[-3:])) == 1:
                step.observation = "Repeated the same tool and input three times — stopping."
                steps.append(step)
                await self._emit(on_event, {"type": "step", "mode": "agent", "step": step.to_dict()})
                result = self._finish(
                    mode="agent",
                    session_id=session_id,
                    task=task,
                    answer=_fallback_answer(steps),
                    steps=steps,
                    ctx=ctx,
                    llm_calls=llm_calls,
                    tool_calls=tool_calls,
                    tokens=tokens,
                    started=started,
                    stop_reason="repeated_action",
                    memories_used=memories_used,
                    persist_memory=False,
                    facts=facts,
                    model=selected_model,
                )
                await self._emit(on_event, {"type": "done", "mode": "agent", "result": result.to_dict()})
                return result

            observation = execute_tool(decision.action, decision.action_input, ctx)
            tool_calls += 1
            if decision.action == "save_memory":
                fact = ""
                if isinstance(decision.action_input, dict):
                    fact = str(decision.action_input.get("fact") or "")
                if fact:
                    facts.append(fact)
            step.observation = observation
            step.elapsed_ms = (self.time_fn() - started) * 1000
            steps.append(step)
            scratch.append(
                f"Thought: {decision.thought}\nAction: {decision.action} {decision.action_input}\n"
                f"Observation: {observation}"
            )
            logger.info(
                "agent_act",
                action=decision.action,
                step=step.index,
                observation_chars=len(observation),
            )
            await self._emit(on_event, {"type": "step", "mode": "agent", "step": step.to_dict()})

    def _build_react_prompt(self, task: str, scratch: list[str], memory_block: str) -> str:
        pad = _compact_scratchpad(scratch)
        parts = [
            f"Task:\n{task.strip()}",
            "Tools:\n" + tools_prompt_block(),
        ]
        if memory_block:
            parts.append(memory_block)
        if pad:
            parts.append("Scratchpad (short-term memory):\n" + pad)
        else:
            parts.append("Scratchpad (short-term memory):\n(empty — first step)")
        parts.append("Next JSON decision:")
        return "\n\n".join(parts)

    def _finish(
        self,
        *,
        mode: str,
        session_id: str,
        task: str,
        answer: str,
        steps: list[AgentStep],
        ctx: ToolContext,
        llm_calls: int,
        tool_calls: int,
        tokens: int,
        started: float,
        stop_reason: str,
        memories_used: list[str],
        persist_memory: bool,
        facts: list[str],
        model: str,
    ) -> AgentRunResult:
        metrics = self._finalize_metrics(
            steps=steps,
            llm_calls=llm_calls,
            tool_calls=tool_calls,
            tokens=tokens,
            started=started,
            stop_reason=stop_reason,
        )
        saved = None
        if stop_reason == "finished":
            saved = self._maybe_persist(
                persist=persist_memory,
                session_id=session_id,
                task=task,
                answer=answer,
                facts=facts,
            )
        logger.info(
            "agent_run_complete",
            mode=mode,
            stop_reason=stop_reason,
            steps=metrics.steps,
            llm_calls=metrics.llm_calls,
            elapsed_ms=round(metrics.elapsed_ms, 1),
        )
        return AgentRunResult(
            mode=mode,
            session_id=session_id,
            task=task,
            answer=answer,
            steps=steps,
            sources=list(ctx.sources),
            metrics=metrics,
            memories_used=memories_used,
            memory_saved=saved,
            model=model,
        )

    async def run_workflow(
        self,
        task: str,
        *,
        session_id: str | None = None,
        model: str | None = None,
        persist_memory: bool = True,
        on_event: EventCallback | None = None,
    ) -> AgentRunResult:
        """Fixed sequence: list → 4 searches → one LLM synthesis. No tool-choice loop."""
        session_id = session_id or str(uuid.uuid4())
        selected_model = (model or self.settings.ollama_model).strip() or self.settings.ollama_model
        ctx = self._ctx(session_id, task)
        started = self.time_fn()
        steps: list[AgentStep] = []
        tokens = 0
        llm_calls = 0
        observations: list[str] = []

        await self._emit(
            on_event,
            {"type": "run_start", "mode": "workflow", "sessionId": session_id, "task": task},
        )

        for action, action_input in WORKFLOW_STEPS:
            observation = execute_tool(action, action_input, ctx)
            step = AgentStep(
                index=len(steps) + 1,
                thought=f"Fixed workflow step: {action}",
                action=action,
                action_input=action_input,
                observation=observation,
                elapsed_ms=(self.time_fn() - started) * 1000,
                llm_calls=0,
            )
            steps.append(step)
            observations.append(f"{action}: {observation}")
            logger.info("workflow_step", action=action, step=step.index)
            await self._emit(on_event, {"type": "step", "mode": "workflow", "step": step.to_dict()})

        memory_hits = self.memory.recall(task, session_id=session_id, limit=3)
        memories_used = [e.summary for e, _ in memory_hits]
        if memory_hits:
            mem_obs = "\n".join(f"- {e.summary}" for e, _ in memory_hits)
            observations.append("search_memory:\n" + mem_obs)
            step = AgentStep(
                index=len(steps) + 1,
                thought="Fixed workflow: recall session memory",
                action="search_memory",
                action_input={"query": task[:180]},
                observation=mem_obs,
                elapsed_ms=(self.time_fn() - started) * 1000,
            )
            steps.append(step)
            await self._emit(on_event, {"type": "step", "mode": "workflow", "step": step.to_dict()})

        synth_prompt = (
            f"User task:\n{task.strip()}\n\n"
            "Observations from the fixed investigation checklist:\n"
            + "\n\n".join(observations)
            + "\n\nWrite the final answer."
        )
        answer = await self.llm.generate(
            prompt=synth_prompt,
            system=WORKFLOW_SYNTH_SYSTEM,
            model=selected_model,
        )
        llm_calls = 1
        tokens += estimate_tokens(WORKFLOW_SYNTH_SYSTEM, synth_prompt, answer)
        if not (answer or "").strip():
            answer = "I could not find sufficient information in the provided documents."
        finish_step = AgentStep(
            index=len(steps) + 1,
            thought="Synthesize the checklist into one answer",
            action="finish",
            action_input={"answer": answer[:200]},
            observation="(finished)",
            elapsed_ms=(self.time_fn() - started) * 1000,
            llm_calls=llm_calls,
        )
        steps.append(finish_step)
        await self._emit(on_event, {"type": "step", "mode": "workflow", "step": finish_step.to_dict()})

        result = self._finish(
            mode="workflow",
            session_id=session_id,
            task=task,
            answer=answer.strip(),
            steps=steps,
            ctx=ctx,
            llm_calls=llm_calls,
            tool_calls=len(WORKFLOW_STEPS),
            tokens=tokens,
            started=started,
            stop_reason="finished",
            memories_used=memories_used,
            persist_memory=persist_memory,
            facts=[],
            model=selected_model,
        )
        await self._emit(on_event, {"type": "done", "mode": "workflow", "result": result.to_dict()})
        return result

    async def run_race(
        self,
        task: str,
        *,
        session_id: str | None = None,
        model: str | None = None,
        max_steps: int = 8,
        max_llm_calls: int = 10,
        max_seconds: float = 90.0,
        persist_memory: bool = True,
        on_event: EventCallback | None = None,
    ) -> dict[str, Any]:
        session_id = session_id or str(uuid.uuid4())
        await self._emit(on_event, {"type": "race_start", "sessionId": session_id, "task": task})
        agent = await self.run_agent(
            task,
            session_id=session_id,
            model=model,
            max_steps=max_steps,
            max_llm_calls=max_llm_calls,
            max_seconds=max_seconds,
            persist_memory=persist_memory,
            on_event=on_event,
        )
        workflow = await self.run_workflow(
            task,
            session_id=session_id,
            model=model,
            persist_memory=False,
            on_event=on_event,
        )
        judged = compare_runs(agent, workflow)
        payload = {
            "task": task,
            "sessionId": session_id,
            "agent": agent.to_dict(),
            "workflow": workflow.to_dict(),
            **judged,
        }
        await self._emit(on_event, {"type": "race", "result": payload})
        return payload

    async def stream(
        self,
        *,
        mode: str,
        task: str,
        session_id: str | None = None,
        model: str | None = None,
        max_steps: int = 8,
        max_llm_calls: int = 10,
        max_seconds: float = 90.0,
        persist_memory: bool = True,
    ) -> AsyncIterator[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        async def on_event(event: dict[str, Any]) -> None:
            await queue.put(event)

        async def runner() -> None:
            try:
                kwargs = {
                    "session_id": session_id,
                    "model": model,
                    "persist_memory": persist_memory,
                    "on_event": on_event,
                }
                if mode == "workflow":
                    await self.run_workflow(task, **kwargs)
                elif mode == "race":
                    await self.run_race(
                        task,
                        max_steps=max_steps,
                        max_llm_calls=max_llm_calls,
                        max_seconds=max_seconds,
                        **kwargs,
                    )
                else:
                    await self.run_agent(
                        task,
                        max_steps=max_steps,
                        max_llm_calls=max_llm_calls,
                        max_seconds=max_seconds,
                        **kwargs,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.error("agent_stream_failed", error=str(exc))
                await queue.put({"type": "error", "message": str(exc)})
            finally:
                await queue.put(None)

        worker = asyncio.create_task(runner())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
        finally:
            await worker
