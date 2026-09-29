"""Race the specialist team against the single claims agent on the same cases.

Scripted (no Ollama, no Chroma) — retrieval is a fixed corpus, each model call
waits `--delay` seconds, and the answer is the retrieved text:

  python eval/multi_agent/run_race.py --scripted --delay 0.05

Live (Ollama + the document index):

  python eval/multi_agent/run_race.py --live
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(HERE))

from score import summarize  # noqa: E402

from app.services.agent_loop import ClaimAgentRunner  # noqa: E402
from app.services.agent_memory import AgentMemoryStore  # noqa: E402
from app.services.multi_agent import (  # noqa: E402
    OBS_MARKER,
    REPORTS_MARKER,
    TeamRunner,
    load_cases,
    route_task,
    search_query_for,
)
from app.services.vector_gateway_client import VectorSearchHit, VectorSearchResult  # noqa: E402


class OfflineToolBus:
    """Keeps the race on the injected search gateway instead of the shared MCP client."""

    connected = False
    trusted_tools: list[Any] = []
    last_error = None

    async def ensure_connected(self) -> None:
        return None

    def trusted_names(self) -> set[str]:
        return set()

    def tool_schemas(self) -> dict[str, Any]:
        return {}

    def status_dict(self) -> dict[str, Any]:
        return {
            "transport": "offline",
            "serverName": "claims-docs",
            "connected": False,
            "whereAiRuns": "On the host.",
            "whereAiDoesNotRun": "The tool server.",
            "endpoint": "/mcp",
            "stdio": "",
            "authRequired": False,
            "tools": [],
            "resources": [],
            "prompts": [],
            "handshake": [],
            "lastError": None,
            "roles": {},
        }


def _hit(name: str, content: str) -> VectorSearchHit:
    return VectorSearchHit(
        chunk_id=str(uuid4()),
        document_id=str(uuid4()),
        file_name=name,
        page_number=1,
        chunk_index=0,
        content=content,
        score=0.91,
    )


LOSS_HIT = _hit(
    "claim-investigation-report.pdf",
    "The cause of the loss was a burst flexible supply line. Sudden and accidental water discharge.",
)
COVERAGE_HIT = _hit(
    "policy-schedule.pdf",
    "Flood remains excluded. Gradual seepage remains excluded. "
    "Water Backup and Sump Overflow endorsement limit USD 10,000. All Peril Deductible: USD 1,000.",
)
SETTLEMENT_HIT = _hit(
    "claim-settlement-letter.pdf",
    "Contractor repair estimate USD 18,450. Net settlement: USD 19,550.",
)


def scripted_gateway() -> MagicMock:
    gateway = MagicMock()

    def search(question: str, **_kwargs: Any) -> VectorSearchResult:
        query = (question or "").lower()
        hits: list[VectorSearchHit] = []
        if any(key in query for key in ("burst", "cause of loss", "timeline")):
            hits.append(LOSS_HIT)
        if any(key in query for key in ("deductible", "flood", "endorsement", "exclusion")):
            hits.append(COVERAGE_HIT)
        if any(key in query for key in ("settlement", "repair estimate", "amount paid")):
            hits.append(SETTLEMENT_HIT)
        if not hits:
            hits.append(LOSS_HIT)
        return VectorSearchResult(hits=hits, search_mode="hybrid")

    gateway.search.side_effect = search
    return gateway


class ExtractiveLLM:
    """Stand-in model: wait `delay_s`, then answer with the retrieved text.

    The single agent still has to choose searches. It uses the same specialist
    queries the team uses, one at a time, then finishes with those observations.
    """

    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s
        self._task = ""
        self._queue: list[str] = []

    async def generate(self, *, prompt: str, system: str | None = None, model: str | None = None) -> str:
        if self.delay_s > 0:
            await asyncio.sleep(self.delay_s)
        if system and "claims investigation agent" in system:
            return self._react(prompt)
        return _echo(prompt)

    def _react(self, prompt: str) -> str:
        task = _task_from_prompt(prompt)
        if task != self._task:
            self._task = task
            plan = route_task(task)
            self._queue = [search_query_for(worker, task) for worker in plan.workers]
        if self._queue:
            query = self._queue.pop(0)
            return json.dumps(
                {
                    "thought": "look up one fact",
                    "action": "search_documents",
                    "action_input": {"query": query},
                }
            )
        evidence = _observations(prompt) or (
            "I could not find sufficient information in the provided documents."
        )
        return json.dumps(
            {
                "thought": "the observations cover the task",
                "action": "finish",
                "action_input": {"answer": evidence},
            }
        )


def _task_from_prompt(prompt: str) -> str:
    marker = "Task:\n"
    start = prompt.find(marker)
    if start < 0:
        return ""
    return prompt[start + len(marker) :].split("\n\n", 1)[0].strip()


def _observations(prompt: str) -> str:
    import re

    parts = re.findall(
        r"Observation: (.*?)(?=\n\nThought:|\n\nNext JSON decision:|\Z)",
        prompt,
        flags=re.DOTALL,
    )
    return "\n".join(part.strip() for part in parts if part.strip())


def _echo(prompt: str) -> str:
    if OBS_MARKER in prompt:
        body = prompt.split(OBS_MARKER, 1)[1]
        body = body.split("\n\nWrite a short finding", 1)[0]
        return body.strip()
    if REPORTS_MARKER in prompt:
        body = prompt.split(REPORTS_MARKER, 1)[1]
        body = body.split("\n\nAnswer the original task", 1)[0]
        return body.strip()
    return "I could not find sufficient information in the provided documents."


def _clip(text: str, limit: int = 700) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1] + "…"


async def run_scripted(out: Path, *, delay_s: float) -> dict[str, Any]:
    memory = AgentMemoryStore(out.parent / "_scripted_memory.json")
    runner = TeamRunner(
        ClaimAgentRunner(
            llm=ExtractiveLLM(delay_s),
            vector_gateway=scripted_gateway(),
            memory=memory,
            mcp=OfflineToolBus(),
            guardrails_enabled=True,
        )
    )
    rows: list[dict[str, Any]] = []
    for case in load_cases():
        payload = await runner.race(
            case["task"],
            case_id=case["id"],
            must_contain_any=case["must_contain_any"],
            max_steps=8,
            max_llm_calls=10,
            max_seconds=30,
        )
        single = payload["single"]
        team = payload["team"]
        rows.append(
            {
                "id": case["id"],
                "label": case["label"],
                "task": case["task"],
                "why": case["why"],
                "parallel": bool(team["plan"]["parallel"]),
                "workers": list(team["plan"]["workers"]),
                "planReason": team["plan"]["reason"],
                "contextResendTokens": team["metrics"]["contextResendTokens"],
                "single": {
                    **single["summary"],
                    "llmCalls": single["metrics"]["llmCalls"],
                    "answer": _clip(single["answer"]),
                },
                "team": {
                    **team["summary"],
                    "llmCalls": team["metrics"]["llmCalls"],
                    "workerElapsedMs": team["metrics"]["workerElapsedMs"],
                    "answer": _clip(team["answer"]),
                },
                "verdict": payload["verdict"],
            }
        )
    summary = summarize(rows)
    document = {
        "label": "scripted",
        "createdAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": {
            "model": "extractive-stand-in",
            "llmDelaySeconds": delay_s,
            "note": (
                "Both sides answer with the text they retrieved, on the same four questions. "
                f"Each model call waits {delay_s:.2f}s so a parallel team can overlap that wait. "
                "Tokens are chars/4 of the real prompts, including every re-sent task. "
                "Cost is llm_calls + tokens/4000, the same formula as the single agent. "
                "This is not an Ollama wall-clock. Re-run with --live for generation quality and true waiting time."
            ),
        },
        "cases": rows,
        "summary": summary,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2), encoding="utf-8")
    memory_path = out.parent / "_scripted_memory.json"
    if memory_path.is_file():
        memory_path.unlink()
    return document


async def run_live(out: Path) -> dict[str, Any]:
    runner = TeamRunner(ClaimAgentRunner(memory=AgentMemoryStore(out.parent / "_live_memory.json")))
    rows: list[dict[str, Any]] = []
    for case in load_cases():
        payload = await runner.race(
            case["task"],
            case_id=case["id"],
            must_contain_any=case["must_contain_any"],
            max_steps=8,
            max_llm_calls=10,
            max_seconds=120,
        )
        single = payload["single"]
        team = payload["team"]
        rows.append(
            {
                "id": case["id"],
                "label": case["label"],
                "task": case["task"],
                "why": case["why"],
                "parallel": bool(team["plan"]["parallel"]),
                "workers": list(team["plan"]["workers"]),
                "planReason": team["plan"]["reason"],
                "contextResendTokens": team["metrics"]["contextResendTokens"],
                "single": {
                    **single["summary"],
                    "llmCalls": single["metrics"]["llmCalls"],
                    "answer": _clip(single["answer"]),
                },
                "team": {
                    **team["summary"],
                    "llmCalls": team["metrics"]["llmCalls"],
                    "workerElapsedMs": team["metrics"]["workerElapsedMs"],
                    "answer": _clip(team["answer"]),
                },
                "verdict": payload["verdict"],
            }
        )
    summary = summarize(rows)
    document = {
        "label": "live",
        "createdAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": {
            "model": "ollama",
            "llmDelaySeconds": 0,
            "note": "Live single agent and live team on the same four questions. Speed is wall-clock.",
        },
        "cases": rows,
        "summary": summary,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2), encoding="utf-8")
    memory_path = out.parent / "_live_memory.json"
    if memory_path.is_file():
        memory_path.unlink()
    return document


def _print_summary(document: dict[str, Any]) -> None:
    summary = document["summary"]
    single = summary["single"]
    team = summary["team"]
    verdict = summary["verdict"]
    print(f"label: {document['label']}")
    print(
        "quality  single {q:.0%}   team {t:.0%}".format(
            q=single["quality"], t=team["quality"]
        )
    )
    print(
        "speed    single {s:.0f} ms   team {t:.0f} ms".format(
            s=single["elapsedMs"], t=team["elapsedMs"]
        )
    )
    print(
        "tokens   single {s:.0f}   team {t:.0f}".format(
            s=single["estimatedTokens"], t=team["estimatedTokens"]
        )
    )
    print(
        "cost     single {s:.3f}   team {t:.3f}".format(
            s=single["costUnits"], t=team["costUnits"]
        )
    )
    print(f"ship: {verdict['ship']} ({verdict['decidedBy']})")
    print(verdict["rationale"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Race the single agent against the specialist team")
    parser.add_argument("--scripted", action="store_true", help="Extractive stand-in, no Ollama")
    parser.add_argument("--live", action="store_true", help="Real Ollama and the document index")
    parser.add_argument("--delay", type=float, default=0.05, help="Seconds each scripted model call waits")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.live == args.scripted:
        parser.error("choose one of --scripted or --live")
    if args.live:
        out = args.out or (HERE / "results" / "live.json")
        document = asyncio.run(run_live(out))
    else:
        out = args.out or (HERE / "results" / "race.json")
        document = asyncio.run(run_scripted(out, delay_s=args.delay))
    print(f"wrote {out}")
    _print_summary(document)


if __name__ == "__main__":
    main()
