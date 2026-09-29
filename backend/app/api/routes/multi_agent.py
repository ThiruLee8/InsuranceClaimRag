"""HTTP surface for the specialist team and the single-vs-team race."""

from typing import Any, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.dependencies import get_team_runner
from app.core.exceptions import raise_http
from app.schemas import ApiResponse
from app.services.a2a import TASK_STATES
from app.services.multi_agent import (
    FRAMEWORKS,
    MCP_VS_A2A,
    TeamRunner,
    _score_module,
    list_cards,
    load_cases,
    load_suite_result,
)

router = APIRouter(prefix="/multi-agent", tags=["multi-agent"])


class MultiAgentTaskIn(BaseModel):
    specialistId: str
    task: str = Field(min_length=1, max_length=4000)
    model: Optional[str] = None


class MultiAgentRaceIn(BaseModel):
    task: str = Field(min_length=1, max_length=4000)
    caseId: Optional[str] = None
    model: Optional[str] = None
    maxSteps: int = 8
    maxLlmCalls: int = 10
    maxSeconds: float = 120


_FALLBACK_RULE = (
    "Quality first (0.05 gap on the same assertions). "
    "If quality ties, lower cost wins (5% gap). "
    "If quality and cost tie, lower latency wins (10% gap). "
    "If all three tie, keep the single agent."
)


@router.get("/meta", response_model=ApiResponse[dict])
async def multi_agent_meta():
    try:
        decision_rule = _score_module().DECISION_RULE
    except Exception:  # noqa: BLE001
        decision_rule = _FALLBACK_RULE
    return ApiResponse(
        data={
            "cards": list_cards(),
            "lifecycle": list(TASK_STATES),
            "mcpVsA2a": MCP_VS_A2A,
            "frameworks": FRAMEWORKS,
            "decisionRule": decision_rule,
            "cases": load_cases(),
            "suite": load_suite_result(),
        }
    )


@router.get("/cards", response_model=ApiResponse[dict])
async def multi_agent_cards(runner: TeamRunner = Depends(get_team_runner)):
    return ApiResponse(data={"cards": runner.cards(), "lifecycle": list(TASK_STATES)})


@router.post("/tasks", response_model=ApiResponse[dict])
async def multi_agent_task(
    payload: MultiAgentTaskIn,
    runner: TeamRunner = Depends(get_team_runner),
):
    """A2A-style hand-off: one specialist, one task, one artifact."""
    try:
        task = await runner.run_specialist(
            payload.specialistId.strip(),
            payload.task.strip(),
            model=payload.model,
        )
    except KeyError as exc:
        raise_http(str(exc), "VALIDATION_ERROR", 422)
    except Exception as exc:  # noqa: BLE001
        raise_http(f"Specialist task failed: {exc}", "AGENT_ERROR", 500)
    return ApiResponse(data=task)


@router.post("/race", response_model=ApiResponse[dict])
async def multi_agent_race(
    payload: MultiAgentRaceIn,
    runner: TeamRunner = Depends(get_team_runner),
):
    """Same task, single agent then the team. Quality, speed, tokens, cost."""
    try:
        result: dict[str, Any] = await runner.race(
            payload.task.strip(),
            case_id=payload.caseId,
            model=payload.model,
            max_steps=payload.maxSteps,
            max_llm_calls=payload.maxLlmCalls,
            max_seconds=payload.maxSeconds,
        )
    except Exception as exc:  # noqa: BLE001
        raise_http(f"Race failed: {exc}", "AGENT_ERROR", 500)
    return ApiResponse(data=result)
