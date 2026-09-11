import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_claim_agent_runner
from app.core.config import get_settings
from app.core.exceptions import raise_http
from app.schemas import (
    AgentLoopMetaOut,
    AgentLoopRunRequest,
    AgentMemoryEntryOut,
    AgentMemoryListOut,
    AgentSampleTaskOut,
    AgentToolOut,
    ApiResponse,
)
from app.services.agent_guardrails import OWASP_LLM_TOP_10, RESIDUAL_RISKS
from app.services.agent_loop import SAMPLE_TASKS, ClaimAgentRunner
from app.services.agent_memory import AgentMemoryStore
from app.services.agent_tools import TOOLS

router = APIRouter(prefix="/agent-loop", tags=["agent-loop"])


@router.get("/meta", response_model=ApiResponse[AgentLoopMetaOut])
async def agent_loop_meta():
    settings = get_settings()
    return ApiResponse(
        data=AgentLoopMetaOut(
            tools=[
                AgentToolOut(name=t.name, description=t.description, parameters=t.parameters)
                for t in TOOLS
            ],
            sampleTasks=[AgentSampleTaskOut(**item) for item in SAMPLE_TASKS],
            defaultMaxSteps=settings.agent_max_steps,
            defaultMaxLlmCalls=settings.agent_max_llm_calls,
            defaultMaxSeconds=settings.agent_max_seconds,
            defaultModel=settings.ollama_model,
            guardrailsEnabled=settings.agent_guardrails_enabled,
            residualRisks=list(RESIDUAL_RISKS),
            owasp=[{"id": x["id"], "name": x["name"], "how": x["how"]} for x in OWASP_LLM_TOP_10],
        )
    )


@router.post("/run")
async def agent_loop_run(
    payload: AgentLoopRunRequest,
    runner: ClaimAgentRunner = Depends(get_claim_agent_runner),
):
    mode = (payload.mode or "agent").strip().lower()
    if mode not in {"agent", "workflow", "race"}:
        raise_http("mode must be agent, workflow, or race", "VALIDATION_ERROR", 422)
    try:
        if mode == "workflow":
            result = await runner.run_workflow(
                payload.task,
                session_id=payload.sessionId,
                model=payload.model,
                persist_memory=payload.persistMemory,
            )
            return ApiResponse(data=result.to_dict())
        if mode == "race":
            result = await runner.run_race(
                payload.task,
                session_id=payload.sessionId,
                model=payload.model,
                max_steps=payload.maxSteps,
                max_llm_calls=payload.maxLlmCalls,
                max_seconds=payload.maxSeconds,
                persist_memory=payload.persistMemory,
            )
            return ApiResponse(data=result)
        result = await runner.run_agent(
            payload.task,
            session_id=payload.sessionId,
            model=payload.model,
            max_steps=payload.maxSteps,
            max_llm_calls=payload.maxLlmCalls,
            max_seconds=payload.maxSeconds,
            persist_memory=payload.persistMemory,
        )
        return ApiResponse(data=result.to_dict())
    except Exception as exc:  # noqa: BLE001
        raise_http(f"Agent run failed: {exc}", "AGENT_ERROR", 500)


@router.post("/run/stream")
async def agent_loop_run_stream(
    payload: AgentLoopRunRequest,
    runner: ClaimAgentRunner = Depends(get_claim_agent_runner),
):
    mode = (payload.mode or "agent").strip().lower()
    if mode not in {"agent", "workflow", "race"}:
        raise_http("mode must be agent, workflow, or race", "VALIDATION_ERROR", 422)

    async def event_generator():
        try:
            async for event in runner.stream(
                mode=mode,
                task=payload.task,
                session_id=payload.sessionId,
                model=payload.model,
                max_steps=payload.maxSteps,
                max_llm_calls=payload.maxLlmCalls,
                max_seconds=payload.maxSeconds,
                persist_memory=payload.persistMemory,
            ):
                yield f"data: {json.dumps(event, default=str)}\n\n"
        except Exception as exc:  # noqa: BLE001
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/memory", response_model=ApiResponse[AgentMemoryListOut])
def list_memory(sessionId: str | None = None):
    store = AgentMemoryStore()
    items = store.list_entries(session_id=sessionId)
    return ApiResponse(
        data=AgentMemoryListOut(
            sessionId=sessionId,
            items=[
                AgentMemoryEntryOut(
                    id=e.id,
                    sessionId=e.session_id,
                    createdAt=e.created_at,
                    task=e.task,
                    summary=e.summary,
                    facts=e.facts,
                )
                for e in items
            ],
        )
    )


@router.delete("/memory", response_model=ApiResponse[dict])
def clear_memory(sessionId: str | None = None):
    removed = AgentMemoryStore().clear(session_id=sessionId)
    return ApiResponse(data={"removed": removed, "sessionId": sessionId})
