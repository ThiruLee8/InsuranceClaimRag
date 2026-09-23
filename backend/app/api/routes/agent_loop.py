import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_claim_agent_runner
from app.core.config import get_settings
from app.core.exceptions import raise_http
from app.mcp.gateway import get_shared_gateway
from app.schemas import (
    AgentLoopMetaOut,
    AgentLoopRunRequest,
    AgentMemoryEntryOut,
    AgentMemoryListOut,
    AgentSampleTaskOut,
    AgentToolOut,
    ApiResponse,
    McpStatusOut,
)
from app.services.agent_guardrails import OWASP_LLM_TOP_10, RESIDUAL_RISKS
from app.services.agent_loop import SAMPLE_TASKS, ClaimAgentRunner
from app.services.agent_memory import AgentMemoryStore
from app.services.agent_tools import TOOLS

router = APIRouter(prefix="/agent-loop", tags=["agent-loop"])


@router.get("/meta", response_model=ApiResponse[AgentLoopMetaOut])
async def agent_loop_meta():
    settings = get_settings()
    mcp_out: McpStatusOut | None = None
    tools = [
        AgentToolOut(name=t.name, description=t.description, parameters=t.parameters, source="local")
        for t in TOOLS
    ]
    if settings.mcp_enabled:
        try:
            gw = get_shared_gateway()
            await gw.ensure_connected()
            payload = gw.status_dict()
            mcp_out = McpStatusOut(
                enabled=True,
                transport=payload["transport"],
                serverName=payload["serverName"],
                connected=payload["connected"],
                whereAiRuns=payload["whereAiRuns"],
                whereAiDoesNotRun=payload["whereAiDoesNotRun"],
                endpoint=payload["endpoint"],
                stdio=payload["stdio"],
                authRequired=payload["authRequired"],
                tools=[AgentToolOut(**t) for t in payload["tools"]],
                resources=payload["resources"],
                prompts=payload["prompts"],
                handshake=payload["handshake"],
                lastError=payload["lastError"],
                roles=payload["roles"],
            )
            if gw.trusted_tools:
                tools = [
                    AgentToolOut(
                        name=t.name,
                        description=t.description,
                        parameters=t.parameters,
                        server=t.server,
                        trusted=t.trusted,
                        trustReason=t.trust_reason,
                        source="mcp",
                    )
                    for t in gw.trusted_tools
                ]
                tools.append(
                    AgentToolOut(
                        name="finish",
                        description="End the loop on the host and answer the user.",
                        parameters='{"answer": "final answer"}',
                        server="host",
                        trusted=True,
                        trustReason="loop control stays on the host",
                        source="host",
                    )
                )
        except Exception as exc:  # noqa: BLE001
            mcp_out = McpStatusOut(
                enabled=True,
                transport="memory",
                serverName="claims-docs",
                connected=False,
                whereAiRuns="On the host — this ClaimIntel app, which calls Ollama.",
                whereAiDoesNotRun="The MCP server only offers tools. It does not run the model.",
                endpoint="/mcp",
                authRequired=bool(settings.mcp_auth_token),
                lastError=str(exc),
            )
    return ApiResponse(
        data=AgentLoopMetaOut(
            tools=tools,
            sampleTasks=[AgentSampleTaskOut(**item) for item in SAMPLE_TASKS],
            defaultMaxSteps=settings.agent_max_steps,
            defaultMaxLlmCalls=settings.agent_max_llm_calls,
            defaultMaxSeconds=settings.agent_max_seconds,
            defaultModel=settings.ollama_model,
            guardrailsEnabled=settings.agent_guardrails_enabled,
            residualRisks=list(RESIDUAL_RISKS),
            owasp=[{"id": x["id"], "name": x["name"], "how": x["how"]} for x in OWASP_LLM_TOP_10],
            mcp=mcp_out,
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
