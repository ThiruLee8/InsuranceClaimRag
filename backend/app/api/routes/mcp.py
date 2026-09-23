from fastapi import APIRouter

from app.core.config import get_settings
from app.mcp.gateway import WHERE_AI_DOES_NOT_RUN, WHERE_AI_RUNS, get_shared_gateway
from app.schemas import AgentToolOut, ApiResponse, McpStatusOut

router = APIRouter(prefix="/mcp", tags=["mcp"])


def _status_from_gateway() -> McpStatusOut:
    settings = get_settings()
    gw = get_shared_gateway()
    payload = gw.status_dict()
    return McpStatusOut(
        enabled=settings.mcp_enabled,
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


@router.get("/status", response_model=ApiResponse[McpStatusOut])
async def mcp_status():
    settings = get_settings()
    if not settings.mcp_enabled:
        return ApiResponse(
            data=McpStatusOut(
                enabled=False,
                transport="off",
                serverName="claims-docs",
                connected=False,
                whereAiRuns=WHERE_AI_RUNS,
                whereAiDoesNotRun=WHERE_AI_DOES_NOT_RUN,
                endpoint="/mcp",
                authRequired=bool(settings.mcp_auth_token),
            )
        )
    gw = get_shared_gateway()
    await gw.ensure_connected()
    return ApiResponse(data=_status_from_gateway())


@router.post("/rediscover", response_model=ApiResponse[McpStatusOut])
async def mcp_rediscover():
    gw = get_shared_gateway()
    await gw.ensure_connected()
    await gw.discover()
    return ApiResponse(data=_status_from_gateway())
