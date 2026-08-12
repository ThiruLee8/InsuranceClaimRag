from fastapi import APIRouter

from app.core.config import get_settings
from app.schemas import AgentOut, AgentsResponse, ApiResponse
from app.services.agents import DEFAULT_AGENT_ID, list_agents
from app.services.llm_service import OllamaLLMService

router = APIRouter(prefix="/agents", tags=["agents"])


@router.get("", response_model=ApiResponse[AgentsResponse])
async def get_agents():
    settings = get_settings()
    models = await OllamaLLMService().list_models()
    agents = [
        AgentOut(id=agent.id, name=agent.name, description=agent.description)
        for agent in list_agents()
    ]
    return ApiResponse(
        data=AgentsResponse(
            agents=agents,
            models=models,
            defaultAgentId=DEFAULT_AGENT_ID,
            defaultModel=settings.ollama_model,
        )
    )
