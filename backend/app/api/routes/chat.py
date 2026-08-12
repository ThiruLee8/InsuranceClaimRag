from fastapi import APIRouter, Depends

from app.api.dependencies import get_conversation_service
from app.core.exceptions import raise_http
from app.schemas import ApiResponse, ChatRequest, ChatResponse
from app.services.conversation_service import ConversationService

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ApiResponse[ChatResponse])
async def chat(
    payload: ChatRequest,
    service: ConversationService = Depends(get_conversation_service),
):
    try:
        result = await service.chat(payload)
        return ApiResponse(data=result)
    except LookupError as exc:
        raise_http(str(exc), "CHAT_NOT_FOUND", 404)
    except Exception as exc:  # noqa: BLE001
        raise_http(f"Chat failed: {exc}", "CHAT_ERROR", 500)
