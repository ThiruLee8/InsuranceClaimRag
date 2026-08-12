import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

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


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    service: ConversationService = Depends(get_conversation_service),
):
    """Server-Sent Events stream: meta → status/sources → token* → done|error."""

    async def event_generator():
        try:
            async for event in service.chat_stream(payload):
                yield f"data: {json.dumps(event, default=str)}\n\n"
        except LookupError as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc), 'errorCode': 'CHAT_NOT_FOUND'})}\n\n"
        except Exception as exc:  # noqa: BLE001
            yield f"data: {json.dumps({'type': 'error', 'message': f'Chat failed: {exc}', 'errorCode': 'CHAT_ERROR'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
