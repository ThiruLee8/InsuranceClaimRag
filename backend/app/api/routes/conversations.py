from uuid import UUID

from fastapi import APIRouter, Depends

from app.api.dependencies import get_conversation_service
from app.core.exceptions import raise_http
from app.schemas import (
    ApiResponse,
    ConversationCreate,
    ConversationOut,
    ConversationUpdate,
    MessageOut,
)
from app.services.conversation_service import ConversationService

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("", response_model=ApiResponse[list[ConversationOut]])
def list_conversations(service: ConversationService = Depends(get_conversation_service)):
    return ApiResponse(data=service.list())


@router.post("", response_model=ApiResponse[ConversationOut], status_code=201)
def create_conversation(
    payload: ConversationCreate,
    service: ConversationService = Depends(get_conversation_service),
):
    return ApiResponse(data=service.create(payload), message="Conversation created")


@router.get("/{conversation_id}", response_model=ApiResponse[ConversationOut])
def get_conversation(
    conversation_id: UUID,
    service: ConversationService = Depends(get_conversation_service),
):
    try:
        return ApiResponse(data=service.get(conversation_id))
    except LookupError:
        raise_http("Conversation not found", "CONVERSATION_NOT_FOUND", 404)


@router.put("/{conversation_id}", response_model=ApiResponse[ConversationOut])
def update_conversation(
    conversation_id: UUID,
    payload: ConversationUpdate,
    service: ConversationService = Depends(get_conversation_service),
):
    try:
        return ApiResponse(data=service.update(conversation_id, payload))
    except LookupError:
        raise_http("Conversation not found", "CONVERSATION_NOT_FOUND", 404)


@router.delete("/{conversation_id}", response_model=ApiResponse[None])
def delete_conversation(
    conversation_id: UUID,
    service: ConversationService = Depends(get_conversation_service),
):
    try:
        service.delete(conversation_id)
        return ApiResponse(message="Conversation deleted")
    except LookupError:
        raise_http("Conversation not found", "CONVERSATION_NOT_FOUND", 404)


@router.get("/{conversation_id}/messages", response_model=ApiResponse[list[MessageOut]])
def get_messages(
    conversation_id: UUID,
    service: ConversationService = Depends(get_conversation_service),
):
    try:
        return ApiResponse(data=service.get_messages(conversation_id))
    except LookupError:
        raise_http("Conversation not found", "CONVERSATION_NOT_FOUND", 404)


@router.delete("/{conversation_id}/messages", response_model=ApiResponse[None])
def clear_messages(
    conversation_id: UUID,
    service: ConversationService = Depends(get_conversation_service),
):
    try:
        service.clear_messages(conversation_id)
        return ApiResponse(message="Conversation cleared")
    except LookupError:
        raise_http("Conversation not found", "CONVERSATION_NOT_FOUND", 404)
