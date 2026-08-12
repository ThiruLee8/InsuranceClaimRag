"""
Additional unit tests for conversation/RAG wiring with mocks.
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.schemas import ChatRequest
from app.services.conversation_service import ConversationService
from app.services.rag_service import RAGResult
from app.services.vector_gateway_client import VectorSearchHit


@pytest.mark.asyncio
async def test_chat_persists_messages_and_sources():
    db = MagicMock()
    rag = MagicMock()
    service = ConversationService(db, rag_service=rag)

    conversation_id = uuid4()
    conversation = MagicMock()
    conversation.Id = conversation_id
    conversation.Title = "New Conversation"

    service.repo.get = MagicMock(return_value=conversation)
    service.repo.add_message = MagicMock(side_effect=lambda m: m)
    service.repo.add_sources = MagicMock()
    service.repo.update = MagicMock(return_value=conversation)

    hit = VectorSearchHit(
        chunk_id=str(uuid4()),
        document_id=str(uuid4()),
        file_name="claim-investigation-report.pdf",
        page_number=4,
        chunk_index=0,
        content="Cause of loss was a burst pipe.",
        score=0.91,
    )

    rag.answer = AsyncMock(
        return_value=RAGResult(
            answer="Burst pipe caused the loss.",
            sources=[hit],
            agent_id="claims_assistant",
            model="llama3.2",
        )
    )

    result = await service.chat(
        ChatRequest(conversationId=conversation_id, question="What caused the loss?")
    )

    assert result.answer.startswith("Burst pipe")
    assert len(result.sources) == 1
    assert result.sources[0].pageNumber == 4
    assert service.repo.add_message.call_count == 2
    assert service.repo.add_sources.call_count == 1
