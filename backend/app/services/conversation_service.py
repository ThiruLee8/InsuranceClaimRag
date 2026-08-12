from __future__ import annotations

import uuid
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.models import Conversation, Message, MessageRole, RAGSource
from app.db.repositories import ConversationRepository, DocumentRepository
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationCreate,
    ConversationOut,
    ConversationUpdate,
    MessageOut,
    RAGSourceOut,
)
from app.services.rag_service import RAGService

logger = get_logger(__name__)


class ConversationService:
    def __init__(self, db: Session, rag_service: RAGService | None = None) -> None:
        self.db = db
        self.repo = ConversationRepository(db)
        self.document_repo = DocumentRepository(db)
        self._rag_service = rag_service
        self.settings = get_settings()

    @property
    def rag_service(self) -> RAGService:
        if self._rag_service is None:
            self._rag_service = RAGService()
        return self._rag_service

    def _to_conversation_out(self, conversation: Conversation) -> ConversationOut:
        out = ConversationOut.model_validate(conversation)
        out.messageCount = self.repo.message_count(conversation.Id)
        return out

    def _to_message_out(self, message: Message) -> MessageOut:
        out = MessageOut.model_validate(message)
        sources: list[RAGSourceOut] = []
        for source in message.Sources or []:
            file_name = None
            if source.DocumentId:
                doc = self.document_repo.get(source.DocumentId)
                file_name = doc.OriginalFileName if doc else None
            sources.append(
                RAGSourceOut(
                    documentId=source.DocumentId,
                    chunkId=source.ChunkId,
                    fileName=file_name,
                    pageNumber=source.PageNumber,
                    relevanceScore=source.RelevanceScore,
                )
            )
        out.sources = sources
        return out

    def create(self, payload: ConversationCreate) -> ConversationOut:
        conversation = Conversation(
            Id=uuid.uuid4(),
            Title=payload.title or "New Conversation",
            CreatedBy=payload.createdBy,
        )
        conversation = self.repo.create(conversation)
        logger.info("conversation_created", conversation_id=str(conversation.Id))
        return self._to_conversation_out(conversation)

    def list(self) -> list[ConversationOut]:
        return [self._to_conversation_out(c) for c in self.repo.list()]

    def get(self, conversation_id: UUID) -> ConversationOut:
        conversation = self.repo.get(conversation_id)
        if not conversation:
            raise LookupError("Conversation not found")
        return self._to_conversation_out(conversation)

    def update(self, conversation_id: UUID, payload: ConversationUpdate) -> ConversationOut:
        conversation = self.repo.get(conversation_id)
        if not conversation:
            raise LookupError("Conversation not found")
        conversation.Title = payload.title
        conversation.UpdatedAt = datetime.now(timezone.utc)
        conversation = self.repo.update(conversation)
        return self._to_conversation_out(conversation)

    def delete(self, conversation_id: UUID) -> None:
        conversation = self.repo.get(conversation_id)
        if not conversation:
            raise LookupError("Conversation not found")
        self.repo.soft_delete(conversation)

    def get_messages(self, conversation_id: UUID) -> list[MessageOut]:
        conversation = self.repo.get(conversation_id)
        if not conversation:
            raise LookupError("Conversation not found")
        return [self._to_message_out(m) for m in self.repo.get_messages(conversation_id)]

    def clear_messages(self, conversation_id: UUID) -> None:
        conversation = self.repo.get(conversation_id)
        if not conversation:
            raise LookupError("Conversation not found")
        self.repo.clear_messages(conversation_id)
        conversation.UpdatedAt = datetime.now(timezone.utc)
        self.repo.update(conversation)

    async def chat(self, payload: ChatRequest) -> ChatResponse:
        if payload.conversationId:
            conversation = self.repo.get(payload.conversationId)
            if not conversation:
                raise LookupError("Conversation not found")
        else:
            title = payload.question[:80] + ("..." if len(payload.question) > 80 else "")
            conversation = self.repo.create(
                Conversation(Id=uuid.uuid4(), Title=title or "New Conversation")
            )

        if payload.regenerateMessageId:
            target = self.repo.get_message(payload.regenerateMessageId)
            if not target or target.ConversationId != conversation.Id:
                raise LookupError("Message not found for regeneration")
            self.repo.delete_messages_after(conversation.Id, target)
            user_message = None
        else:
            user_message = Message(
                Id=uuid.uuid4(),
                ConversationId=conversation.Id,
                Role=MessageRole.user,
                Content=payload.question,
            )
            self.repo.add_message(user_message)

        result = await self.rag_service.answer(
            payload.question,
            agent_id=payload.agentId,
            model=payload.model,
        )

        assistant_message = Message(
            Id=uuid.uuid4(),
            ConversationId=conversation.Id,
            Role=MessageRole.assistant,
            Content=result.answer,
            ModelName=result.model,
            TokenCount=None,
        )
        self.repo.add_message(assistant_message)

        source_rows: list[RAGSource] = []
        source_out: list[RAGSourceOut] = []
        for hit in result.sources:
            doc_id = UUID(hit.document_id) if hit.document_id else None
            chunk_id = UUID(hit.chunk_id) if hit.chunk_id else None
            source_rows.append(
                RAGSource(
                    Id=uuid.uuid4(),
                    MessageId=assistant_message.Id,
                    DocumentId=doc_id,
                    ChunkId=chunk_id,
                    PageNumber=hit.page_number,
                    RelevanceScore=hit.score,
                )
            )
            source_out.append(
                RAGSourceOut(
                    documentId=doc_id,
                    chunkId=chunk_id,
                    fileName=hit.file_name,
                    pageNumber=hit.page_number,
                    relevanceScore=hit.score,
                )
            )
        if source_rows:
            self.repo.add_sources(source_rows)

        if conversation.Title == "New Conversation":
            conversation.Title = payload.question[:80] + ("..." if len(payload.question) > 80 else "")
        conversation.UpdatedAt = datetime.now(timezone.utc)
        self.repo.update(conversation)

        logger.info(
            "chat_completed",
            conversation_id=str(conversation.Id),
            message_id=str(assistant_message.Id),
            sources=len(source_out),
        )
        return ChatResponse(
            conversationId=conversation.Id,
            messageId=assistant_message.Id,
            answer=result.answer,
            sources=source_out,
            agentId=result.agent_id,
            model=result.model,
        )
