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
    RetrieveRequest,
    RetrieveResponse,
)
from app.services.rag_service import RAGService
from app.services.trace_service import get_trace_service

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

    def _hit_to_source_out(self, hit, *, debug: bool) -> RAGSourceOut:
        doc_id = UUID(hit.document_id) if hit.document_id else None
        chunk_id = UUID(hit.chunk_id) if hit.chunk_id else None
        return RAGSourceOut(
            documentId=doc_id,
            chunkId=chunk_id,
            fileName=hit.file_name,
            pageNumber=hit.page_number,
            relevanceScore=hit.score,
            content=hit.content if debug else None,
        )

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

    def _reuse_last_user_message(self, conversation_id: UUID) -> None:
        messages = self.repo.get_messages(conversation_id)
        last_user_idx = next(
            (i for i in range(len(messages) - 1, -1, -1) if messages[i].Role == MessageRole.user),
            None,
        )
        if last_user_idx is None:
            raise LookupError("No prior user message to retry")
        trailing = messages[last_user_idx + 1 :]
        for msg in trailing:
            self.db.delete(msg)
        if trailing:
            self.db.commit()

    async def retrieve(self, payload: RetrieveRequest) -> RetrieveResponse:
        result = await self.rag_service.retrieve(
            payload.question,
            top_k=payload.topK,
            similarity_threshold=payload.similarityThreshold,
            search_mode=payload.searchMode,
            rewrite=payload.rewrite,
            rerank=payload.rerank,
        )
        sources = [
            self._hit_to_source_out(hit, debug=payload.debug) for hit in result.hits
        ]
        return RetrieveResponse(
            originalQuestion=result.original_question,
            searchQuery=result.search_query,
            searchMode=result.search_mode,
            sources=sources,
        )

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
        elif payload.reuseLastUserMessage:
            self._reuse_last_user_message(conversation.Id)
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
            debug=payload.debug,
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
            source_out.append(self._hit_to_source_out(hit, debug=payload.debug))
        if source_rows:
            self.repo.add_sources(source_rows)

        if conversation.Title == "New Conversation":
            conversation.Title = payload.question[:80] + ("..." if len(payload.question) > 80 else "")
        conversation.UpdatedAt = datetime.now(timezone.utc)
        self.repo.update(conversation)

        # Complete trace for error analysis (always includes chunk text).
        try:
            get_trace_service().record(
                question=payload.question,
                answer=result.answer,
                sources=[
                    {
                        "documentId": hit.document_id,
                        "chunkId": hit.chunk_id,
                        "fileName": hit.file_name,
                        "pageNumber": hit.page_number,
                        "score": hit.score,
                        "content": hit.content,
                    }
                    for hit in result.sources
                ],
                conversation_id=str(conversation.Id),
                message_id=str(assistant_message.Id),
                agent_id=result.agent_id,
                model=result.model,
                original_question=result.original_question,
                search_query=result.search_query,
                search_mode=result.search_mode,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("trace_record_failed", error=str(exc))

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
            originalQuestion=result.original_question if payload.debug else None,
            searchQuery=result.search_query if payload.debug else None,
            searchMode=result.search_mode if payload.debug else None,
        )

    async def chat_stream(self, payload: ChatRequest):
        """Async generator of SSE-ready dict events for streaming chat."""
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
        elif payload.reuseLastUserMessage:
            self._reuse_last_user_message(conversation.Id)
        else:
            user_message = Message(
                Id=uuid.uuid4(),
                ConversationId=conversation.Id,
                Role=MessageRole.user,
                Content=payload.question,
            )
            self.repo.add_message(user_message)

        yield {
            "type": "meta",
            "conversationId": str(conversation.Id),
        }

        async for event in self.rag_service.answer_stream(
            payload.question,
            agent_id=payload.agentId,
            model=payload.model,
            debug=payload.debug,
        ):
            if event.get("type") != "done":
                yield event
                continue

            answer = str(event.get("answer") or "")
            agent_id = str(event.get("agentId") or "")
            model_name = str(event.get("model") or self.settings.ollama_model)
            hits = event.get("_hits") or []
            source_payload = event.get("sources") or []
            original_question = event.get("originalQuestion")
            search_query = event.get("searchQuery")
            search_mode = event.get("searchMode")

            assistant_message = Message(
                Id=uuid.uuid4(),
                ConversationId=conversation.Id,
                Role=MessageRole.assistant,
                Content=answer,
                ModelName=model_name,
                TokenCount=None,
            )
            self.repo.add_message(assistant_message)

            source_rows: list[RAGSource] = []
            source_out: list[RAGSourceOut] = []
            for hit in hits:
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
            for item in source_payload:
                source_out.append(
                    RAGSourceOut(
                        documentId=UUID(item["documentId"]) if item.get("documentId") else None,
                        chunkId=UUID(item["chunkId"]) if item.get("chunkId") else None,
                        fileName=item.get("fileName"),
                        pageNumber=item.get("pageNumber"),
                        relevanceScore=item.get("relevanceScore"),
                        content=item.get("content") if payload.debug else None,
                    )
                )
            if source_rows:
                self.repo.add_sources(source_rows)

            if conversation.Title == "New Conversation":
                conversation.Title = payload.question[:80] + (
                    "..." if len(payload.question) > 80 else ""
                )
            conversation.UpdatedAt = datetime.now(timezone.utc)
            self.repo.update(conversation)

            try:
                get_trace_service().record(
                    question=payload.question,
                    answer=answer,
                    sources=[
                        {
                            "documentId": hit.document_id,
                            "chunkId": hit.chunk_id,
                            "fileName": hit.file_name,
                            "pageNumber": hit.page_number,
                            "score": hit.score,
                            "content": hit.content,
                        }
                        for hit in hits
                    ],
                    conversation_id=str(conversation.Id),
                    message_id=str(assistant_message.Id),
                    agent_id=agent_id,
                    model=model_name,
                    original_question=str(original_question or payload.question),
                    search_query=str(search_query or payload.question),
                    search_mode=str(search_mode) if search_mode else None,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("trace_record_failed", error=str(exc))

            logger.info(
                "chat_stream_completed",
                conversation_id=str(conversation.Id),
                message_id=str(assistant_message.Id),
                sources=len(source_out),
            )
            done_event: dict = {
                "type": "done",
                "conversationId": str(conversation.Id),
                "messageId": str(assistant_message.Id),
                "answer": answer,
                "sources": [s.model_dump(mode="json") for s in source_out],
                "agentId": agent_id,
                "model": model_name,
            }
            if payload.debug:
                done_event["originalQuestion"] = original_question
                done_event["searchQuery"] = search_query
                done_event["searchMode"] = search_mode
            yield done_event
