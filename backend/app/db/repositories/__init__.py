from __future__ import annotations

from uuid import UUID

from sqlalchemy import func, or_, select, update
from sqlalchemy.orm import Session, selectinload

from app.db.models import Conversation, Document, DocumentChunk, DocumentStatus, Message, MessageRole, RAGSource


class DocumentRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, document: Document) -> Document:
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)
        return document

    def get(self, document_id: UUID) -> Document | None:
        return self.db.scalar(
            select(Document).where(Document.Id == document_id, Document.IsDeleted == False)
        )

    def get_by_hash(self, file_hash: str) -> Document | None:
        return self.db.scalar(
            select(Document).where(Document.FileHash == file_hash, Document.IsDeleted == False)
        )

    def get_by_blob_name(self, blob_name: str) -> Document | None:
        return self.db.scalar(
            select(Document).where(Document.BlobName == blob_name, Document.IsDeleted == False)
        )

    def list(
        self,
        *,
        search: str | None = None,
        status: DocumentStatus | None = None,
        content_type: str | None = None,
        skip: int = 0,
        limit: int = 50,
    ) -> tuple[list[Document], int]:
        query = select(Document).where(Document.IsDeleted == False)
        count_query = select(func.count()).select_from(Document).where(Document.IsDeleted == False)

        if search:
            like = f"%{search.lower()}%"
            filt = or_(
                func.lower(Document.OriginalFileName).like(like),
                func.lower(Document.FileName).like(like),
            )
            query = query.where(filt)
            count_query = count_query.where(filt)
        if status:
            query = query.where(Document.Status == status)
            count_query = count_query.where(Document.Status == status)
        if content_type:
            query = query.where(Document.ContentType == content_type)
            count_query = count_query.where(Document.ContentType == content_type)

        total = self.db.scalar(count_query) or 0
        items = list(
            self.db.scalars(
                query.order_by(Document.UploadedAt.desc()).offset(skip).limit(limit)
            ).all()
        )
        return items, total

    def list_active_ids(self) -> list[UUID]:
        return list(
            self.db.scalars(
                select(Document.Id)
                .where(Document.IsDeleted == False)
                .order_by(Document.UploadedAt.asc())
            ).all()
        )

    def list_active_for_process_all(self, *, force: bool = False) -> list[Document]:
        query = select(Document).where(Document.IsDeleted == False)
        if not force:
            query = query.where(
                Document.Status.notin_(
                    [
                        DocumentStatus.Completed,
                        DocumentStatus.Processed,
                        DocumentStatus.Processing,
                        DocumentStatus.ExtractingText,
                        DocumentStatus.Chunking,
                        DocumentStatus.GeneratingEmbeddings,
                        DocumentStatus.Indexing,
                        DocumentStatus.Queued,
                    ]
                )
            )
        return list(self.db.scalars(query.order_by(Document.UploadedAt.asc())).all())

    def update(self, document: Document) -> Document:
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)
        return document

    def soft_delete(self, document: Document) -> Document:
        document.IsDeleted = True
        document.Status = DocumentStatus.Deleted
        return self.update(document)

    def chunk_count(self, document_id: UUID) -> int:
        return (
            self.db.scalar(
                select(func.count()).select_from(DocumentChunk).where(DocumentChunk.DocumentId == document_id)
            )
            or 0
        )

    def _detach_chunk_references(self, document_id: UUID) -> None:
        chunk_ids = list(
            self.db.scalars(select(DocumentChunk.Id).where(DocumentChunk.DocumentId == document_id)).all()
        )
        if not chunk_ids:
            return
        self.db.execute(
            update(RAGSource).where(RAGSource.ChunkId.in_(chunk_ids)).values(ChunkId=None)
        )

    def clear_chunks(self, document_id: UUID) -> None:
        self._detach_chunk_references(document_id)
        existing = self.db.scalars(
            select(DocumentChunk).where(DocumentChunk.DocumentId == document_id)
        ).all()
        for row in existing:
            self.db.delete(row)
        self.db.commit()

    def replace_chunks(self, document_id: UUID, chunks: list[DocumentChunk]) -> None:
        self._detach_chunk_references(document_id)
        existing = self.db.scalars(
            select(DocumentChunk).where(DocumentChunk.DocumentId == document_id)
        ).all()
        for row in existing:
            self.db.delete(row)
        self.db.flush()
        for chunk in chunks:
            self.db.add(chunk)
        self.db.commit()

    def count_by_status(self) -> dict[str, int]:
        rows = self.db.execute(
            select(Document.Status, func.count())
            .where(Document.IsDeleted == False)
            .group_by(Document.Status)
        ).all()
        return {status.value if hasattr(status, "value") else str(status): count for status, count in rows}


class ConversationRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, conversation: Conversation) -> Conversation:
        self.db.add(conversation)
        self.db.commit()
        self.db.refresh(conversation)
        return conversation

    def get(self, conversation_id: UUID) -> Conversation | None:
        return self.db.scalar(
            select(Conversation).where(
                Conversation.Id == conversation_id, Conversation.IsDeleted == False
            )
        )

    def list(self, skip: int = 0, limit: int = 100) -> list[Conversation]:
        return list(
            self.db.scalars(
                select(Conversation)
                .where(Conversation.IsDeleted == False)
                .order_by(Conversation.UpdatedAt.desc())
                .offset(skip)
                .limit(limit)
            ).all()
        )

    def update(self, conversation: Conversation) -> Conversation:
        self.db.add(conversation)
        self.db.commit()
        self.db.refresh(conversation)
        return conversation

    def soft_delete(self, conversation: Conversation) -> Conversation:
        conversation.IsDeleted = True
        return self.update(conversation)

    def message_count(self, conversation_id: UUID) -> int:
        return (
            self.db.scalar(
                select(func.count()).select_from(Message).where(Message.ConversationId == conversation_id)
            )
            or 0
        )

    def get_messages(self, conversation_id: UUID) -> list[Message]:
        return list(
            self.db.scalars(
                select(Message)
                .options(selectinload(Message.Sources))
                .where(Message.ConversationId == conversation_id)
                .order_by(Message.CreatedAt.asc())
            ).all()
        )

    def add_message(self, message: Message) -> Message:
        self.db.add(message)
        self.db.commit()
        self.db.refresh(message)
        return message

    def add_sources(self, sources: list[RAGSource]) -> None:
        for source in sources:
            self.db.add(source)
        self.db.commit()

    def get_message(self, message_id: UUID) -> Message | None:
        return self.db.scalar(
            select(Message).options(selectinload(Message.Sources)).where(Message.Id == message_id)
        )

    def delete_messages_after(self, conversation_id: UUID, after_message: Message) -> None:
        messages = self.db.scalars(
            select(Message)
            .where(
                Message.ConversationId == conversation_id,
                Message.CreatedAt >= after_message.CreatedAt,
            )
            .order_by(Message.CreatedAt.asc())
        ).all()
        for msg in messages:
            self.db.delete(msg)
        self.db.commit()

    def clear_messages(self, conversation_id: UUID) -> None:
        messages = self.db.scalars(
            select(Message).where(Message.ConversationId == conversation_id)
        ).all()
        for msg in messages:
            self.db.delete(msg)
        self.db.commit()

    def count_conversations(self) -> int:
        return (
            self.db.scalar(
                select(func.count()).select_from(Conversation).where(Conversation.IsDeleted == False)
            )
            or 0
        )

    def count_user_questions(self) -> int:
        return (
            self.db.scalar(
                select(func.count())
                .select_from(Message)
                .join(Conversation, Conversation.Id == Message.ConversationId)
                .where(Message.Role == MessageRole.user, Conversation.IsDeleted == False)
            )
            or 0
        )
