from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select, text, update
from sqlalchemy.orm import Session

from models.document_models import Document, DocumentChunk, DocumentStatus


class DocumentRepository:
    def __init__(self, db: Session):
        self.db = db

    def get(self, document_id: UUID, *, include_deleted: bool = False) -> Document | None:
        query = select(Document).where(Document.Id == document_id)
        if not include_deleted:
            query = query.where(Document.IsDeleted == False)  # noqa: E712
        return self.db.scalar(query)

    def get_by_blob_name(self, blob_name: str) -> Document | None:
        return self.db.scalar(
            select(Document).where(Document.BlobName == blob_name, Document.IsDeleted == False)  # noqa: E712
        )

    def list_active(self, *, force: bool = False) -> list[Document]:
        query = select(Document).where(Document.IsDeleted == False)  # noqa: E712
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

    def try_claim_for_processing(self, document_id: UUID, *, correlation_id: str | None) -> Document | None:
        """
        Atomically claim a document for processing.
        Uses a conditional UPDATE so only one worker can claim a given document.
        """
        now = datetime.now(timezone.utc)
        blocked = list(DocumentStatus.active_processing_values())
        stmt = (
            update(Document)
            .where(
                Document.Id == document_id,
                Document.IsDeleted == False,  # noqa: E712
                Document.Status.notin_(blocked),
            )
            .values(
                Status=DocumentStatus.Processing,
                ProgressPercentage=5,
                CurrentStep="Starting processing",
                StartedAt=now,
                UpdatedAt=now,
                CorrelationId=correlation_id,
                ProcessingError=None,
            )
        )
        result = self.db.execute(stmt)
        self.db.commit()
        if not result.rowcount:
            return None
        return self.get(document_id, include_deleted=True)

    def mark_progress(
        self,
        document: Document,
        *,
        status: DocumentStatus,
        progress: int,
        step: str,
        total_chunks: int | None = None,
        processed_chunks: int | None = None,
    ) -> Document:
        document.Status = status
        document.ProgressPercentage = max(0, min(100, progress))
        document.CurrentStep = step
        document.UpdatedAt = datetime.now(timezone.utc)
        if total_chunks is not None:
            document.TotalChunks = total_chunks
        if processed_chunks is not None:
            document.ProcessedChunks = processed_chunks
        return self.update(document)

    def mark_completed(self, document: Document, *, page_count: int, total_chunks: int) -> Document:
        now = datetime.now(timezone.utc)
        document.Status = DocumentStatus.Completed
        document.ProgressPercentage = 100
        document.CurrentStep = "Completed"
        document.PageCount = page_count
        document.TotalChunks = total_chunks
        document.ProcessedChunks = total_chunks
        document.ProcessedAt = now
        document.UpdatedAt = now
        document.ProcessingError = None
        return self.update(document)

    def mark_failed(self, document: Document, *, error: str) -> Document:
        document.Status = DocumentStatus.Failed
        document.CurrentStep = "Failed"
        document.ProcessingError = error[:4000]
        document.RetryCount = (document.RetryCount or 0) + 1
        document.UpdatedAt = datetime.now(timezone.utc)
        return self.update(document)

    def mark_queued(self, document: Document, *, correlation_id: str) -> Document:
        document.Status = DocumentStatus.Queued
        document.ProgressPercentage = 0
        document.CurrentStep = "Queued for processing"
        document.CorrelationId = correlation_id
        document.ProcessingError = None
        document.UpdatedAt = datetime.now(timezone.utc)
        return self.update(document)

    def replace_chunks(self, document_id: UUID, chunks: list[DocumentChunk]) -> None:
        chunk_ids = list(
            self.db.scalars(select(DocumentChunk.Id).where(DocumentChunk.DocumentId == document_id)).all()
        )
        for chunk_id in chunk_ids:
            self.db.execute(
                text("UPDATE RAGSources SET ChunkId = NULL WHERE ChunkId = :cid"),
                {"cid": str(chunk_id)},
            )

        existing = self.db.scalars(select(DocumentChunk).where(DocumentChunk.DocumentId == document_id)).all()
        for row in existing:
            self.db.delete(row)
        self.db.flush()
        for chunk in chunks:
            self.db.add(chunk)
        self.db.commit()
