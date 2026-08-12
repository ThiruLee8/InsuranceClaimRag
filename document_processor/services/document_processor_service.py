from __future__ import annotations

import uuid
from uuid import UUID

from models.document_models import DocumentChunk, DocumentStatus
from repositories.document_repository import DocumentRepository
from services.blob_storage import BlobService
from services.document_extraction_service import DocumentProcessor
from services.embedding_service import get_embedding_service
from services.vector_store_service import VectorStoreService
from shared.configuration import get_settings
from shared.database import SessionLocal
from shared.logging_utils import get_logger

logger = get_logger(__name__)


class DocumentProcessorService:
    """Single owner of extract → chunk → embed → index. Invoked only by the Queue Trigger."""

    def process(self, document_id: UUID, *, correlation_id: str | None = None, operation: str = "PROCESS") -> None:
        settings = get_settings()
        db = SessionLocal()
        try:
            repo = DocumentRepository(db)
            existing = repo.get(document_id, include_deleted=True)
            if not existing:
                logger.info(
                    "document_missing_skip",
                    document_id=str(document_id),
                    correlation_id=correlation_id,
                )
                return
            if existing.IsDeleted:
                logger.info(
                    "document_deleted_skip",
                    document_id=str(document_id),
                    correlation_id=correlation_id,
                    message=f"Document {document_id} is deleted. Skipping processing.",
                )
                return

            claimed = repo.try_claim_for_processing(document_id, correlation_id=correlation_id)
            if not claimed:
                logger.info(
                    "document_claim_skipped",
                    document_id=str(document_id),
                    correlation_id=correlation_id,
                    status=str(existing.Status.value),
                    reason="already_processing_or_unavailable",
                )
                return

            document = claimed
            logger.info(
                "document_processing_started",
                document_id=str(document_id),
                correlation_id=correlation_id or document.CorrelationId,
                operation=operation,
                blob_name=document.BlobName,
                status=document.Status.value,
                current_step=document.CurrentStep,
            )

            blob_service = BlobService()
            processor = DocumentProcessor()
            vector_store = VectorStoreService()
            embedding_service = get_embedding_service()

            try:
                content = blob_service.download_bytes(document.BlobName)

                repo.mark_progress(
                    document,
                    status=DocumentStatus.ExtractingText,
                    progress=20,
                    step="Extracting text",
                )
                logger.info(
                    "text_extraction_started",
                    document_id=str(document_id),
                    correlation_id=correlation_id,
                )
                extracted = processor.extract_text(
                    content, document.OriginalFileName, document.ContentType
                )
                logger.info(
                    "text_extraction_completed",
                    document_id=str(document_id),
                    correlation_id=correlation_id,
                    pages=extracted.page_count,
                )

                repo.mark_progress(
                    document,
                    status=DocumentStatus.Chunking,
                    progress=40,
                    step="Chunking document",
                )
                chunk_size = document.ChunkSize or settings.chunk_size
                chunk_overlap = (
                    document.ChunkOverlap if document.ChunkOverlap is not None else settings.chunk_overlap
                )
                chunks = processor.chunk_pages(
                    extracted,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                if not chunks:
                    raise ValueError("No extractable text found in document")

                total = len(chunks)
                repo.mark_progress(
                    document,
                    status=DocumentStatus.GeneratingEmbeddings,
                    progress=45,
                    step="Generating embeddings",
                    total_chunks=total,
                    processed_chunks=0,
                )
                logger.info(
                    "chunking_completed",
                    document_id=str(document_id),
                    correlation_id=correlation_id,
                    total_chunks=total,
                )

                vector_store.ensure_collection()
                vector_store.delete_document(document.Id)

                db_chunks: list[DocumentChunk] = []
                ids: list[str] = []
                texts: list[str] = []
                metadatas: list[dict] = []

                for chunk in chunks:
                    chunk_id = uuid.uuid4()
                    db_chunks.append(
                        DocumentChunk(
                            Id=chunk_id,
                            DocumentId=document.Id,
                            ChunkIndex=chunk.chunk_index,
                            PageNumber=chunk.page_number,
                            Content=chunk.content,
                            CharacterCount=len(chunk.content),
                        )
                    )
                    ids.append(str(chunk_id))
                    texts.append(chunk.content)
                    metadatas.append(
                        {
                            "document_id": str(document.Id),
                            "file_name": document.OriginalFileName,
                            "page_number": chunk.page_number or 0,
                            "chunk_id": str(chunk_id),
                            "chunk_index": chunk.chunk_index,
                        }
                    )

                batch_size = max(1, settings.processing_batch_size)
                all_embeddings: list[list[float]] = []
                for start in range(0, total, batch_size):
                    end = min(start + batch_size, total)
                    batch_embeddings = embedding_service.embed_texts(texts[start:end])
                    all_embeddings.extend(batch_embeddings)
                    processed = end
                    # Map 45% → 80% across embedding progress
                    pct = 45 + int((processed / total) * 35)
                    repo.mark_progress(
                        document,
                        status=DocumentStatus.GeneratingEmbeddings,
                        progress=pct,
                        step="Generating embeddings",
                        total_chunks=total,
                        processed_chunks=processed,
                    )
                    logger.info(
                        "embedding_progress",
                        document_id=str(document_id),
                        correlation_id=correlation_id,
                        processed_chunks=processed,
                        total_chunks=total,
                        progress_percentage=pct,
                    )

                repo.mark_progress(
                    document,
                    status=DocumentStatus.Indexing,
                    progress=85,
                    step="Indexing vectors",
                    total_chunks=total,
                    processed_chunks=total,
                )
                logger.info(
                    "vector_indexing_started",
                    document_id=str(document_id),
                    correlation_id=correlation_id,
                    total_chunks=total,
                )

                vector_store.upsert_chunks(
                    document_id=document.Id,
                    file_name=document.OriginalFileName,
                    ids=ids,
                    documents=texts,
                    embeddings=all_embeddings,
                    metadatas=metadatas,
                )
                repo.replace_chunks(document.Id, db_chunks)

                repo.mark_progress(
                    document,
                    status=DocumentStatus.Indexing,
                    progress=95,
                    step="Finalizing",
                    total_chunks=total,
                    processed_chunks=total,
                )
                repo.mark_completed(document, page_count=extracted.page_count, total_chunks=total)
                logger.info(
                    "document_completed",
                    document_id=str(document_id),
                    correlation_id=correlation_id,
                    status="Completed",
                    current_step="Completed",
                    total_chunks=total,
                    pages=extracted.page_count,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "document_failed",
                    document_id=str(document_id),
                    correlation_id=correlation_id,
                    error=str(exc),
                )
                fresh = repo.get(document_id, include_deleted=True)
                if fresh and not fresh.IsDeleted:
                    repo.mark_failed(fresh, error=str(exc))
                raise
        finally:
            db.close()
