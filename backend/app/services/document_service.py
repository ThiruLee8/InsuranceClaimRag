from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from fastapi import BackgroundTasks, UploadFile
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.models import Document, DocumentChunk, DocumentStatus
from app.db.repositories import DocumentRepository
from app.schemas import DocumentOut, DocumentStatusOut, VectorStoreActionOut
from app.services.blob_service import BlobService
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import get_embedding_service
from app.services.vector_store_service import VectorStoreService

logger = get_logger(__name__)

CONTENT_TYPE_MAP = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain",
}


class DocumentService:
    def __init__(
        self,
        db: Session,
        *,
        blob_service: BlobService | None = None,
        vector_store: VectorStoreService | None = None,
        processor: DocumentProcessor | None = None,
    ) -> None:
        self.settings = get_settings()
        self.db = db
        self.repo = DocumentRepository(db)
        self._blob_service = blob_service
        self.processor = processor or DocumentProcessor()
        self._vector_store = vector_store

    @property
    def blob_service(self) -> BlobService:
        if self._blob_service is None:
            self._blob_service = BlobService()
        return self._blob_service

    @property
    def vector_store(self) -> VectorStoreService:
        if self._vector_store is None:
            self._vector_store = VectorStoreService()
        return self._vector_store

    def _to_out(self, document: Document) -> DocumentOut:
        out = DocumentOut.model_validate(document)
        out.chunkCount = self.repo.chunk_count(document.Id)
        return out

    def _normalize_chunk_settings(
        self, chunk_size: int | None, chunk_overlap: int | None
    ) -> tuple[int, int]:
        size = chunk_size if chunk_size is not None else self.settings.chunk_size
        overlap = chunk_overlap if chunk_overlap is not None else self.settings.chunk_overlap
        if size < 100 or size > 8000:
            raise ValueError("chunkSize must be between 100 and 8000")
        if overlap < 0 or overlap >= size:
            raise ValueError("chunkOverlap must be >= 0 and less than chunkSize")
        return size, overlap

    async def upload(
        self,
        file: UploadFile,
        background_tasks: BackgroundTasks,
        created_by: str | None = None,
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> DocumentOut:
        if not file.filename:
            raise ValueError("Filename is required")

        suffix = Path(file.filename).suffix.lower()
        if suffix not in self.settings.allowed_extensions:
            raise ValueError(f"Unsupported file type. Allowed: {', '.join(self.settings.allowed_extensions)}")

        content = await file.read()
        if not content:
            raise ValueError("Uploaded file is empty")
        if len(content) > self.settings.max_upload_bytes:
            raise ValueError(f"File exceeds maximum size of {self.settings.max_upload_size_mb} MB")

        size, overlap = self._normalize_chunk_settings(chunk_size, chunk_overlap)

        content_type = file.content_type or CONTENT_TYPE_MAP.get(suffix, "application/octet-stream")
        if suffix in CONTENT_TYPE_MAP and content_type == "application/octet-stream":
            content_type = CONTENT_TYPE_MAP[suffix]

        file_hash = self.blob_service.compute_hash(content)
        existing = self.repo.get_by_hash(file_hash)
        if existing and existing.Status != DocumentStatus.Failed:
            raise FileExistsError("A document with the same content already exists")

        self.blob_service.ensure_container()
        blob_name, blob_url, safe_name = self.blob_service.upload_bytes(
            content=content,
            original_filename=file.filename,
            content_type=content_type,
        )

        document = Document(
            Id=uuid.uuid4(),
            FileName=safe_name,
            OriginalFileName=file.filename,
            BlobContainerName=self.settings.azure_storage_container,
            BlobName=blob_name,
            BlobUrl=blob_url,
            ContentType=content_type,
            FileSize=len(content),
            FileHash=file_hash,
            Status=DocumentStatus.Uploaded,
            ChunkSize=size,
            ChunkOverlap=overlap,
            CreatedBy=created_by,
        )
        document = self.repo.create(document)
        logger.info("document_uploaded", document_id=str(document.Id), file_name=safe_name)

        background_tasks.add_task(process_document_task, document.Id)
        return self._to_out(document)

    def list_documents(
        self,
        *,
        search: str | None = None,
        status: DocumentStatus | None = None,
        content_type: str | None = None,
        skip: int = 0,
        limit: int = 50,
    ) -> tuple[list[DocumentOut], int]:
        items, total = self.repo.list(
            search=search, status=status, content_type=content_type, skip=skip, limit=limit
        )
        return [self._to_out(item) for item in items], total

    def get_document(self, document_id: UUID) -> DocumentOut:
        document = self.repo.get(document_id)
        if not document:
            raise LookupError("Document not found")
        return self._to_out(document)

    def get_status(self, document_id: UUID) -> DocumentStatusOut:
        document = self.repo.get(document_id)
        if not document:
            raise LookupError("Document not found")
        return DocumentStatusOut(
            id=document.Id,
            status=document.Status,
            processingError=document.ProcessingError,
            pageCount=document.PageCount,
            chunkCount=self.repo.chunk_count(document.Id),
            processedAt=document.ProcessedAt,
        )

    def download(self, document_id: UUID) -> tuple[Document, bytes]:
        document = self.repo.get(document_id)
        if not document:
            raise LookupError("Document not found")
        data = self.blob_service.download_bytes(document.BlobName)
        return document, data

    def delete(self, document_id: UUID) -> None:
        document = self.repo.get(document_id)
        if not document:
            raise LookupError("Document not found")
        try:
            self.vector_store.delete_document(document.Id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("vector_delete_failed", error=str(exc), document_id=str(document_id))
        try:
            self.blob_service.delete_blob(document.BlobName)
        except Exception as exc:  # noqa: BLE001
            logger.warning("blob_delete_failed", error=str(exc), document_id=str(document_id))
        try:
            self.repo.clear_chunks(document.Id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("chunk_clear_failed", error=str(exc), document_id=str(document_id))
        self.repo.soft_delete(document)
        logger.info("document_soft_deleted", document_id=str(document_id))

    def reprocess(self, document_id: UUID, background_tasks: BackgroundTasks) -> DocumentOut:
        document = self.repo.get(document_id)
        if not document:
            raise LookupError("Document not found")
        document.Status = DocumentStatus.Uploaded
        document.ProcessingError = None
        document.ProcessedAt = None
        self.repo.update(document)
        background_tasks.add_task(process_document_task, document.Id)
        return self._to_out(document)

    def clear_vector_store(self) -> VectorStoreActionOut:
        self.vector_store.clear_collection()
        logger.info("vector_store_cleared")
        return VectorStoreActionOut(cleared=True, queued=0, documentIds=[])

    def reprocess_all(self, background_tasks: BackgroundTasks) -> VectorStoreActionOut:
        ids = self.repo.list_active_ids()
        for document_id in ids:
            document = self.repo.get(document_id)
            if not document:
                continue
            document.Status = DocumentStatus.Uploaded
            document.ProcessingError = None
            document.ProcessedAt = None
            self.repo.update(document)
            background_tasks.add_task(process_document_task, document_id)
        logger.info("reprocess_all_queued", count=len(ids))
        return VectorStoreActionOut(cleared=False, queued=len(ids), documentIds=ids)

    def clear_and_reprocess_all(self, background_tasks: BackgroundTasks) -> VectorStoreActionOut:
        self.vector_store.clear_collection()
        result = self.reprocess_all(background_tasks)
        result.cleared = True
        return result


def process_document_task(document_id: UUID) -> None:
    """Background processing entrypoint (isolated for future worker migration)."""
    from app.db.database import SessionLocal

    db = SessionLocal()
    try:
        _process_document(db, document_id)
    finally:
        db.close()


def _process_document(db: Session, document_id: UUID) -> None:
    settings = get_settings()
    repo = DocumentRepository(db)
    blob_service = BlobService()
    processor = DocumentProcessor()
    vector_store = VectorStoreService()
    embedding_service = get_embedding_service()

    document = repo.get(document_id)
    if not document:
        logger.error("process_document_missing", document_id=str(document_id))
        return

    document.Status = DocumentStatus.Processing
    document.ProcessingError = None
    repo.update(document)
    logger.info("document_processing_started", document_id=str(document_id))

    try:
        content = blob_service.download_bytes(document.BlobName)
        extracted = processor.extract_text(content, document.OriginalFileName, document.ContentType)
        chunk_size = document.ChunkSize or settings.chunk_size
        chunk_overlap = document.ChunkOverlap if document.ChunkOverlap is not None else settings.chunk_overlap
        chunks = processor.chunk_pages(
            extracted,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if not chunks:
            raise ValueError("No extractable text found in document")

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

        embeddings = embedding_service.embed_texts(texts)
        vector_store.upsert_chunks(
            document_id=document.Id,
            file_name=document.OriginalFileName,
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        repo.replace_chunks(document.Id, db_chunks)

        document.Status = DocumentStatus.Processed
        document.PageCount = extracted.page_count
        document.ProcessedAt = datetime.now(timezone.utc)
        document.ProcessingError = None
        repo.update(document)
        logger.info(
            "document_processing_completed",
            document_id=str(document_id),
            chunks=len(chunks),
            pages=extracted.page_count,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("document_processing_failed", document_id=str(document_id), error=str(exc))
        document = repo.get(document_id)
        if document:
            document.Status = DocumentStatus.Failed
            document.ProcessingError = str(exc)
            repo.update(document)
