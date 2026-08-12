from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.models import Document, DocumentStatus
from app.db.repositories import DocumentRepository
from app.schemas import (
    DocumentOut,
    DocumentQueueAcceptedOut,
    DocumentStatusOut,
    ProcessAllAcceptedOut,
    VectorStoreActionOut,
)
from app.services.blob_service import BlobService
from app.services.document_queue_service import DocumentQueueOperation, DocumentQueueService
from app.services.vector_gateway_client import VectorGatewayClient

logger = get_logger(__name__)

CONTENT_TYPE_MAP = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain",
}


class DocumentService:
    """Web API document orchestration. Never runs extract/chunk/embed or touches Chroma here."""

    def __init__(
        self,
        db: Session,
        *,
        blob_service: BlobService | None = None,
        vector_gateway: VectorGatewayClient | None = None,
        queue_service: DocumentQueueService | None = None,
    ) -> None:
        self.settings = get_settings()
        self.db = db
        self.repo = DocumentRepository(db)
        self._blob_service = blob_service
        self._vector_gateway = vector_gateway
        self._queue_service = queue_service

    @property
    def blob_service(self) -> BlobService:
        if self._blob_service is None:
            self._blob_service = BlobService()
        return self._blob_service

    @property
    def vector_gateway(self) -> VectorGatewayClient:
        if self._vector_gateway is None:
            self._vector_gateway = VectorGatewayClient()
        return self._vector_gateway

    @property
    def queue_service(self) -> DocumentQueueService:
        if self._queue_service is None:
            self._queue_service = DocumentQueueService()
        return self._queue_service

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

    def _mark_queued(self, document: Document, *, correlation_id: str, operation: str) -> None:
        now = datetime.now(timezone.utc)
        document.Status = DocumentStatus.Queued
        document.ProgressPercentage = 0
        document.CurrentStep = "Queued for processing"
        document.ProcessingError = None
        document.ProcessedChunks = 0
        document.TotalChunks = document.TotalChunks or 0
        document.CorrelationId = correlation_id
        document.UpdatedAt = now
        if operation == DocumentQueueOperation.REPROCESS.value:
            document.ProcessedAt = None
            document.StartedAt = None
        self.repo.update(document)

    async def upload(
        self,
        file: UploadFile,
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

        document_id = uuid.uuid4()
        safe_name = self.blob_service.sanitize_filename(file.filename)
        blob_name = f"{document_id.hex}/{safe_name}"

        # Create metadata before blob write so the Blob Trigger can resolve the document.
        document = Document(
            Id=document_id,
            FileName=safe_name,
            OriginalFileName=file.filename,
            BlobContainerName=self.settings.azure_storage_container,
            BlobName=blob_name,
            BlobUrl="",
            ContentType=content_type,
            FileSize=len(content),
            FileHash=file_hash,
            Status=DocumentStatus.Uploaded,
            ChunkSize=size,
            ChunkOverlap=overlap,
            ProgressPercentage=0,
            CurrentStep="Uploaded",
            TotalChunks=0,
            ProcessedChunks=0,
            RetryCount=0,
            CreatedBy=created_by,
        )
        document = self.repo.create(document)

        self.blob_service.ensure_container()
        _, blob_url, _ = self.blob_service.upload_bytes(
            content=content,
            original_filename=file.filename,
            content_type=content_type,
            blob_name=blob_name,
        )
        document.BlobUrl = blob_url
        document.UpdatedAt = datetime.now(timezone.utc)
        document = self.repo.update(document)
        logger.info("document_uploaded", document_id=str(document.Id), file_name=safe_name)

        # Blob Trigger is the primary enqueue path. API also enqueues as a reliable local/Azure fallback.
        # Concurrent claim + idempotent vector replace protect against duplicate messages.
        self.enqueue_process(document.Id, operation=DocumentQueueOperation.PROCESS, requested_by="api-upload")
        return self._to_out(self.repo.get(document.Id) or document)

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
            progressPercentage=document.ProgressPercentage or 0,
            currentStep=document.CurrentStep,
            totalChunks=document.TotalChunks or 0,
            processedChunks=document.ProcessedChunks or 0,
            startedAt=document.StartedAt,
            updatedAt=document.UpdatedAt,
            retryCount=document.RetryCount or 0,
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
            self.vector_gateway.delete_document(document.Id)
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

    def enqueue_process(
        self,
        document_id: UUID,
        *,
        operation: DocumentQueueOperation | str = DocumentQueueOperation.REPROCESS,
        requested_by: str = "api",
        force: bool = True,
    ) -> DocumentQueueAcceptedOut:
        document = self.repo.get(document_id)
        if not document:
            raise LookupError("Document not found")
        if document.IsDeleted:
            raise LookupError("Document not found")

        op = operation.value if isinstance(operation, DocumentQueueOperation) else str(operation)
        if (
            not force
            and document.Status in {DocumentStatus.Completed, DocumentStatus.Processed}
            and op == DocumentQueueOperation.PROCESS.value
        ):
            return DocumentQueueAcceptedOut(
                documentId=document.Id,
                status=str(document.Status.value),
                message="Document already completed; skipped (use force/reprocess to queue again).",
            )

        message = self.queue_service.enqueue_document(
            document.Id,
            operation=op,
            requested_by=requested_by,
        )
        self._mark_queued(document, correlation_id=message.correlationId or "", operation=op)
        return DocumentQueueAcceptedOut(
            documentId=document.Id,
            status="Queued",
            message="Document processing has been queued.",
            correlationId=message.correlationId,
        )

    def reprocess(self, document_id: UUID) -> DocumentQueueAcceptedOut:
        return self.enqueue_process(
            document_id,
            operation=DocumentQueueOperation.REPROCESS,
            requested_by="api-reprocess",
            force=True,
        )

    def process_all(self, *, force: bool = False, requested_by: str = "api-process-all") -> ProcessAllAcceptedOut:
        active = self.repo.list(
            skip=0,
            limit=10_000,
        )[0]
        total_found = len(active)
        to_queue = self.repo.list_active_for_process_all(force=force)
        correlation_id = str(uuid.uuid4())
        queued = 0
        for document in to_queue:
            op = (
                DocumentQueueOperation.REPROCESS
                if force or document.Status in {DocumentStatus.Failed, DocumentStatus.Completed, DocumentStatus.Processed}
                else DocumentQueueOperation.PROCESS
            )
            message = self.queue_service.enqueue_document(
                document.Id,
                operation=op,
                correlation_id=correlation_id,
                requested_by=requested_by,
            )
            self._mark_queued(document, correlation_id=message.correlationId or correlation_id, operation=op.value)
            queued += 1

        skipped = total_found - queued
        logger.info(
            "process_all_queued",
            total=total_found,
            queued=queued,
            skipped=skipped,
            force=force,
            correlation_id=correlation_id,
        )
        return ProcessAllAcceptedOut(
            status="Queued",
            totalDocumentsFound=total_found,
            documentsQueued=queued,
            documentsSkipped=skipped,
            correlationId=correlation_id,
        )

    def clear_vector_store(self) -> VectorStoreActionOut:
        self.vector_gateway.clear_collection()
        logger.info("vector_store_cleared_via_functions")
        return VectorStoreActionOut(cleared=True, queued=0, documentIds=[])

    def clear_and_reprocess_all(self) -> ProcessAllAcceptedOut:
        self.vector_gateway.clear_collection()
        return self.process_all(force=True, requested_by="api-clear-and-reprocess")
