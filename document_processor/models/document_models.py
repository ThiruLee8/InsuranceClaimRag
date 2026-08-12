from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, Enum, Integer, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from shared.database import Base


class DocumentStatus(str, enum.Enum):
    Uploaded = "Uploaded"
    Queued = "Queued"
    Processing = "Processing"
    ExtractingText = "ExtractingText"
    Chunking = "Chunking"
    GeneratingEmbeddings = "GeneratingEmbeddings"
    Indexing = "Indexing"
    Completed = "Completed"
    Processed = "Processed"
    Failed = "Failed"
    Deleted = "Deleted"

    @classmethod
    def active_processing_values(cls) -> set[str]:
        return {
            cls.Processing.value,
            cls.ExtractingText.value,
            cls.Chunking.value,
            cls.GeneratingEmbeddings.value,
            cls.Indexing.value,
        }


class Document(Base):
    __tablename__ = "Documents"

    Id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    FileName: Mapped[str] = mapped_column(String(512), nullable=False)
    OriginalFileName: Mapped[str] = mapped_column(String(512), nullable=False)
    BlobContainerName: Mapped[str] = mapped_column(String(256), nullable=False)
    BlobName: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)
    BlobUrl: Mapped[str] = mapped_column(String(1024), nullable=False)
    ContentType: Mapped[str] = mapped_column(String(128), nullable=False)
    FileSize: Mapped[int] = mapped_column(BigInteger, nullable=False)
    FileHash: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    Status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus, name="DocumentStatus", native_enum=False),
        nullable=False,
        default=DocumentStatus.Uploaded,
    )
    ProcessingError: Mapped[str | None] = mapped_column(Text, nullable=True)
    PageCount: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ChunkSize: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ChunkOverlap: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ProgressPercentage: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    CurrentStep: Mapped[str | None] = mapped_column(String(128), nullable=True)
    TotalChunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ProcessedChunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    StartedAt: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    UploadedAt: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.sysutcdatetime())
    ProcessedAt: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    UpdatedAt: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    RetryCount: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    CorrelationId: Mapped[str | None] = mapped_column(String(64), nullable=True)
    CreatedBy: Mapped[str | None] = mapped_column(String(256), nullable=True)
    IsDeleted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class DocumentChunk(Base):
    __tablename__ = "DocumentChunks"

    Id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    DocumentId: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), nullable=False, index=True)
    ChunkIndex: Mapped[int] = mapped_column(Integer, nullable=False)
    PageNumber: Mapped[int | None] = mapped_column(Integer, nullable=True)
    Content: Mapped[str] = mapped_column(Text, nullable=False)
    CharacterCount: Mapped[int] = mapped_column(Integer, nullable=False)
    CreatedAt: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.sysutcdatetime())
