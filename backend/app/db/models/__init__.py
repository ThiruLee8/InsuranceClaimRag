import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class DocumentStatus(str, enum.Enum):
    Uploaded = "Uploaded"
    Queued = "Queued"
    Processing = "Processing"
    ExtractingText = "ExtractingText"
    Chunking = "Chunking"
    GeneratingEmbeddings = "GeneratingEmbeddings"
    Indexing = "Indexing"
    Completed = "Completed"
    # Legacy alias kept for older rows/clients
    Processed = "Processed"
    Failed = "Failed"
    Deleted = "Deleted"

    @classmethod
    def is_terminal(cls, status: "DocumentStatus | str") -> bool:
        value = status.value if isinstance(status, DocumentStatus) else str(status)
        return value in {cls.Completed.value, cls.Processed.value, cls.Failed.value, cls.Deleted.value}

    @classmethod
    def is_actively_processing(cls, status: "DocumentStatus | str") -> bool:
        value = status.value if isinstance(status, DocumentStatus) else str(status)
        return value in {
            cls.Processing.value,
            cls.ExtractingText.value,
            cls.Chunking.value,
            cls.GeneratingEmbeddings.value,
            cls.Indexing.value,
        }


class MessageRole(str, enum.Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


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

    Chunks: Mapped[list["DocumentChunk"]] = relationship(
        "DocumentChunk", back_populates="Document", cascade="all, delete-orphan"
    )


class DocumentChunk(Base):
    __tablename__ = "DocumentChunks"

    Id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    DocumentId: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("Documents.Id", ondelete="CASCADE"), nullable=False, index=True
    )
    ChunkIndex: Mapped[int] = mapped_column(Integer, nullable=False)
    PageNumber: Mapped[int | None] = mapped_column(Integer, nullable=True)
    Content: Mapped[str] = mapped_column(Text, nullable=False)
    CharacterCount: Mapped[int] = mapped_column(Integer, nullable=False)
    CreatedAt: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.sysutcdatetime())

    Document: Mapped["Document"] = relationship("Document", back_populates="Chunks")


class Conversation(Base):
    __tablename__ = "Conversations"

    Id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    Title: Mapped[str] = mapped_column(String(512), nullable=False, default="New Conversation")
    CreatedAt: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.sysutcdatetime())
    UpdatedAt: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.sysutcdatetime(), onupdate=func.sysutcdatetime()
    )
    CreatedBy: Mapped[str | None] = mapped_column(String(256), nullable=True)
    IsDeleted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    Messages: Mapped[list["Message"]] = relationship(
        "Message", back_populates="Conversation", cascade="all, delete-orphan"
    )


class Message(Base):
    __tablename__ = "Messages"

    Id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ConversationId: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("Conversations.Id", ondelete="CASCADE"), nullable=False, index=True
    )
    Role: Mapped[MessageRole] = mapped_column(
        Enum(MessageRole, name="MessageRole", native_enum=False),
        nullable=False,
    )
    Content: Mapped[str] = mapped_column(Text, nullable=False)
    CreatedAt: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.sysutcdatetime())
    TokenCount: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ModelName: Mapped[str | None] = mapped_column(String(128), nullable=True)

    Conversation: Mapped["Conversation"] = relationship("Conversation", back_populates="Messages")
    Sources: Mapped[list["RAGSource"]] = relationship(
        "RAGSource", back_populates="Message", cascade="all, delete-orphan"
    )


class RAGSource(Base):
    __tablename__ = "RAGSources"

    Id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    MessageId: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("Messages.Id", ondelete="CASCADE"), nullable=False, index=True
    )
    DocumentId: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("Documents.Id", ondelete="NO ACTION"), nullable=True
    )
    ChunkId: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("DocumentChunks.Id", ondelete="NO ACTION"), nullable=True
    )
    PageNumber: Mapped[int | None] = mapped_column(Integer, nullable=True)
    RelevanceScore: Mapped[float | None] = mapped_column(Float, nullable=True)
    CreatedAt: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.sysutcdatetime())

    Message: Mapped["Message"] = relationship("Message", back_populates="Sources")
