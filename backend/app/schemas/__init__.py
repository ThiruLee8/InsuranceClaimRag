from datetime import datetime
from typing import Any, Generic, Optional, TypeVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.db.models import DocumentStatus, MessageRole

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    success: bool = True
    message: str = "OK"
    data: Optional[T] = None
    errorCode: Optional[str] = None


class ErrorResponse(BaseModel):
    success: bool = False
    message: str
    errorCode: str
    details: Optional[Any] = None


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(validation_alias="Id")
    fileName: str = Field(validation_alias="FileName")
    originalFileName: str = Field(validation_alias="OriginalFileName")
    blobContainerName: str = Field(validation_alias="BlobContainerName")
    blobName: str = Field(validation_alias="BlobName")
    blobUrl: str = Field(validation_alias="BlobUrl")
    contentType: str = Field(validation_alias="ContentType")
    fileSize: int = Field(validation_alias="FileSize")
    fileHash: str = Field(validation_alias="FileHash")
    status: DocumentStatus = Field(validation_alias="Status")
    processingError: Optional[str] = Field(default=None, validation_alias="ProcessingError")
    pageCount: Optional[int] = Field(default=None, validation_alias="PageCount")
    chunkSize: Optional[int] = Field(default=None, validation_alias="ChunkSize")
    chunkOverlap: Optional[int] = Field(default=None, validation_alias="ChunkOverlap")
    progressPercentage: int = Field(default=0, validation_alias="ProgressPercentage")
    currentStep: Optional[str] = Field(default=None, validation_alias="CurrentStep")
    totalChunks: int = Field(default=0, validation_alias="TotalChunks")
    processedChunks: int = Field(default=0, validation_alias="ProcessedChunks")
    startedAt: Optional[datetime] = Field(default=None, validation_alias="StartedAt")
    uploadedAt: datetime = Field(validation_alias="UploadedAt")
    processedAt: Optional[datetime] = Field(default=None, validation_alias="ProcessedAt")
    updatedAt: Optional[datetime] = Field(default=None, validation_alias="UpdatedAt")
    retryCount: int = Field(default=0, validation_alias="RetryCount")
    correlationId: Optional[str] = Field(default=None, validation_alias="CorrelationId")
    createdBy: Optional[str] = Field(default=None, validation_alias="CreatedBy")
    chunkCount: Optional[int] = None


class DocumentStatusOut(BaseModel):
    id: UUID
    status: DocumentStatus
    processingError: Optional[str] = None
    pageCount: Optional[int] = None
    chunkCount: int = 0
    processedAt: Optional[datetime] = None
    progressPercentage: int = 0
    currentStep: Optional[str] = None
    totalChunks: int = 0
    processedChunks: int = 0
    startedAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None
    retryCount: int = 0


class DocumentQueueAcceptedOut(BaseModel):
    documentId: UUID
    status: str = "Queued"
    message: str = "Document processing has been queued."
    correlationId: Optional[str] = None


class ProcessAllAcceptedOut(BaseModel):
    status: str = "Queued"
    totalDocumentsFound: int
    documentsQueued: int
    documentsSkipped: int
    correlationId: Optional[str] = None


class DocumentListOut(BaseModel):
    items: list[DocumentOut]
    total: int


class ConversationCreate(BaseModel):
    title: Optional[str] = "New Conversation"
    createdBy: Optional[str] = None


class ConversationUpdate(BaseModel):
    title: str


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(validation_alias="Id")
    title: str = Field(validation_alias="Title")
    createdAt: datetime = Field(validation_alias="CreatedAt")
    updatedAt: datetime = Field(validation_alias="UpdatedAt")
    createdBy: Optional[str] = Field(default=None, validation_alias="CreatedBy")
    messageCount: Optional[int] = None


class RAGSourceOut(BaseModel):
    documentId: Optional[UUID] = None
    chunkId: Optional[UUID] = None
    fileName: Optional[str] = None
    pageNumber: Optional[int] = None
    relevanceScore: Optional[float] = None
    content: Optional[str] = None


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(validation_alias="Id")
    conversationId: UUID = Field(validation_alias="ConversationId")
    role: MessageRole = Field(validation_alias="Role")
    content: str = Field(validation_alias="Content")
    createdAt: datetime = Field(validation_alias="CreatedAt")
    tokenCount: Optional[int] = Field(default=None, validation_alias="TokenCount")
    modelName: Optional[str] = Field(default=None, validation_alias="ModelName")
    sources: list[RAGSourceOut] = []


class ChatRequest(BaseModel):
    conversationId: Optional[UUID] = None
    question: str = Field(min_length=1, max_length=4000)
    regenerateMessageId: Optional[UUID] = None
    # When true, do not insert another user row (retry after a failed generation).
    reuseLastUserMessage: bool = False
    agentId: Optional[str] = None
    model: Optional[str] = None
    # When true, include chunk text + search rewrite metadata in sources/response.
    debug: bool = False


class ChatResponse(BaseModel):
    conversationId: UUID
    messageId: UUID
    answer: str
    sources: list[RAGSourceOut]
    agentId: Optional[str] = None
    model: Optional[str] = None
    originalQuestion: Optional[str] = None
    searchQuery: Optional[str] = None
    searchMode: Optional[str] = None


class RetrieveRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    topK: Optional[int] = Field(default=None, ge=1, le=50)
    similarityThreshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    searchMode: Optional[str] = None
    rewrite: bool = True
    rerank: Optional[bool] = None
    debug: bool = True


class RetrieveResponse(BaseModel):
    originalQuestion: str
    searchQuery: str
    searchMode: Optional[str] = None
    sources: list[RAGSourceOut]


class TraceRetrievedChunk(BaseModel):
    documentId: Optional[str] = None
    chunkId: Optional[str] = None
    fileName: Optional[str] = None
    pageNumber: Optional[int] = None
    relevanceScore: Optional[float] = None
    content: Optional[str] = None


class TraceOut(BaseModel):
    traceId: str
    timestamp: str
    conversationId: Optional[str] = None
    messageId: Optional[str] = None
    question: str
    originalQuestion: Optional[str] = None
    searchQuery: Optional[str] = None
    searchMode: Optional[str] = None
    agentId: Optional[str] = None
    model: Optional[str] = None
    retrieved: list[TraceRetrievedChunk] = []
    answer: str


class TraceListOut(BaseModel):
    total: int
    items: list[TraceOut]


class AgentOut(BaseModel):
    id: str
    name: str
    description: str


class AgentsResponse(BaseModel):
    agents: list[AgentOut]
    models: list[str]
    defaultAgentId: str
    defaultModel: str


class VectorStoreActionOut(BaseModel):
    cleared: bool = False
    queued: int = 0
    documentIds: list[UUID] = Field(default_factory=list)


class DashboardStats(BaseModel):
    totalDocuments: int
    processedDocuments: int
    processingDocuments: int
    failedDocuments: int
    totalConversations: int
    totalQuestions: int
    recentDocuments: list[DocumentOut]
    recentConversations: list[ConversationOut]


class HealthOut(BaseModel):
    status: str
    sqlServer: str
    azurite: str
    chroma: str
    ollama: str
