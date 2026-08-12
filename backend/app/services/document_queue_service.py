from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from enum import Enum

from azure.storage.queue import QueueClient, TextBase64EncodePolicy, TextBase64DecodePolicy

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class DocumentQueueOperation(str, Enum):
    PROCESS = "PROCESS"
    REPROCESS = "REPROCESS"


@dataclass
class DocumentQueueMessage:
    documentId: str
    operation: str = DocumentQueueOperation.PROCESS.value
    correlationId: str | None = None
    requestedBy: str = "system"

    def to_json(self) -> str:
        payload = asdict(self)
        if not payload.get("correlationId"):
            payload["correlationId"] = str(uuid.uuid4())
        return json.dumps(payload)

    @classmethod
    def from_json(cls, raw: str | bytes) -> "DocumentQueueMessage":
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        data = json.loads(raw)
        return cls(
            documentId=str(data["documentId"]),
            operation=str(data.get("operation") or DocumentQueueOperation.PROCESS.value),
            correlationId=data.get("correlationId"),
            requestedBy=str(data.get("requestedBy") or "system"),
        )


class DocumentQueueService:
    """Shared enqueue helper used by the Web API (and mirrored in Functions)."""

    def __init__(self, connection_string: str | None = None, queue_name: str | None = None) -> None:
        settings = get_settings()
        self.connection_string = connection_string or settings.azure_storage_connection_string
        self.queue_name = queue_name or settings.document_processing_queue_name
        self._client: QueueClient | None = None

    @property
    def client(self) -> QueueClient:
        if self._client is None:
            self._client = QueueClient.from_connection_string(
                self.connection_string,
                self.queue_name,
                message_encode_policy=TextBase64EncodePolicy(),
                message_decode_policy=TextBase64DecodePolicy(),
            )
        return self._client

    def ensure_queue(self) -> None:
        try:
            self.client.create_queue()
            logger.info("document_queue_created", queue=self.queue_name)
        except Exception:
            # QueueAlreadyExists and transient races are fine.
            logger.info("document_queue_ready", queue=self.queue_name)

    def enqueue_document(
        self,
        document_id: str | uuid.UUID,
        *,
        operation: DocumentQueueOperation | str = DocumentQueueOperation.PROCESS,
        correlation_id: str | None = None,
        requested_by: str = "api",
    ) -> DocumentQueueMessage:
        self.ensure_queue()
        message = DocumentQueueMessage(
            documentId=str(document_id),
            operation=operation.value if isinstance(operation, DocumentQueueOperation) else str(operation),
            correlationId=correlation_id or str(uuid.uuid4()),
            requestedBy=requested_by,
        )
        self.client.send_message(message.to_json())
        logger.info(
            "document_queued",
            document_id=message.documentId,
            operation=message.operation,
            correlation_id=message.correlationId,
            requested_by=message.requestedBy,
            queue=self.queue_name,
        )
        return message
