from __future__ import annotations

import uuid

from azure.storage.queue import QueueClient, TextBase64DecodePolicy, TextBase64EncodePolicy

from models.queue_message_models import DocumentQueueMessage, DocumentQueueOperation
from shared.configuration import get_settings
from shared.logging_utils import get_logger

logger = get_logger(__name__)


class DocumentQueueService:
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
            logger.info("document_queue_ready", queue=self.queue_name)

    def enqueue_document(
        self,
        document_id: str | uuid.UUID,
        *,
        operation: DocumentQueueOperation | str = DocumentQueueOperation.PROCESS,
        correlation_id: str | None = None,
        requested_by: str = "functions",
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
        )
        return message
