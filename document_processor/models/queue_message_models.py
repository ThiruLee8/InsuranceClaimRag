from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from enum import Enum


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
