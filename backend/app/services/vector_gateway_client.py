from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VectorSearchHit:
    chunk_id: str
    document_id: str
    file_name: str
    page_number: int | None
    chunk_index: int
    content: str
    score: float


class VectorGatewayClient:
    """
    HTTP client to the document-processor Azure Functions vector APIs.
    The Web API must never open a direct ChromaDB connection.
    """

    def __init__(self, base_url: str | None = None, timeout: float = 60.0) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.document_processor_base_url).rstrip("/")
        self.timeout = timeout

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def search(
        self,
        *,
        question: str,
        top_k: int,
        similarity_threshold: float,
    ) -> list[VectorSearchHit]:
        payload = {
            "question": question,
            "topK": top_k,
            "similarityThreshold": similarity_threshold,
        }
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(self._url("/api/vectors/search"), json=payload)
            response.raise_for_status()
            data = response.json()
        hits: list[VectorSearchHit] = []
        for item in data.get("hits") or []:
            hits.append(
                VectorSearchHit(
                    chunk_id=str(item.get("chunkId") or ""),
                    document_id=str(item.get("documentId") or ""),
                    file_name=str(item.get("fileName") or ""),
                    page_number=item.get("pageNumber"),
                    chunk_index=int(item.get("chunkIndex") or 0),
                    content=str(item.get("content") or ""),
                    score=float(item.get("score") or 0.0),
                )
            )
        logger.info("vector_gateway_search_completed", hit_count=len(hits), top_k=top_k)
        return hits

    def delete_document(self, document_id: UUID) -> None:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.delete(self._url(f"/api/vectors/documents/{document_id}"))
            response.raise_for_status()
        logger.info("vector_gateway_document_deleted", document_id=str(document_id))

    def clear_collection(self) -> None:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(self._url("/api/vectors/clear"), json={})
            response.raise_for_status()
        logger.info("vector_gateway_cleared")

    def health_check(self) -> str:
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(self._url("/api/vectors/health"))
                response.raise_for_status()
                data = response.json()
            return "ok" if data.get("status") == "ok" else "unavailable"
        except Exception as exc:  # noqa: BLE001
            logger.error("vector_gateway_health_failed", error=str(exc))
            return "unavailable"
