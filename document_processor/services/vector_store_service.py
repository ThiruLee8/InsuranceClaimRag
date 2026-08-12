from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

# Azure Functions Python base image ships an older system sqlite; Chroma needs >= 3.35.
try:
    import pysqlite3 as sqlite3  # type: ignore
    import sys

    sys.modules["sqlite3"] = sqlite3
except Exception:
    pass

import chromadb
from chromadb.config import Settings as ChromaSettings

from shared.configuration import get_settings
from shared.logging_utils import get_logger

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


class VectorStoreService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = None
        self._collection = None

    @property
    def client(self):
        if self._client is None:
            self._client = chromadb.HttpClient(
                host=self.settings.chroma_host,
                port=self.settings.chroma_port,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    def ensure_collection(self) -> None:
        self._collection = self.client.get_or_create_collection(
            name=self.settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("chroma_collection_ready", collection=self.settings.chroma_collection)

    @property
    def collection(self):
        if self._collection is None:
            self.ensure_collection()
        return self._collection

    def upsert_chunks(
        self,
        *,
        document_id: UUID,
        file_name: str,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not ids:
            return
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info(
            "chroma_upserted",
            document_id=str(document_id),
            file_name=file_name,
            count=len(ids),
        )

    def delete_document(self, document_id: UUID) -> None:
        try:
            self.collection.delete(where={"document_id": str(document_id)})
            logger.info("chroma_document_deleted", document_id=str(document_id))
        except Exception as exc:  # noqa: BLE001
            logger.warning("chroma_delete_failed", document_id=str(document_id), error=str(exc))

    def clear_collection(self) -> None:
        name = self.settings.chroma_collection
        try:
            self.client.delete_collection(name)
            logger.info("chroma_collection_deleted", collection=name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("chroma_collection_delete_failed", collection=name, error=str(exc))
        self._collection = None
        self.ensure_collection()
        logger.info("chroma_collection_cleared", collection=name)

    def search(
        self,
        *,
        query_embedding: list[float],
        top_k: int,
        similarity_threshold: float,
    ) -> list[VectorSearchHit]:
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[VectorSearchHit] = []
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        for idx, chunk_id in enumerate(ids):
            distance = float(distances[idx]) if distances else 1.0
            # Cosine distance -> similarity
            score = 1.0 - distance
            if score < similarity_threshold:
                continue
            meta = metadatas[idx] or {}
            hits.append(
                VectorSearchHit(
                    chunk_id=str(chunk_id),
                    document_id=str(meta.get("document_id", "")),
                    file_name=str(meta.get("file_name", "")),
                    page_number=meta.get("page_number"),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    content=documents[idx] or "",
                    score=score,
                )
            )
        logger.info("chroma_search_completed", hit_count=len(hits), top_k=top_k)
        return hits

    def health_check(self) -> str:
        try:
            self.ensure_collection()
            self.client.heartbeat()
            return "ok"
        except Exception as exc:  # noqa: BLE001
            logger.error("chroma_health_failed", error=str(exc))
            return "unavailable"
