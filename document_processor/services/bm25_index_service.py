from __future__ import annotations

import re
from dataclasses import dataclass
from threading import Lock
from typing import Any
from uuid import UUID

from rank_bm25 import BM25Okapi

from shared.logging_utils import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


@dataclass
class Bm25Document:
    chunk_id: str
    document_id: str
    file_name: str
    page_number: int | None
    chunk_index: int
    content: str


class Bm25IndexService:
    """In-process BM25 index synced with Chroma chunk upserts/deletes."""

    _instance: Bm25IndexService | None = None
    _lock = Lock()

    def __init__(self) -> None:
        self._docs: dict[str, Bm25Document] = {}
        self._ids: list[str] = []
        self._corpus_tokens: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._dirty = True

    @classmethod
    def get(cls) -> Bm25IndexService:
        with cls._lock:
            if cls._instance is None:
                cls._instance = Bm25IndexService()
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._instance = Bm25IndexService()

    def _rebuild(self) -> None:
        self._ids = list(self._docs.keys())
        self._corpus_tokens = [tokenize(self._docs[cid].content) for cid in self._ids]
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._ids else None
        self._dirty = False
        logger.info("bm25_index_rebuilt", documents=len(self._ids))

    def document_count(self) -> int:
        return len(self._docs)

    def ensure_ready(self) -> None:
        if self._dirty:
            self._rebuild()

    def upsert(
        self,
        *,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        for idx, chunk_id in enumerate(ids):
            meta = metadatas[idx] if idx < len(metadatas) else {}
            self._docs[str(chunk_id)] = Bm25Document(
                chunk_id=str(chunk_id),
                document_id=str(meta.get("document_id", "")),
                file_name=str(meta.get("file_name", "")),
                page_number=meta.get("page_number"),
                chunk_index=int(meta.get("chunk_index", 0)),
                content=documents[idx] if idx < len(documents) else "",
            )
        self._dirty = True

    def delete_document(self, document_id: UUID) -> None:
        doc_id = str(document_id)
        to_remove = [cid for cid, doc in self._docs.items() if doc.document_id == doc_id]
        for cid in to_remove:
            del self._docs[cid]
        if to_remove:
            self._dirty = True

    def clear(self) -> None:
        self._docs.clear()
        self._ids = []
        self._corpus_tokens = []
        self._bm25 = None
        self._dirty = False

    def load_from_chroma(self, collection) -> None:
        """Cold-start rebuild from Chroma collection contents."""
        try:
            result = collection.get(include=["documents", "metadatas"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("bm25_chroma_load_failed", error=str(exc))
            return
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        self._docs.clear()
        for idx, chunk_id in enumerate(ids):
            meta = metadatas[idx] or {}
            self._docs[str(chunk_id)] = Bm25Document(
                chunk_id=str(chunk_id),
                document_id=str(meta.get("document_id", "")),
                file_name=str(meta.get("file_name", "")),
                page_number=meta.get("page_number"),
                chunk_index=int(meta.get("chunk_index", 0)),
                content=documents[idx] or "",
            )
        self._dirty = True
        self.ensure_ready()
        logger.info("bm25_loaded_from_chroma", documents=len(self._docs))

    def search(self, query: str, *, top_k: int) -> list[tuple[Bm25Document, float]]:
        self.ensure_ready()
        if not self._bm25 or not self._ids or not query.strip():
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            ((self._ids[i], float(scores[i])) for i in range(len(self._ids))),
            key=lambda x: x[1],
            reverse=True,
        )
        hits: list[tuple[Bm25Document, float]] = []
        for chunk_id, score in ranked[: max(1, top_k)]:
            if score <= 0:
                continue
            doc = self._docs.get(chunk_id)
            if doc:
                hits.append((doc, score))
        return hits
