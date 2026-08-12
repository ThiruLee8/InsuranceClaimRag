from __future__ import annotations

from functools import lru_cache
from typing import Protocol

from sentence_transformers import SentenceTransformer

from shared.configuration import get_settings
from shared.logging_utils import get_logger

logger = get_logger(__name__)


class EmbeddingProvider(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbeddingService:
    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        logger.info("loading_embedding_model", model=self.model_name)
        self._model = SentenceTransformer(self.model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        logger.info("embeddings_generated", count=len(texts), model=self.model_name)
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


@lru_cache
def get_embedding_service() -> SentenceTransformerEmbeddingService:
    return SentenceTransformerEmbeddingService()
