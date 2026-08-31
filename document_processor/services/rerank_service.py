from __future__ import annotations

from threading import Lock

from shared.configuration import get_settings
from shared.logging_utils import get_logger

logger = get_logger(__name__)


class RerankService:
    """Lazy-loaded cross-encoder reranker for query↔chunk pairs."""

    _instance: RerankService | None = None
    _lock = Lock()

    def __init__(self) -> None:
        self.settings = get_settings()
        self._model = None

    @classmethod
    def get(cls) -> RerankService:
        with cls._lock:
            if cls._instance is None:
                cls._instance = RerankService()
            return cls._instance

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            model_name = self.settings.rerank_model
            logger.info("rerank_model_loading", model=model_name)
            self._model = CrossEncoder(model_name)
            logger.info("rerank_model_ready", model=model_name)
        return self._model

    def rerank(
        self,
        query: str,
        *,
        texts: list[str],
        top_k: int,
    ) -> list[tuple[int, float]]:
        """Return (original_index, score) sorted by score descending."""
        if not texts:
            return []
        model = self._ensure_model()
        pairs = [(query, text) for text in texts]
        scores = model.predict(pairs)
        ranked = sorted(
            ((idx, float(scores[idx])) for idx in range(len(texts))),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[: max(1, top_k)]
