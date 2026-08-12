from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.agents import get_agent
from app.services.embedding_service import EmbeddingProvider, get_embedding_service
from app.services.llm_service import LLMProvider, OllamaLLMService
from app.services.vector_store_service import VectorSearchHit, VectorStoreService

logger = get_logger(__name__)


@dataclass
class RAGResult:
    answer: str
    sources: list[VectorSearchHit]
    agent_id: str
    model: str


class RAGService:
    def __init__(
        self,
        *,
        embedding_service: EmbeddingProvider | None = None,
        vector_store: VectorStoreService | None = None,
        llm_service: LLMProvider | None = None,
    ) -> None:
        self.settings = get_settings()
        self.embedding_service = embedding_service or get_embedding_service()
        self.vector_store = vector_store or VectorStoreService()
        self.llm_service = llm_service or OllamaLLMService()

    def _build_context(self, hits: list[VectorSearchHit]) -> str:
        blocks: list[str] = []
        for i, hit in enumerate(hits, start=1):
            page = hit.page_number if hit.page_number is not None else "N/A"
            blocks.append(
                f"[Source {i}] Document: {hit.file_name} | Page: {page} | Score: {hit.score:.3f}\n"
                f"{hit.content}"
            )
        return "\n\n".join(blocks)

    async def answer(
        self,
        question: str,
        *,
        agent_id: str | None = None,
        model: str | None = None,
    ) -> RAGResult:
        agent = get_agent(agent_id)
        selected_model = (model or self.settings.ollama_model).strip() or self.settings.ollama_model

        query_embedding = self.embedding_service.embed_query(question)
        hits = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.settings.top_k,
            similarity_threshold=self.settings.similarity_threshold,
        )
        logger.info(
            "rag_search_complete",
            question_len=len(question),
            hits=len(hits),
            agent_id=agent.id,
            model=selected_model,
        )

        if not hits:
            return RAGResult(
                answer="I could not find sufficient information in the provided documents.",
                sources=[],
                agent_id=agent.id,
                model=selected_model,
            )

        context = self._build_context(hits)
        prompt = (
            f"Document context:\n{context}\n\n"
            f"User question: {question}\n\n"
            "Provide a clear answer based only on the context above."
        )
        answer = await self.llm_service.generate(
            prompt=prompt,
            system=agent.system_prompt,
            model=selected_model,
        )
        if not answer:
            answer = "I could not find sufficient information in the provided documents."
        return RAGResult(
            answer=answer,
            sources=hits,
            agent_id=agent.id,
            model=selected_model,
        )
