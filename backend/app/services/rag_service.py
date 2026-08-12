from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.agents import get_agent
from app.services.llm_service import LLMProvider, OllamaLLMService
from app.services.vector_gateway_client import VectorGatewayClient, VectorSearchHit

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
        vector_gateway: VectorGatewayClient | None = None,
        llm_service: LLMProvider | None = None,
    ) -> None:
        self.settings = get_settings()
        self.vector_gateway = vector_gateway or VectorGatewayClient()
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

    def _prepare(
        self,
        question: str,
        *,
        agent_id: str | None = None,
        model: str | None = None,
    ) -> tuple[Any, str, list[VectorSearchHit], str]:
        agent = get_agent(agent_id)
        selected_model = (model or self.settings.ollama_model).strip() or self.settings.ollama_model

        hits = self.vector_gateway.search(
            question=question,
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
        prompt = ""
        if hits:
            prompt = (
                f"Document context:\n{self._build_context(hits)}\n\n"
                f"User question: {question}\n\n"
                "Provide a clear answer based only on the context above."
            )
        return agent, selected_model, hits, prompt

    async def answer(
        self,
        question: str,
        *,
        agent_id: str | None = None,
        model: str | None = None,
    ) -> RAGResult:
        agent, selected_model, hits, prompt = self._prepare(
            question, agent_id=agent_id, model=model
        )

        if not hits:
            return RAGResult(
                answer="I could not find sufficient information in the provided documents.",
                sources=[],
                agent_id=agent.id,
                model=selected_model,
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

    async def answer_stream(
        self,
        question: str,
        *,
        agent_id: str | None = None,
        model: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yields dict events: status, sources, token, done."""
        agent, selected_model, hits, prompt = self._prepare(
            question, agent_id=agent_id, model=model
        )

        yield {
            "type": "status",
            "stage": "retrieved",
            "agentId": agent.id,
            "model": selected_model,
            "sourceCount": len(hits),
        }

        if not hits:
            fallback = "I could not find sufficient information in the provided documents."
            yield {"type": "sources", "sources": []}
            yield {"type": "token", "content": fallback}
            yield {
                "type": "done",
                "answer": fallback,
                "sources": [],
                "agentId": agent.id,
                "model": selected_model,
                "_hits": [],
            }
            return

        source_payload = [
            {
                "documentId": hit.document_id or None,
                "chunkId": hit.chunk_id or None,
                "fileName": hit.file_name,
                "pageNumber": hit.page_number,
                "relevanceScore": hit.score,
            }
            for hit in hits
        ]
        yield {"type": "sources", "sources": source_payload}
        yield {"type": "status", "stage": "generating"}

        parts: list[str] = []
        async for token in self.llm_service.generate_stream(
            prompt=prompt,
            system=agent.system_prompt,
            model=selected_model,
        ):
            parts.append(token)
            yield {"type": "token", "content": token}

        answer = "".join(parts).strip()
        if not answer:
            answer = "I could not find sufficient information in the provided documents."
            yield {"type": "token", "content": answer}

        yield {
            "type": "done",
            "answer": answer,
            "sources": source_payload,
            "agentId": agent.id,
            "model": selected_model,
            "_hits": hits,
        }
