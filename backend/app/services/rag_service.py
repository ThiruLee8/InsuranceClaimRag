from __future__ import annotations

import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.agents import get_agent
from app.services.llm_service import LLMProvider, OllamaLLMService
from app.services.vector_gateway_client import VectorGatewayClient, VectorSearchHit

logger = get_logger(__name__)

_REWRITE_SYSTEM = (
    "You rewrite insurance claim questions into concise search queries. "
    "Preserve claim numbers, policy IDs, dollar amounts, and proper nouns exactly. "
    "Output only the rewritten search query, nothing else."
)

# Skip LLM rewrite when the question already looks precise / short.
_ID_PATTERN = re.compile(
    r"\b(?:CLM|HO|EST|POL)[-_]?\d[\w-]*\b|\bUSD\s?\d|\b\d{1,3}(?:,\d{3})+\b",
    re.IGNORECASE,
)


@dataclass
class RAGResult:
    answer: str
    sources: list[VectorSearchHit]
    agent_id: str
    model: str
    original_question: str = ""
    search_query: str = ""
    search_mode: str | None = None


@dataclass
class RetrieveResult:
    hits: list[VectorSearchHit]
    original_question: str
    search_query: str
    search_mode: str | None = None


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

    def _should_rewrite(self, question: str) -> bool:
        q = question.strip()
        if len(q) < 12:
            return False
        # Already contains an exact ID / code — keep as-is for keyword hit rate.
        if _ID_PATTERN.search(q) and len(q.split()) <= 8:
            return False
        # Very short factual questions are usually fine raw.
        if len(q.split()) <= 6 and "?" in q:
            return False
        return True

    async def _rewrite_question(self, question: str, *, model: str) -> str:
        if not self._should_rewrite(question):
            return question.strip()
        try:
            rewritten = await self.llm_service.generate(
                prompt=(
                    f"Original question:\n{question}\n\n"
                    "Rewrite as a single search query for insurance claim documents."
                ),
                system=_REWRITE_SYSTEM,
                model=model,
            )
            cleaned = (rewritten or "").strip().strip('"').strip("'")
            # Guard against verbose model output.
            if not cleaned or len(cleaned) > 500:
                return question.strip()
            first_line = cleaned.splitlines()[0].strip()
            return first_line or question.strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("query_rewrite_failed", error=str(exc))
            return question.strip()

    def _hits_to_source_payload(
        self, hits: list[VectorSearchHit], *, debug: bool
    ) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for hit in hits:
            item: dict[str, Any] = {
                "documentId": hit.document_id or None,
                "chunkId": hit.chunk_id or None,
                "fileName": hit.file_name,
                "pageNumber": hit.page_number,
                "relevanceScore": hit.score,
            }
            if debug:
                item["content"] = hit.content
            payload.append(item)
        return payload

    async def retrieve(
        self,
        question: str,
        *,
        model: str | None = None,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        search_mode: str | None = None,
        rewrite: bool = True,
        rerank: bool | None = None,
    ) -> RetrieveResult:
        selected_model = (model or self.settings.ollama_model).strip() or self.settings.ollama_model
        original = question.strip()
        search_query = (
            await self._rewrite_question(original, model=selected_model)
            if rewrite
            else original
        )
        result = self.vector_gateway.search(
            question=search_query,
            top_k=top_k or self.settings.top_k,
            similarity_threshold=(
                similarity_threshold
                if similarity_threshold is not None
                else self.settings.similarity_threshold
            ),
            search_mode=search_mode or self.settings.search_mode,
            rerank=self.settings.enable_rerank if rerank is None else rerank,
        )
        logger.info(
            "rag_retrieve_complete",
            question_len=len(original),
            search_query_len=len(search_query),
            hits=len(result.hits),
            search_mode=result.search_mode,
        )
        return RetrieveResult(
            hits=result.hits,
            original_question=original,
            search_query=search_query,
            search_mode=result.search_mode,
        )

    async def _prepare(
        self,
        question: str,
        *,
        agent_id: str | None = None,
        model: str | None = None,
        rewrite: bool = True,
    ) -> tuple[Any, str, list[VectorSearchHit], str, str, str, str | None]:
        agent = get_agent(agent_id)
        selected_model = (model or self.settings.ollama_model).strip() or self.settings.ollama_model
        original = question.strip()
        search_query = (
            await self._rewrite_question(original, model=selected_model)
            if rewrite
            else original
        )

        search_result = self.vector_gateway.search(
            question=search_query,
            top_k=self.settings.top_k,
            similarity_threshold=self.settings.similarity_threshold,
            search_mode=self.settings.search_mode,
            rerank=self.settings.enable_rerank,
        )
        hits = search_result.hits
        search_mode = search_result.search_mode
        logger.info(
            "rag_search_complete",
            question_len=len(original),
            search_query_len=len(search_query),
            hits=len(hits),
            agent_id=agent.id,
            model=selected_model,
            search_mode=search_mode,
            rewritten=search_query != original,
        )
        prompt = ""
        if hits:
            prompt = (
                f"Document context:\n{self._build_context(hits)}\n\n"
                f"User question: {original}\n\n"
                "Provide a clear answer based only on the context above."
            )
        return agent, selected_model, hits, prompt, original, search_query, search_mode

    async def answer(
        self,
        question: str,
        *,
        agent_id: str | None = None,
        model: str | None = None,
        debug: bool = False,
    ) -> RAGResult:
        agent, selected_model, hits, prompt, original, search_query, search_mode = (
            await self._prepare(question, agent_id=agent_id, model=model)
        )

        if not hits:
            return RAGResult(
                answer="I could not find sufficient information in the provided documents.",
                sources=[],
                agent_id=agent.id,
                model=selected_model,
                original_question=original,
                search_query=search_query,
                search_mode=search_mode,
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
            original_question=original,
            search_query=search_query,
            search_mode=search_mode,
        )

    async def answer_stream(
        self,
        question: str,
        *,
        agent_id: str | None = None,
        model: str | None = None,
        debug: bool = False,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yields dict events: status, sources, token, done."""
        agent, selected_model, hits, prompt, original, search_query, search_mode = (
            await self._prepare(question, agent_id=agent_id, model=model)
        )

        yield {
            "type": "status",
            "stage": "retrieved",
            "agentId": agent.id,
            "model": selected_model,
            "sourceCount": len(hits),
            "originalQuestion": original,
            "searchQuery": search_query,
            "searchMode": search_mode,
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
                "originalQuestion": original,
                "searchQuery": search_query,
                "searchMode": search_mode,
                "_hits": [],
            }
            return

        source_payload = self._hits_to_source_payload(hits, debug=debug)
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
            "originalQuestion": original,
            "searchQuery": search_query,
            "searchMode": search_mode,
            "_hits": hits,
        }
