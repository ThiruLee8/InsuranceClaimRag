from __future__ import annotations

from dataclasses import dataclass, field

from shared.configuration import get_settings
from shared.logging_utils import get_logger
from services.bm25_index_service import Bm25IndexService
from services.rerank_service import RerankService
from services.vector_store_service import VectorSearchHit, VectorStoreService

logger = get_logger(__name__)


@dataclass
class HybridSearchResult:
    hits: list[VectorSearchHit]
    search_mode: str
    score_semantic: dict[str, float] = field(default_factory=dict)
    score_keyword: dict[str, float] = field(default_factory=dict)


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    *,
    k: int = 60,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridSearchService:
    def __init__(
        self,
        *,
        vector_store: VectorStoreService | None = None,
        bm25: Bm25IndexService | None = None,
    ) -> None:
        self.settings = get_settings()
        self.vector_store = vector_store or VectorStoreService()
        self.bm25 = bm25 or Bm25IndexService.get()

    def _ensure_bm25(self) -> None:
        self.bm25.ensure_ready()
        try:
            chroma_count = int(self.vector_store.collection.count() or 0)
        except Exception:  # noqa: BLE001
            chroma_count = -1
        if self.bm25.document_count() == 0 or (
            chroma_count >= 0 and self.bm25.document_count() != chroma_count
        ):
            self.bm25.load_from_chroma(self.vector_store.collection)

    def search(
        self,
        *,
        question: str,
        query_embedding: list[float],
        top_k: int,
        similarity_threshold: float,
        search_mode: str | None = None,
        rerank: bool | None = None,
    ) -> HybridSearchResult:
        mode = (search_mode or self.settings.search_mode or "hybrid").strip().lower()
        do_rerank = self.settings.enable_rerank if rerank is None else bool(rerank)
        pool = max(top_k * max(1, self.settings.candidate_multiplier), top_k)
        if do_rerank:
            pool = max(pool, self.settings.rerank_candidates)

        semantic_hits: list[VectorSearchHit] = []
        keyword_hits: list[VectorSearchHit] = []
        score_semantic: dict[str, float] = {}
        score_keyword: dict[str, float] = {}

        if mode in {"semantic", "hybrid"}:
            # Pull a wider pool; apply threshold after fusion/rerank so keyword can rescue.
            threshold = 0.0 if mode == "hybrid" else similarity_threshold
            semantic_hits = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=pool,
                similarity_threshold=threshold,
            )
            for hit in semantic_hits:
                score_semantic[hit.chunk_id] = hit.score

        if mode in {"keyword", "hybrid"}:
            self._ensure_bm25()
            for doc, raw_score in self.bm25.search(question, top_k=pool):
                # Normalize BM25 roughly into 0..1 via rank-friendly raw score storage.
                score_keyword[doc.chunk_id] = raw_score
                keyword_hits.append(
                    VectorSearchHit(
                        chunk_id=doc.chunk_id,
                        document_id=doc.document_id,
                        file_name=doc.file_name,
                        page_number=doc.page_number,
                        chunk_index=doc.chunk_index,
                        content=doc.content,
                        score=raw_score,
                    )
                )

        by_id: dict[str, VectorSearchHit] = {}
        for hit in semantic_hits + keyword_hits:
            by_id[hit.chunk_id] = hit

        if mode == "semantic":
            ordered_ids = [h.chunk_id for h in semantic_hits]
            fused_scores = {cid: score_semantic.get(cid, 0.0) for cid in ordered_ids}
        elif mode == "keyword":
            ordered_ids = [h.chunk_id for h in keyword_hits]
            # Rank-normalize keyword scores for response.
            max_kw = max((score_keyword.get(cid, 0.0) for cid in ordered_ids), default=1.0) or 1.0
            fused_scores = {cid: score_keyword.get(cid, 0.0) / max_kw for cid in ordered_ids}
        else:
            ranked_lists = [
                [h.chunk_id for h in semantic_hits],
                [h.chunk_id for h in keyword_hits],
            ]
            fused = reciprocal_rank_fusion(ranked_lists, k=self.settings.rrf_k)
            ordered_ids = [cid for cid, _ in fused]
            fused_scores = {cid: score for cid, score in fused}

        candidates: list[VectorSearchHit] = []
        for chunk_id in ordered_ids:
            hit = by_id.get(chunk_id)
            if not hit:
                continue
            candidates.append(
                VectorSearchHit(
                    chunk_id=hit.chunk_id,
                    document_id=hit.document_id,
                    file_name=hit.file_name,
                    page_number=hit.page_number,
                    chunk_index=hit.chunk_index,
                    content=hit.content,
                    score=float(fused_scores.get(chunk_id, hit.score)),
                )
            )

        # For pure semantic, keep original threshold filter.
        if mode == "semantic":
            candidates = [h for h in candidates if h.score >= similarity_threshold]

        if do_rerank and candidates:
            try:
                rerank_n = min(len(candidates), max(top_k, self.settings.rerank_candidates))
                subset = candidates[:rerank_n]
                ranked = RerankService.get().rerank(
                    question,
                    texts=[h.content for h in subset],
                    top_k=top_k,
                )
                reranked: list[VectorSearchHit] = []
                for idx, score in ranked:
                    base = subset[idx]
                    reranked.append(
                        VectorSearchHit(
                            chunk_id=base.chunk_id,
                            document_id=base.document_id,
                            file_name=base.file_name,
                            page_number=base.page_number,
                            chunk_index=base.chunk_index,
                            content=base.content,
                            score=float(score),
                        )
                    )
                candidates = reranked
                mode = f"{mode}+rerank"
            except Exception as exc:  # noqa: BLE001
                logger.warning("rerank_failed", error=str(exc))
                candidates = candidates[:top_k]
        else:
            candidates = candidates[:top_k]

        logger.info(
            "hybrid_search_complete",
            mode=mode,
            semantic=len(semantic_hits),
            keyword=len(keyword_hits),
            returned=len(candidates),
            rerank=do_rerank,
        )
        return HybridSearchResult(
            hits=candidates,
            search_mode=mode,
            score_semantic=score_semantic,
            score_keyword=score_keyword,
        )
