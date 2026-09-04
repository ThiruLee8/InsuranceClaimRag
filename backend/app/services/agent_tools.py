from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.repositories import DocumentRepository
from app.services.agent_memory import AgentMemoryStore
from app.services.vector_gateway_client import VectorGatewayClient, VectorSearchHit

logger = get_logger(__name__)


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: str


@dataclass
class ToolContext:
    session_id: str
    task: str
    document_repo: DocumentRepository | None = None
    vector_gateway: VectorGatewayClient | None = None
    memory: AgentMemoryStore | None = None
    sources: list[VectorSearchHit] = field(default_factory=list)


TOOLS: list[ToolSpec] = [
    ToolSpec(
        name="list_documents",
        description=(
            "List uploaded claim documents (file name, status, page count). "
            "Use once at the start if you do not already know which files exist."
        ),
        parameters="{}",
    ),
    ToolSpec(
        name="search_documents",
        description=(
            "Search indexed claim documents for ONE focused fact "
            "(cause of loss, water exclusion, deductible, repair cost, settlement). "
            "Do not paste the whole user task as the query."
        ),
        parameters='{"query": "short search query"}',
    ),
    ToolSpec(
        name="search_memory",
        description=(
            "Recall summarised findings from earlier runs in this session. "
            "Use for follow-up questions or to avoid re-searching known facts."
        ),
        parameters='{"query": "what to recall"}',
    ),
    ToolSpec(
        name="save_memory",
        description=(
            "Store a durable fact from this investigation for later tasks in the same session."
        ),
        parameters='{"fact": "one sentence fact"}',
    ),
    ToolSpec(
        name="finish",
        description=(
            "End the loop and answer the user. Cite document names and pages when possible. "
            "If the documents are insufficient, say so clearly."
        ),
        parameters='{"answer": "final answer"}',
    ),
]

TOOL_NAMES = {spec.name for spec in TOOLS}


def tools_prompt_block() -> str:
    lines = []
    for spec in TOOLS:
        lines.append(f"- {spec.name}")
        lines.append(f"  description: {spec.description}")
        lines.append(f"  action_input: {spec.parameters}")
    return "\n".join(lines)


def _truncate(text: str, limit: int = 280) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rsplit(" ", 1)[0] + "…"


def format_hits(hits: list[VectorSearchHit], *, limit: int = 4) -> str:
    if not hits:
        return "No matching document chunks were found."
    lines = [f"Found {len(hits)} chunks (showing {min(limit, len(hits))}):"]
    for i, hit in enumerate(hits[:limit], start=1):
        page = hit.page_number if hit.page_number is not None else "n/a"
        lines.append(
            f"{i}. {hit.file_name} p.{page} ({hit.score:.2f}): {_truncate(hit.content)}"
        )
    return "\n".join(lines)


def _as_query(action_input: Any, key: str = "query") -> str:
    if isinstance(action_input, dict):
        value = action_input.get(key) or action_input.get("q") or action_input.get("fact")
        return str(value or "").strip()
    return str(action_input or "").strip()


def execute_tool(name: str, action_input: Any, ctx: ToolContext) -> str:
    if name == "list_documents":
        return _list_documents(ctx)
    if name == "search_documents":
        return _search_documents(_as_query(action_input), ctx)
    if name == "search_memory":
        return _search_memory(_as_query(action_input), ctx)
    if name == "save_memory":
        return _save_memory(_as_query(action_input, "fact"), ctx)
    if name == "finish":
        return "finish"
    return f"Unknown tool '{name}'. Valid tools: {', '.join(sorted(TOOL_NAMES))}."


def _list_documents(ctx: ToolContext) -> str:
    if ctx.document_repo is None:
        return "Document list is unavailable in this run."
    try:
        items, total = ctx.document_repo.list(limit=50)
    except Exception as exc:  # noqa: BLE001
        logger.warning("agent_list_documents_failed", error=str(exc))
        return f"Could not list documents: {exc}"
    if not items:
        return "No documents are uploaded yet. Ask the user to upload claim files first."
    lines = [f"{total} document(s):"]
    for doc in items:
        status = doc.Status.value if hasattr(doc.Status, "value") else str(doc.Status)
        pages = doc.PageCount if doc.PageCount is not None else "?"
        lines.append(f"- {doc.OriginalFileName} [{status}, {pages} pages]")
    return "\n".join(lines)


def _search_documents(query: str, ctx: ToolContext) -> str:
    if not query:
        return "search_documents requires a non-empty query."
    if ctx.vector_gateway is None:
        return "Document search is unavailable in this run."
    settings = get_settings()
    try:
        result = ctx.vector_gateway.search(
            question=query,
            top_k=settings.top_k,
            similarity_threshold=settings.similarity_threshold,
            search_mode=settings.search_mode,
            rerank=settings.enable_rerank,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("agent_search_failed", error=str(exc))
        return f"Search failed: {exc}"
    for hit in result.hits:
        ctx.sources.append(hit)
    mode = result.search_mode or settings.search_mode
    return f"Query: {query}\nSearch mode: {mode}\n{format_hits(result.hits)}"


def _search_memory(query: str, ctx: ToolContext) -> str:
    if ctx.memory is None:
        return "Memory is unavailable in this run."
    q = query or ctx.task
    hits = ctx.memory.recall(q, session_id=ctx.session_id, limit=5)
    if not hits:
        return "No prior memories matched this query for this session."
    lines = [f"Recalled {len(hits)} memor(ies):"]
    for entry, score in hits:
        fact_bit = f" Facts: {'; '.join(entry.facts)}" if entry.facts else ""
        lines.append(f"- ({score:.2f}) {entry.summary}{fact_bit}")
    return "\n".join(lines)


def _save_memory(fact: str, ctx: ToolContext) -> str:
    if not fact:
        return "save_memory requires a non-empty fact."
    if ctx.memory is None:
        return "Memory is unavailable in this run."
    entry = ctx.memory.add_fact(session_id=ctx.session_id, fact=fact, task=ctx.task)
    return f"Saved memory {entry.id}: {fact}"
