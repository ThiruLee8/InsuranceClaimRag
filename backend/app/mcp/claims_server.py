"""Claims MCP server — tools, one resource, one prompt. No LLM lives here."""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any

from app.core.logging import get_logger
from app.services.agent_tools import ToolContext, execute_tool

logger = get_logger(__name__)

_TOOL_CTX: ContextVar[ToolContext | None] = ContextVar("mcp_tool_ctx", default=None)

SERVER_NAME = "claims-docs"
SERVER_INSTRUCTIONS = (
    "Insurance claim document tools for ClaimIntel. "
    "This server does not run an AI model. The host that calls these tools "
    "is responsible for planning and answering."
)

try:
    from fastmcp.exceptions import ToolError
except ImportError:  # pragma: no cover - older FastMCP
    class ToolError(Exception):  # type: ignore[no-redef]
        """Recoverable tool failure (invalid args, missing data)."""


def set_tool_context(ctx: ToolContext) -> Token[ToolContext | None]:
    return _TOOL_CTX.set(ctx)


def reset_tool_context(token: Token[ToolContext | None]) -> None:
    _TOOL_CTX.reset(token)


def current_tool_context() -> ToolContext:
    ctx = _TOOL_CTX.get()
    if ctx is not None:
        return ctx
    return _remote_context()


def _remote_context() -> ToolContext:
    """Used when another agent calls this server over HTTP/stdio (no host ctx)."""
    from app.db.database import SessionLocal
    from app.db.repositories import DocumentRepository
    from app.services.agent_memory import AgentMemoryStore
    from app.services.vector_gateway_client import VectorGatewayClient

    db = SessionLocal()
    return ToolContext(
        session_id="mcp-remote",
        task="remote MCP call",
        document_repo=DocumentRepository(db),
        vector_gateway=VectorGatewayClient(),
        memory=AgentMemoryStore(),
    )


def _run(name: str, arguments: dict[str, Any] | None = None) -> str:
    args = dict(arguments or {})
    session_id = str(args.pop("session_id", "") or "")
    owned_db = None
    ctx = _TOOL_CTX.get()
    if ctx is None:
        ctx = _remote_context()
        owned_db = getattr(ctx.document_repo, "db", None)
    if session_id:
        ctx.session_id = session_id
    try:
        result = execute_tool(name, args, ctx)
    except Exception as exc:  # noqa: BLE001
        logger.warning("mcp_tool_failed", tool=name, error=str(exc))
        raise ToolError(f"{name} failed: {exc}") from exc
    finally:
        if owned_db is not None:
            try:
                owned_db.close()
            except Exception:  # noqa: BLE001
                pass
    if isinstance(result, str) and result.startswith("Unknown tool"):
        raise ToolError(result)
    return result


def create_claims_server():
    """Build a FastMCP server exposing real claim-file capabilities.

    Add a new @mcp.tool here and the agent will discover it on the next
    handshake — no change to the agent loop is required.
    """
    from fastmcp import FastMCP

    mcp = FastMCP(SERVER_NAME)

    @mcp.tool()
    def list_documents() -> str:
        """List uploaded claim documents (file name, status, page count)."""
        return _run("list_documents", {})

    @mcp.tool()
    def search_documents(query: str) -> str:
        """Search indexed claim documents for one focused fact (cause, exclusion, deductible, settlement)."""
        if not (query or "").strip():
            raise ToolError("search_documents requires a non-empty query.")
        return _run("search_documents", {"query": query})

    @mcp.tool()
    def search_memory(query: str, session_id: str = "") -> str:
        """Recall summarised findings from earlier runs in this session."""
        if not (query or "").strip():
            raise ToolError("search_memory requires a non-empty query.")
        return _run("search_memory", {"query": query, "session_id": session_id})

    @mcp.tool()
    def save_memory(fact: str, session_id: str = "") -> str:
        """Store a durable claim fact for later tasks in the same session."""
        if not (fact or "").strip():
            raise ToolError("save_memory requires a non-empty fact.")
        return _run("save_memory", {"fact": fact, "session_id": session_id})

    @mcp.tool()
    def document_status(file_name: str = "") -> str:
        """Return processing status for uploaded claim files. Optional file_name focuses on one document."""
        return _run("document_status", {"file_name": file_name})

    @mcp.resource("claim://documents")
    def documents_catalog() -> str:
        """Catalog of uploaded claim files (status and page counts)."""
        return _run("list_documents", {})

    @mcp.prompt()
    def investigate_claim(focus: str = "cause of loss") -> str:
        """Prompt template another host can use to investigate a claim file."""
        return (
            "You are investigating an insurance claim using MCP tools. "
            f"Focus on: {focus}. "
            "Call list_documents or document_status first, then search_documents "
            "for one fact at a time. Do not invent document facts. "
            "The model runs on your host, not on this tool server."
        )

    return mcp


_SERVER = None


def get_claims_server():
    global _SERVER
    if _SERVER is None:
        _SERVER = create_claims_server()
    return _SERVER
