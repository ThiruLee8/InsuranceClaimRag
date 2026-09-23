"""MCP: discovery, handshake, trust checks, second tool without agent code changes."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

pytest.importorskip("fastmcp")

from app.mcp.claims_server import create_claims_server
from app.mcp.gateway import McpToolGateway
from app.mcp.http_app import BearerAuthASGI
from app.services.agent_loop import ClaimAgentRunner
from app.services.agent_memory import AgentMemoryStore
from app.services.agent_tools import ToolContext, execute_tool
from app.services.vector_gateway_client import VectorSearchHit, VectorSearchResult


class ScriptedLLM:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls = 0
        self.prompts: list[str] = []

    async def generate(self, *, prompt: str, system: str | None = None, model: str | None = None) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        if not self.replies:
            return json.dumps(
                {
                    "thought": "fallback finish",
                    "action": "finish",
                    "action_input": {"answer": "fallback"},
                }
            )
        return self.replies.pop(0)


def _hit(name: str, content: str, score: float = 0.9, page: int = 1) -> VectorSearchHit:
    return VectorSearchHit(
        chunk_id=str(uuid4()),
        document_id=str(uuid4()),
        file_name=name,
        page_number=page,
        chunk_index=0,
        content=content,
        score=score,
    )


def _gateway(hits: list[VectorSearchHit] | None = None) -> MagicMock:
    gw = MagicMock()
    gw.search.return_value = VectorSearchResult(
        hits=hits or [_hit("claim-investigation-report.pdf", "Cause of loss was a burst pipe.")],
        search_mode="hybrid",
    )
    return gw


def _repo(names: list[str] | None = None) -> MagicMock:
    repo = MagicMock()
    docs = []
    for name in names or ["policy-schedule.pdf", "claim-investigation-report.pdf"]:
        doc = MagicMock()
        doc.OriginalFileName = name
        doc.Status = MagicMock(value="Processed")
        doc.PageCount = 4
        doc.ProcessingError = None
        docs.append(doc)
    repo.list.return_value = (docs, len(docs))
    return repo


@pytest.mark.asyncio
async def test_mcp_discovers_tools_instead_of_hardcoding():
    gw = McpToolGateway(server=create_claims_server())
    await gw.connect()
    try:
        names = gw.trusted_names()
        assert "list_documents" in names
        assert "search_documents" in names
        assert "document_status" in names
        assert "finish" not in names
        methods = [m.method for m in gw.messages]
        assert "initialize" in methods
        assert "notifications/initialized" in methods
        assert "tools/list" in methods
        listed = next(m for m in gw.messages if m.method == "tools/list" and m.direction == "response")
        tool_names = {t["name"] for t in listed.payload["tools"]}
        assert "document_status" in tool_names
        assert gw.resources
        assert "investigate_claim" in gw.prompts
    finally:
        await gw.close()


@pytest.mark.asyncio
async def test_second_mcp_tool_needs_no_agent_code_change(tmp_path: Path):
    server = create_claims_server()

    @server.tool()
    def claim_ref() -> str:
        """Return the demo claim reference number."""
        return "CLM-1042"

    mcp = McpToolGateway(server=server)
    await mcp.connect()
    try:
        assert "claim_ref" in mcp.trusted_names()
        llm = ScriptedLLM(
            [
                json.dumps(
                    {
                        "thought": "use the new MCP tool",
                        "action": "claim_ref",
                        "action_input": {},
                    }
                ),
                json.dumps(
                    {
                        "thought": "done",
                        "action": "finish",
                        "action_input": {"answer": "Claim reference is CLM-1042."},
                    }
                ),
            ]
        )
        runner = ClaimAgentRunner(
            llm=llm,
            vector_gateway=_gateway(),
            document_repo=_repo(),
            memory=AgentMemoryStore(tmp_path / "mem.json"),
            mcp=mcp,
        )
        result = await runner.run_agent("What is the claim reference?", persist_memory=False)
        assert [s.action for s in result.steps] == ["claim_ref", "finish"]
        assert "CLM-1042" in result.steps[0].observation
        assert "claim_ref" in llm.prompts[0]
        assert "CLM-1042" in result.answer
    finally:
        await mcp.close()


@pytest.mark.asyncio
async def test_dangerous_remote_tool_is_blocked():
    from fastmcp import FastMCP

    sketchy = FastMCP("not-ours")

    @sketchy.tool()
    def shell(command: str) -> str:
        """Run a shell command."""
        return command

    gw = McpToolGateway(server=sketchy, trust_own_server=True)
    await gw.connect()
    try:
        assert "shell" not in gw.trusted_names()
        assert any(t.name == "shell" and not t.trusted for t in gw.tools)
        ctx = ToolContext(session_id="s", task="t")
        observation = await gw.call_tool("shell", {"command": "whoami"}, ctx)
        assert "not available" in observation.lower()
    finally:
        await gw.close()


@pytest.mark.asyncio
async def test_mcp_tool_error_is_recoverable():
    gw = McpToolGateway(server=create_claims_server())
    await gw.connect()
    try:
        ctx = ToolContext(session_id="s", task="cause?", document_repo=_repo(), vector_gateway=_gateway())
        text = await gw.call_tool("search_documents", {"query": ""}, ctx)
        assert "empty" in text.lower() or "error" in text.lower() or "non-empty" in text.lower()
        listed = await gw.call_tool("list_documents", {}, ctx)
        assert "policy-schedule.pdf" in listed
    finally:
        await gw.close()


def test_document_status_local_tool():
    ctx = ToolContext(session_id="s", task="status", document_repo=_repo())
    text = execute_tool("document_status", {}, ctx)
    assert "policy-schedule.pdf" in text
    assert "Processed" in text


@pytest.mark.asyncio
async def test_bearer_auth_rejects_missing_token():
    allowed = {"called": False}

    async def inner(scope, receive, send):
        allowed["called"] = True
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/plain")]})
        await send({"type": "http.response.body", "body": b"ok"})

    wrapped = BearerAuthASGI(inner, "secret-token")
    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    await wrapped(
        {"type": "http", "headers": [(b"authorization", b"Bearer wrong")]},
        receive,
        send,
    )
    assert allowed["called"] is False
    assert sent[0]["status"] == 401

    sent.clear()
    await wrapped(
        {"type": "http", "headers": [(b"authorization", b"Bearer secret-token")]},
        receive,
        send,
    )
    assert allowed["called"] is True
    assert sent[0]["status"] == 200
