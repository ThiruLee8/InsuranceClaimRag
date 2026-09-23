"""MCP client used by the host: discover tools, inspect them, then call.

JSON-RPC messages are recorded so MCP is visible instead of a black box.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.mcp.claims_server import (
    SERVER_NAME,
    get_claims_server,
    reset_tool_context,
    set_tool_context,
)
from app.services.agent_tools import LOCAL_LOOP_TOOLS, ToolContext, ToolSpec, execute_tool, tools_prompt_block

logger = get_logger(__name__)

WHERE_AI_RUNS = (
    "On the host — this ClaimIntel app, which calls Ollama. "
    "The model never runs on the MCP server."
)
WHERE_AI_DOES_NOT_RUN = (
    "The MCP server only offers tools, resources, and prompts. "
    "It has no idea which AI called it."
)

DANGEROUS_NAME_BITS = (
    "shell",
    "bash",
    "cmd",
    "powershell",
    "exec",
    "eval",
    "fetch",
    "http",
    "email",
    "write_file",
    "delete_all",
    "rm_rf",
)


@dataclass
class DiscoveredTool:
    name: str
    description: str
    parameters: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    server: str = SERVER_NAME
    trusted: bool = True
    trust_reason: str = "own server"

    def to_spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, parameters=self.parameters)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "server": self.server,
            "trusted": self.trusted,
            "trustReason": self.trust_reason,
            "source": "mcp",
        }


@dataclass
class McpMessage:
    direction: str
    method: str
    payload: Any
    rpc_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {"jsonrpc": "2.0"}
        if self.rpc_id is not None:
            body["id"] = self.rpc_id
        if self.direction == "request":
            body["method"] = self.method
            if self.payload is not None:
                body["params"] = self.payload
        elif self.direction == "notification":
            body["method"] = self.method
            if self.payload is not None:
                body["params"] = self.payload
        else:
            body["result"] = self.payload
        return {
            "direction": self.direction,
            "method": self.method,
            "message": body,
        }


class McpToolGateway:
    """Host-side MCP client: handshake, tools/list, tools/call, trust check."""

    def __init__(
        self,
        *,
        server: Any | None = None,
        trust_own_server: bool = True,
        allowed_tools: set[str] | None = None,
        deny_tools: set[str] | None = None,
    ) -> None:
        settings = get_settings()
        self.server = server if server is not None else get_claims_server()
        self.trust_own_server = trust_own_server
        self.allowed_tools = allowed_tools if allowed_tools is not None else settings.mcp_allowed_tool_set
        self.deny_tools = deny_tools if deny_tools is not None else settings.mcp_deny_tool_set
        self.server_name = getattr(self.server, "name", None) or SERVER_NAME
        self.transport = "memory"
        self.messages: list[McpMessage] = []
        self.tools: list[DiscoveredTool] = []
        self.resources: list[str] = []
        self.prompts: list[str] = []
        self._client: Any | None = None
        self._cm: Any | None = None
        self._rpc_id = 0
        self.connected = False
        self.last_error: str | None = None

    def _next_id(self) -> int:
        self._rpc_id += 1
        return self._rpc_id

    def _record(self, direction: str, method: str, payload: Any, rpc_id: int | None = None) -> None:
        self.messages.append(McpMessage(direction=direction, method=method, payload=payload, rpc_id=rpc_id))

    def inspect_tool(self, name: str, description: str, schema: dict[str, Any] | None = None) -> tuple[bool, str]:
        lowered = (name or "").strip().lower()
        if lowered in self.deny_tools or any(bit in lowered for bit in DANGEROUS_NAME_BITS):
            return False, "blocked: dangerous tool name — inspect before trusting someone else's server"
        if self.allowed_tools and name not in self.allowed_tools:
            return False, "blocked: not on the host allow-list"
        props = (schema or {}).get("properties") or {}
        for key in props:
            if str(key).lower() in {"command", "sql", "url", "headers"}:
                return False, f"blocked: parameter '{key}' looks unsafe without a review"
        if self.trust_own_server:
            return True, "trusted: own claims-docs server after name/schema check"
        if not self.allowed_tools:
            return False, "blocked: remote tool — add it to MCP_ALLOWED_TOOLS after you read the schema"
        return True, "trusted: host allow-list"

    async def connect(self) -> None:
        if self._client is not None:
            return
        from fastmcp import Client

        self.messages = []
        self._rpc_id = 0
        init_id = self._next_id()
        self._record(
            "request",
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "claimintel-host", "version": "1.0.0"},
            },
            init_id,
        )
        self._cm = Client(self.server)
        self._client = await self._cm.__aenter__()
        info = {
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": self.server_name, "version": "1.0.0"},
            "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
        }
        self._record("response", "initialize", info, init_id)
        self._record("notification", "notifications/initialized", None)
        await self.discover()
        self.connected = True
        self.last_error = None
        logger.info(
            "mcp_connected",
            server=self.server_name,
            tools=len(self.trusted_tools),
            transport=self.transport,
        )

    async def close(self) -> None:
        if self._cm is None:
            return
        try:
            await self._cm.__aexit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass
        self._cm = None
        self._client = None
        self.connected = False

    async def ensure_connected(self) -> None:
        if self._client is None:
            await self.connect()

    async def discover(self) -> list[DiscoveredTool]:
        if self._client is None:
            await self.connect()
            return self.tools
        list_id = self._next_id()
        self._record("request", "tools/list", {}, list_id)
        raw_tools = await self._client.list_tools()
        discovered: list[DiscoveredTool] = []
        listed: list[dict[str, Any]] = []
        for tool in raw_tools:
            name = getattr(tool, "name", "")
            description = getattr(tool, "description", "") or ""
            schema_raw = getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", None) or {}
            if hasattr(schema_raw, "model_dump"):
                schema = schema_raw.model_dump()
            elif isinstance(schema_raw, dict):
                schema = schema_raw
            else:
                try:
                    schema = dict(schema_raw)
                except Exception:  # noqa: BLE001
                    schema = {}
            params = _schema_to_params(schema)
            trusted, reason = self.inspect_tool(name, description, schema)
            discovered.append(
                DiscoveredTool(
                    name=name,
                    description=description,
                    parameters=params,
                    input_schema=schema,
                    server=self.server_name,
                    trusted=trusted,
                    trust_reason=reason,
                )
            )
            listed.append({"name": name, "description": description, "inputSchema": schema})
        self._record("response", "tools/list", {"tools": listed}, list_id)
        self.tools = discovered

        try:
            res_id = self._next_id()
            self._record("request", "resources/list", {}, res_id)
            resources = await self._client.list_resources()
            self.resources = [str(getattr(r, "uri", r)) for r in resources]
            self._record(
                "response",
                "resources/list",
                {"resources": [{"uri": u} for u in self.resources]},
                res_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.info("mcp_resources_list_skipped", error=str(exc))

        try:
            p_id = self._next_id()
            self._record("request", "prompts/list", {}, p_id)
            prompts = await self._client.list_prompts()
            self.prompts = [str(getattr(p, "name", p)) for p in prompts]
            self._record(
                "response",
                "prompts/list",
                {"prompts": [{"name": n} for n in self.prompts]},
                p_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.info("mcp_prompts_list_skipped", error=str(exc))

        return self.tools

    @property
    def trusted_tools(self) -> list[DiscoveredTool]:
        return [t for t in self.tools if t.trusted]

    @property
    def blocked_tools(self) -> list[DiscoveredTool]:
        return [t for t in self.tools if not t.trusted]

    def trusted_names(self) -> set[str]:
        return {t.name for t in self.trusted_tools}

    def tool_schemas(self) -> dict[str, dict[str, Any]]:
        return {t.name: t.input_schema for t in self.trusted_tools}

    def prompt_specs(self) -> list[ToolSpec]:
        specs = [t.to_spec() for t in self.trusted_tools]
        specs.append(
            ToolSpec(
                name="finish",
                description=(
                    "End the loop and answer the user. Cite document names and pages when possible. "
                    "If the documents are insufficient, say so clearly. This action stays on the host."
                ),
                parameters='{"answer": "final answer"}',
            )
        )
        return specs

    def tools_prompt_block(self) -> str:
        return tools_prompt_block(self.prompt_specs())

    async def call_tool(self, name: str, arguments: Any, ctx: ToolContext) -> str:
        """Call a discovered MCP tool. Failures become observations, not crashes."""
        args = arguments if isinstance(arguments, dict) else {}
        if name in LOCAL_LOOP_TOOLS:
            return "finish"
        trusted = next((t for t in self.trusted_tools if t.name == name), None)
        if trusted is None:
            return (
                f"Tool '{name}' is not available. "
                f"Discovered (trusted): {', '.join(sorted(self.trusted_names())) or '(none)'}."
            )
        if self._client is None:
            return await self._local_fallback(name, args, ctx)

        call_id = self._next_id()
        self._record("request", "tools/call", {"name": name, "arguments": args}, call_id)
        token = set_tool_context(ctx)
        try:
            result = await self._client.call_tool(name, args)
            text = _result_text(result)
            is_error = bool(getattr(result, "is_error", False) or getattr(result, "isError", False))
            self._record(
                "response",
                "tools/call",
                {"content": [{"type": "text", "text": text}], "isError": is_error},
                call_id,
            )
            if is_error:
                return f"MCP tool '{name}' returned an error (recoverable): {text}"
            return text
        except Exception as exc:  # noqa: BLE001
            logger.warning("mcp_tool_call_failed", tool=name, error=str(exc))
            self._record(
                "response",
                "tools/call",
                {
                    "error": {"code": -32000, "message": str(exc)},
                    "isError": True,
                },
                call_id,
            )
            return f"MCP tool '{name}' failed (recoverable): {exc}"
        finally:
            reset_tool_context(token)

    async def _local_fallback(self, name: str, args: dict[str, Any], ctx: ToolContext) -> str:
        try:
            return execute_tool(name, args, ctx)
        except Exception as exc:  # noqa: BLE001
            return f"Tool '{name}' failed (recoverable): {exc}"

    def status_dict(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "transport": self.transport,
            "serverName": self.server_name,
            "connected": self.connected,
            "whereAiRuns": WHERE_AI_RUNS,
            "whereAiDoesNotRun": WHERE_AI_DOES_NOT_RUN,
            "endpoint": "/mcp",
            "stdio": "python -m app.mcp",
            "authRequired": bool(get_settings().mcp_auth_token),
            "tools": [t.to_dict() for t in self.tools],
            "resources": self.resources,
            "prompts": self.prompts,
            "handshake": [m.to_dict() for m in self.messages[:24]],
            "lastError": self.last_error,
            "roles": {
                "host": "ClaimIntel FastAPI + Angular — this is where the AI (Ollama) runs",
                "client": "McpToolGateway inside the host — speaks JSON-RPC to servers",
                "server": f"{self.server_name} — offers tools/resources/prompts, never runs the model",
            },
        }


def _schema_to_params(schema: dict[str, Any]) -> str:
    props = schema.get("properties") or {}
    if not props:
        return "{}"
    compact = {}
    for key, spec in props.items():
        if not isinstance(spec, dict):
            compact[key] = str(spec)
            continue
        hint = spec.get("description") or spec.get("title") or spec.get("type") or ""
        compact[key] = hint
    return json.dumps(compact)


def _result_text(result: Any) -> str:
    if result is None:
        return ""
    data = getattr(result, "data", None)
    if isinstance(data, str) and data.strip():
        return data
    content = getattr(result, "content", None) or []
    parts: list[str] = []
    for item in content:
        text = getattr(item, "text", None)
        if text:
            parts.append(str(text))
        elif isinstance(item, dict) and item.get("text"):
            parts.append(str(item["text"]))
    if parts:
        return "\n".join(parts)
    if data is not None:
        return str(data)
    return str(result)


_SHARED: McpToolGateway | None = None


def get_shared_gateway() -> McpToolGateway:
    global _SHARED
    if _SHARED is None:
        _SHARED = McpToolGateway()
    return _SHARED


async def reset_shared_gateway() -> None:
    global _SHARED
    if _SHARED is not None:
        await _SHARED.close()
    _SHARED = None
