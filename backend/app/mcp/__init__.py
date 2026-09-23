"""MCP host/client/server pieces for the claims agent.

The model still runs on the host (this FastAPI app + Ollama). The server only
offers tools, resources, and prompts.
"""

from app.mcp.claims_server import create_claims_server, get_claims_server
from app.mcp.gateway import McpToolGateway, get_shared_gateway

__all__ = [
    "McpToolGateway",
    "create_claims_server",
    "get_claims_server",
    "get_shared_gateway",
]
