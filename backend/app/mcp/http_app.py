"""Remote MCP transport: Streamable HTTP mounted on the host, with bearer auth."""

from __future__ import annotations

from collections.abc import Callable

from app.core.config import get_settings
from app.core.logging import get_logger
from app.mcp.claims_server import get_claims_server

logger = get_logger(__name__)


class BearerAuthASGI:
    """Require Authorization: Bearer <token> on HTTP MCP calls when a token is set."""

    def __init__(self, app: Callable, token: str) -> None:
        self.app = app
        self.token = token.strip()

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope.get("type") != "http" or not self.token:
            await self.app(scope, receive, send)
            return
        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        if headers.get("authorization", "") == f"Bearer {self.token}":
            await self.app(scope, receive, send)
            return

        body = b'{"jsonrpc":"2.0","error":{"code":-32001,"message":"unauthorized"}}'
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"www-authenticate", b"Bearer"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def build_mcp_http_app():
    """ASGI app for /mcp. Session lifespan must be wired into FastAPI."""
    server = get_claims_server()
    try:
        mcp_app = server.http_app(path="/")
    except TypeError:
        mcp_app = server.http_app()
    settings = get_settings()
    token = settings.mcp_auth_token
    if token:
        logger.info("mcp_http_auth_enabled")
        return BearerAuthASGI(mcp_app, token), mcp_app
    logger.warning("mcp_http_auth_disabled", hint="set MCP_AUTH_TOKEN for remote callers")
    return mcp_app, mcp_app
