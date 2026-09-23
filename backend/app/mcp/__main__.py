"""Run the claims MCP server so another agent can plug in.

stdio (Claude Desktop / Cursor):
    python -m app.mcp

HTTP (remote MCP):
    python -m app.mcp --transport http --port 8765
"""

from __future__ import annotations

import argparse

from app.mcp.claims_server import get_claims_server


def main() -> None:
    parser = argparse.ArgumentParser(description="ClaimIntel MCP server (no AI runs here)")
    parser.add_argument("--transport", choices=("stdio", "http"), default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    mcp = get_claims_server()
    if args.transport == "http":
        mcp.run(transport="http", host=args.host, port=args.port)
        return
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
