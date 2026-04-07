"""Basic FastMCP server with Pydantic-validated tools.

Run as MCP server (HTTP):
	python src/fastmcp_server.py

Custom host/port/path:
	python src/fastmcp_server.py --host 127.0.0.1 --port 8000 --path /mcp

Required packages:
	pip install fastmcp pydantic
"""

from __future__ import annotations

import argparse
from datetime import date

from fastmcp import FastMCP
from pydantic import BaseModel, Field


mcp = FastMCP("basic-tools")


class AddInput(BaseModel):
	a: int = Field(description="First number")
	b: int = Field(description="Second number")


@mcp.tool
def add_numbers(payload: AddInput) -> int:
	"""Add two integers."""
	return payload.a + payload.b


@mcp.tool
def get_today() -> str:
	"""Return the current date in ISO format."""
	return date.today().isoformat()


def run_server(host: str, port: int, path: str) -> None:
	"""Start FastMCP server in HTTP mode for localhost/remote clients."""
	mcp.run(transport="http", host=host, port=port, path=path)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Basic FastMCP HTTP server with Pydantic tools"
	)
	parser.add_argument(
		"--host",
		default="127.0.0.1",
		help="Host to bind the HTTP MCP server to (default: 127.0.0.1)",
	)
	parser.add_argument(
		"--port",
		type=int,
		default=8000,
		help="Port for the HTTP MCP server (default: 8000)",
	)
	parser.add_argument(
		"--path",
		default="/mcp",
		help="HTTP path for MCP endpoint (default: /mcp)",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	run_server(host=args.host, port=args.port, path=args.path)


if __name__ == "__main__":
	main()
