"""智能受理 MCP Server：向受控 Agent 暴露工单、档案和原文定位工具。

stdio 模式供本地 Agent 直接拉起；HTTP 模式供云端 MCP 客户端（如百炼）接入：
    python scripts/intake_mcp_server.py --http --port 8765
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from app.intake_tools import locate_source_evidence, query_supplier_archive, query_ticket  # noqa: E402


def build_server() -> FastMCP:
    server = FastMCP(
        "intelligent-intake-business-tools",
        host=os.getenv("INTAKE_MCP_HOST", "127.0.0.1"),
        port=int(os.getenv("INTAKE_MCP_PORT", "8765")),
    )

    @server.tool()
    def get_ticket(case_no: str) -> dict:
        """按案件编号查询工单。"""

        return query_ticket(case_no)

    @server.tool()
    def get_supplier_archive(entity_name: str) -> dict:
        """按主体名称查询供应商档案。"""

        return query_supplier_archive(entity_name)

    @server.tool()
    def search_material_evidence(case_no: str, query: str) -> dict:
        """按问题关键词定位材料原文。"""

        return locate_source_evidence(case_no, query)

    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="智能受理 MCP Server")
    parser.add_argument("--http", action="store_true", help="以 Streamable HTTP 传输启动")
    parser.add_argument("--port", type=int, default=int(os.getenv("INTAKE_MCP_PORT", "8765")))
    args = parser.parse_args()
    if args.port:
        os.environ["INTAKE_MCP_PORT"] = str(args.port)
    build_server().run(transport="streamable-http" if args.http else "stdio")
