from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.intake_api import router as intake_router
from app.intake_scenes import scene_names


def create_app() -> FastAPI:
    """创建只承载智能受理业务的 FastAPI 应用。"""

    app = FastAPI(
        title="智能受理 Agent",
        description="材料理解、受理合规核验、异常处置与可追溯业务文书生成。",
        version="1.0.0",
    )
    app.include_router(intake_router)

    web_dir = Path(__file__).resolve().parent.parent / "web"
    app.mount("/demo", StaticFiles(directory=web_dir, html=True), name="demo")

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/demo/")

    @app.get("/api/info")
    def api_info() -> dict[str, Any]:
        return {
            "name": "智能受理 Agent",
            "time": datetime.now().isoformat(timespec="seconds"),
            "business_scenes": scene_names(),
            "architecture": "fixed_pipeline_with_controlled_react_and_reflection",
            "vector_database": "Zilliz Cloud",
            "endpoints": [
                "POST /intake/cases/process",
                "POST /intake/cases/upload",
                "GET /intake/runs/{run_id}",
                "GET /intake/config",
                "GET /intake/health",
            ],
        }

    return app
