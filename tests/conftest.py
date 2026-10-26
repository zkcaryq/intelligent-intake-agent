"""测试隔离：单元测试不得读取本机 .env 后误连真实基础设施。"""

from __future__ import annotations

import os


for name in (
    "DASHSCOPE_API_KEY",
    "LLM_API_KEY",
    "EMBEDDING_API_KEY",
    "RERANKER_API_KEY",
    "INTAKE_DATABASE_URL",
    "DATABASE_URL",
    "LEGAL_DATABASE_URL",
    "INTAKE_REDIS_URL",
    "REDIS_URL",
    "MILVUS_ENDPOINT",
    "MILVUS_URL",
    "MILVUS_TOKEN",
    "OSS_ENDPOINT",
    "OSS_BUCKET",
    "OSS_ACCESS_KEY_ID",
    "OSS_ACCESS_KEY_SECRET",
    "INTAKE_ARCHIVE_API_BASE_URL",
    "INTAKE_ARCHIVE_API_TOKEN",
):
    os.environ[name] = ""

os.environ["LEGAL_KB_BACKEND"] = "mock"
