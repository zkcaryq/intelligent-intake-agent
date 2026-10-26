from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Any

import requests
from dotenv import dotenv_values, load_dotenv

from app.redaction import mask_config_value, redact_sensitive_text


load_dotenv()
load_dotenv(".env.example", override=False)

RERANKER_BASE_URL = os.getenv(
    "RERANKER_BASE_URL",
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
).rstrip("/")
RERANKER_API_KEY = os.getenv("RERANKER_API_KEY", "") or os.getenv("DASHSCOPE_API_KEY", "")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "qwen3-rerank")
RERANKER_TIMEOUT_SECONDS = float(os.getenv("RERANKER_TIMEOUT_SECONDS", "30"))
_LAST_RERANKER_ERROR = ""
_LAST_RERANKER_SOURCE = "disabled"


@dataclass(frozen=True)
class RerankResult:
    """reranker 对单个候选文档的重排结果。"""

    index: int
    score: float


@dataclass(frozen=True)
class RerankerConfig:
    """reranker 运行时配置，优先读取当前 .env，避免模块导入时缓存旧值。"""

    base_url: str
    api_key: str
    model: str
    timeout_seconds: float


def is_reranker_configured() -> bool:
    """判断是否配置了 reranker 服务地址。"""

    config = _read_reranker_config()
    return bool(config.base_url and config.api_key)


def get_reranker_config_status() -> dict[str, Any]:
    """返回脱敏后的 reranker 配置状态。"""

    config = _read_reranker_config()
    return {
        "configured": is_reranker_configured(),
        "base_url": mask_config_value(config.base_url),
        "base_url_set": bool(config.base_url),
        "model": config.model,
        "provider": "dashscope_bailian",
        "timeout_seconds": config.timeout_seconds,
        "api_key_set": bool(config.api_key),
        "last_reranker_source": _LAST_RERANKER_SOURCE,
        "last_error": redact_sensitive_text(_LAST_RERANKER_ERROR),
    }


def get_reranker_runtime_model() -> str:
    """返回当前配置的 reranker 模型名称。"""

    return _read_reranker_config().model


def rerank_documents(query: str, documents: list[str], top_n: int) -> list[RerankResult]:
    """调用 OpenAI-compatible rerank 接口；失败时返回空列表，由上游回退向量分数。"""

    global _LAST_RERANKER_ERROR, _LAST_RERANKER_SOURCE
    config = _read_reranker_config()
    if not query or not documents or not config.base_url:
        _LAST_RERANKER_SOURCE = "disabled"
        _LAST_RERANKER_ERROR = ""
        return []
    try:
        results = _rerank_documents_remote(query, documents, top_n, config)
        _LAST_RERANKER_SOURCE = "remote"
        _LAST_RERANKER_ERROR = ""
        return results
    except Exception as exc:
        _LAST_RERANKER_SOURCE = "fallback_vector_score"
        _LAST_RERANKER_ERROR = redact_sensitive_text(f"{type(exc).__name__}: {exc}")
        return []


def _read_reranker_config() -> RerankerConfig:
    """从 .env.example、.env 和系统环境变量读取最新 reranker 配置。"""

    values: dict[str, Any] = {}
    values.update(dotenv_values(".env.example"))
    values.update(dotenv_values(".env"))
    for key in (
        "DASHSCOPE_API_KEY",
        "RERANKER_BASE_URL",
        "RERANKER_API_KEY",
        "RERANKER_MODEL",
        "RERANKER_TIMEOUT_SECONDS",
    ):
        env_value = os.getenv(key)
        if env_value not in (None, ""):
            values[key] = env_value
    return RerankerConfig(
        base_url=str(
            values.get("RERANKER_BASE_URL")
            or "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
        ).rstrip("/"),
        api_key=str(values.get("RERANKER_API_KEY") or values.get("DASHSCOPE_API_KEY") or ""),
        model=str(values.get("RERANKER_MODEL") or "qwen3-rerank"),
        timeout_seconds=float(values.get("RERANKER_TIMEOUT_SECONDS") or 30),
    )


def _rerank_documents_remote(
    query: str,
    documents: list[str],
    top_n: int,
    config: RerankerConfig,
) -> list[RerankResult]:
    """调用 rerank 接口，并兼容 /v1/rerank 与 /rerank 两种常见路径。"""

    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    if "/api/v1/services/rerank/" in config.base_url:
        response = requests.post(
            config.base_url,
            headers=headers,
            json={
                "model": config.model,
                "input": {"query": query, "documents": documents},
                "parameters": {"top_n": top_n, "return_documents": False},
            },
            timeout=config.timeout_seconds,
        )
        response.raise_for_status()
        return _parse_rerank_response(response.json())

    payload = {"model": config.model, "query": query, "documents": documents, "top_n": top_n}

    last_response = None
    for endpoint in _candidate_rerank_endpoints(config.base_url):
        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=config.timeout_seconds,
        )
        if response.status_code != 404:
            response.raise_for_status()
            return _parse_rerank_response(response.json())
        last_response = response
    response = last_response
    if response is None:
        raise RuntimeError("未生成 reranker 请求地址")
    response.raise_for_status()
    return []


def _candidate_rerank_endpoints(base_url: str) -> list[str]:
    """根据配置地址生成候选 reranker 路径。"""

    if base_url.endswith("/v1"):
        root = base_url[:-3]
        return [f"{base_url}/rerank", f"{root}/rerank"]
    return [f"{base_url}/v1/rerank", f"{base_url}/rerank"]


def _parse_rerank_response(body: dict[str, Any]) -> list[RerankResult]:
    """解析 reranker 返回值，支持 results/data/scores 三类常见格式。"""

    if isinstance(body.get("scores"), list):
        return [
            RerankResult(index=index, score=float(score))
            for index, score in enumerate(body["scores"])
        ]

    raw_results = (body.get("output") or {}).get("results") or body.get("results") or body.get("data") or []
    parsed = []
    for position, item in enumerate(raw_results):
        index = item.get("index", position)
        score = item.get("relevance_score", item.get("score", item.get("similarity", 0.0)))
        parsed.append(RerankResult(index=int(index), score=_normalize_score(float(score))))
    return parsed


def _normalize_score(score: float) -> float:
    """把 reranker 原始分数转换到 0-1，便于配置统一阈值。"""

    if 0.0 <= score <= 1.0:
        return score
    return 1.0 / (1.0 + math.exp(-score))
