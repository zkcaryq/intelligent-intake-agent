from __future__ import annotations

import hashlib
import math
import os
from typing import Any

import requests
from dotenv import load_dotenv

from app.redaction import mask_config_value, redact_sensitive_text


load_dotenv()
load_dotenv(".env.example", override=False)

EMBEDDING_BASE_URL = os.getenv(
    "EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
).rstrip("/")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "") or os.getenv("DASHSCOPE_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3.7-text-embedding")
EMBEDDING_TIMEOUT_SECONDS = float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "30"))
EMBEDDING_FALLBACK_DIMENSION = int(
    os.getenv("EMBEDDING_DIM", os.getenv("EMBEDDING_FALLBACK_DIMENSION", "1024"))
)
_LAST_EMBEDDING_SOURCE = "fallback"
_LAST_EMBEDDING_ERROR = ""


def is_embedding_configured() -> bool:
    """判断是否配置了 embedding 服务地址。"""

    return bool(EMBEDDING_BASE_URL and EMBEDDING_API_KEY)


def get_embedding_config_status() -> dict[str, Any]:
    """返回脱敏后的 embedding 配置状态。"""

    return {
        "configured": is_embedding_configured(),
        "base_url": mask_config_value(EMBEDDING_BASE_URL),
        "base_url_set": bool(EMBEDDING_BASE_URL),
        "model": EMBEDDING_MODEL,
        "provider": "dashscope_bailian",
        "timeout_seconds": EMBEDDING_TIMEOUT_SECONDS,
        "api_key_set": bool(EMBEDDING_API_KEY),
        "fallback_dimension": EMBEDDING_FALLBACK_DIMENSION,
        "last_embedding_source": _LAST_EMBEDDING_SOURCE,
        "last_error": redact_sensitive_text(_LAST_EMBEDDING_ERROR),
    }


def get_embedding_runtime_model() -> str:
    """返回最近一次实际用于检索的向量模型名称。"""

    if _LAST_EMBEDDING_SOURCE == "remote":
        return EMBEDDING_MODEL
    return "local-demo-vector"


def embed_texts(texts: list[str]) -> list[list[float]]:
    """生成文本向量；生产调用百炼 embedding，未配置时使用确定性测试向量兜底。"""

    global _LAST_EMBEDDING_ERROR, _LAST_EMBEDDING_SOURCE
    if not texts:
        return []
    if is_embedding_configured():
        try:
            vectors = _embed_texts_remote(texts)
            _LAST_EMBEDDING_SOURCE = "remote"
            _LAST_EMBEDDING_ERROR = ""
            return vectors
        except Exception as exc:
            _LAST_EMBEDDING_SOURCE = "fallback"
            _LAST_EMBEDDING_ERROR = redact_sensitive_text(f"{type(exc).__name__}: {exc}")
            return [_embed_text_fallback(text) for text in texts]
    _LAST_EMBEDDING_SOURCE = "fallback"
    _LAST_EMBEDDING_ERROR = ""
    return [_embed_text_fallback(text) for text in texts]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """计算两个向量的余弦相似度。"""

    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _embed_texts_remote(texts: list[str]) -> list[list[float]]:
    """调用 OpenAI-compatible embeddings 接口。"""

    endpoint = (
        f"{EMBEDDING_BASE_URL}/embeddings"
        if EMBEDDING_BASE_URL.endswith("/v1")
        else f"{EMBEDDING_BASE_URL}/v1/embeddings"
    )
    headers = {"Content-Type": "application/json"}
    if EMBEDDING_API_KEY:
        headers["Authorization"] = f"Bearer {EMBEDDING_API_KEY}"
    vectors: list[list[float]] = []
    for offset in range(0, len(texts), 10):
        response = requests.post(
            endpoint,
            headers=headers,
            json={
                "model": EMBEDDING_MODEL,
                "input": texts[offset : offset + 10],
                "dimensions": EMBEDDING_FALLBACK_DIMENSION,
            },
            timeout=EMBEDDING_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        body = response.json()
        vectors.extend(
            item["embedding"] for item in sorted(body["data"], key=lambda item: item["index"])
        )
    if vectors and len(vectors[0]) != EMBEDDING_FALLBACK_DIMENSION:
        raise ValueError(
            f"百炼向量维度不匹配：期望 {EMBEDDING_FALLBACK_DIMENSION}，实际 {len(vectors[0])}"
        )
    return vectors


def _embed_text_fallback(text: str) -> list[float]:
    """本地 fallback 向量，仅用于未配置 embedding 服务时保证 demo 可运行。"""

    vector = [0.0] * EMBEDDING_FALLBACK_DIMENSION
    normalized = "".join(text.split())
    tokens = [normalized[index : index + 2] for index in range(max(len(normalized) - 1, 1))]
    if not tokens:
        tokens = [normalized]
    for token in tokens:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % EMBEDDING_FALLBACK_DIMENSION
        vector[index] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if not norm:
        return vector
    return [value / norm for value in vector]
