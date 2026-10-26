from __future__ import annotations

import json
import os
import time
from typing import Any

import requests
from dotenv import load_dotenv

from app.redaction import mask_config_value, redact_sensitive_text


load_dotenv()
load_dotenv(".env.example", override=False)

LLM_BASE_URL = os.getenv(
    "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
).rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "") or os.getenv("DASHSCOPE_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.7-flash")
LLM_ENABLE_THINKING = os.getenv("LLM_ENABLE_THINKING", "false").lower() == "true"
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
LLM_RETRY_BACKOFF_SECONDS = float(os.getenv("LLM_RETRY_BACKOFF_SECONDS", "1.5"))
"""大模型 API 配置。密钥和地址只从环境变量读取，避免把敏感信息写入代码仓库。"""


def is_llm_configured() -> bool:
    """判断是否已经配置大模型调用所需的地址和密钥。"""

    return bool(LLM_BASE_URL and LLM_API_KEY)


def get_llm_config_status() -> dict[str, Any]:
    """返回脱敏后的大模型配置状态，方便排查模型服务连接问题。"""

    return {
        "configured": is_llm_configured(),
        "base_url": mask_config_value(LLM_BASE_URL),
        "base_url_set": bool(LLM_BASE_URL),
        "model": LLM_MODEL,
        "provider": "dashscope_bailian",
        "enable_thinking": LLM_ENABLE_THINKING,
        "timeout_seconds": LLM_TIMEOUT_SECONDS,
        "max_retries": LLM_MAX_RETRIES,
        "retry_backoff_seconds": LLM_RETRY_BACKOFF_SECONDS,
        "api_key_set": bool(LLM_API_KEY),
    }


def ping_llm() -> dict[str, Any]:
    """用极短 prompt 测试模型连通性和 JSON 响应能力。"""

    start = time.perf_counter()
    result = call_llm_json(
        [
            {"role": "system", "content": "只输出 JSON。"},
            {"role": "user", "content": '请输出 {"ok": true}。'},
        ],
        timeout_seconds=min(LLM_TIMEOUT_SECONDS, 30),
        max_retries=0,
    )
    return {
        "ok": True,
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "result": result,
        "config": get_llm_config_status(),
    }


def call_llm_json(
    messages: list[dict[str, str]],
    *,
    timeout_seconds: float | None = None,
    max_retries: int | None = None,
) -> dict[str, Any]:
    """调用 OpenAI 兼容的大模型接口，并带超时、重试和 JSON 解析。"""

    if not LLM_BASE_URL or not LLM_API_KEY:
        return {
            "enabled": False,
            "error": "未配置 LLM_BASE_URL 或 LLM_API_KEY，已使用规则结果。",
        }

    endpoint = f"{LLM_BASE_URL}/chat/completions" if LLM_BASE_URL.endswith("/v1") else f"{LLM_BASE_URL}/v1/chat/completions"
    timeout = timeout_seconds or LLM_TIMEOUT_SECONDS
    retries = LLM_MAX_RETRIES if max_retries is None else max_retries
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "enable_thinking": LLM_ENABLE_THINKING,
    }

    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {LLM_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            break
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc
            if attempt >= retries:
                raise
            time.sleep(LLM_RETRY_BACKOFF_SECONDS * (attempt + 1))
        except requests.HTTPError:
            raise
    else:
        if last_exc:
            raise last_exc
        raise RuntimeError("LLM request failed without a captured exception.")

    content = response.json()["choices"][0]["message"]["content"]
    if isinstance(content, dict):
        return content
    return json.loads(content)


def redact_llm_error(exc: Exception) -> str:
    """Format an exception without exposing model service endpoint details."""

    return redact_sensitive_text(f"{type(exc).__name__}: {exc}")
