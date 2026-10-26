"""扫描件 OCR 后端：优先百炼多模态模型，退回本机 Tesseract。

``OCR_BACKEND`` 取值：
- ``auto``（默认）：配了百炼 key 就用云端 VL，否则用本机 Tesseract
- ``bailian_vl`` / ``tesseract`` / ``off``：强制指定
"""

from __future__ import annotations

import base64
import io
import os
from typing import Any

import requests
from dotenv import load_dotenv


load_dotenv()
load_dotenv(".env.example", override=False)

OCR_BACKEND = os.getenv("OCR_BACKEND", "auto").strip().lower()
OCR_LANG = os.getenv("OCR_LANG", "chi_sim+eng")
VL_MODEL = os.getenv("VL_MODEL", "qwen-vl-ocr")
VL_BASE_URL = (os.getenv("VL_BASE_URL", "") or os.getenv("LLM_BASE_URL", "")).rstrip("/")
VL_API_KEY = os.getenv("VL_API_KEY", "") or os.getenv("DASHSCOPE_API_KEY", "") or os.getenv("LLM_API_KEY", "")
VL_TIMEOUT_SECONDS = float(os.getenv("VL_TIMEOUT_SECONDS", "60"))

_OCR_PROMPT = (
    "逐字转录图片中的全部文字，保留原有换行与字段顺序。"
    "表格按行输出，单元格之间用一个空格分隔。"
    "不要解释、不要补全、不要翻译；看不清的内容直接跳过。"
)


def ocr_image(raw: bytes) -> tuple[str, str]:
    """识别图片文字，返回 (文本, 实际使用的后端)。"""

    if OCR_BACKEND == "off":
        raise RuntimeError("OCR 已通过 OCR_BACKEND=off 关闭")

    if OCR_BACKEND in {"auto", "bailian_vl"}:
        if VL_BASE_URL and VL_API_KEY:
            return _bailian_vl_ocr(raw), "bailian_vl"
        if OCR_BACKEND == "bailian_vl":
            raise RuntimeError("OCR_BACKEND=bailian_vl 但未配置 VL_BASE_URL / VL_API_KEY")

    return _tesseract_ocr(raw), "tesseract"


def get_ocr_status() -> dict[str, Any]:
    """返回当前可用的 OCR 后端，供健康检查与前端展示。"""

    configured = bool(VL_BASE_URL and VL_API_KEY)
    tesseract_version: str | None = None
    try:
        import pytesseract

        tesseract_version = str(pytesseract.get_tesseract_version())
    except Exception:
        tesseract_version = None

    if OCR_BACKEND == "off":
        active = "off"
    elif OCR_BACKEND == "bailian_vl":
        active = "bailian_vl" if configured else "unavailable"
    elif OCR_BACKEND == "tesseract":
        active = "tesseract" if tesseract_version else "unavailable"
    else:
        active = "bailian_vl" if configured else ("tesseract" if tesseract_version else "unavailable")

    return {
        "backend_config": OCR_BACKEND,
        "active_backend": active,
        "vl_model": VL_MODEL,
        "vl_configured": configured,
        "tesseract_version": tesseract_version,
        "tesseract_lang": OCR_LANG,
    }


def _bailian_vl_ocr(raw: bytes) -> str:
    endpoint = f"{VL_BASE_URL}/chat/completions" if VL_BASE_URL.endswith("/v1") else f"{VL_BASE_URL}/v1/chat/completions"
    image_type = _guess_mime(raw)
    response = requests.post(
        endpoint,
        headers={"Authorization": f"Bearer {VL_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": VL_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{image_type};base64,{base64.b64encode(raw).decode('ascii')}"}},
                        {"type": "text", "text": _OCR_PROMPT},
                    ],
                }
            ],
            "temperature": 0,
        },
        timeout=VL_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content).strip()


def _tesseract_ocr(raw: bytes) -> str:
    try:
        from PIL import Image
        import pytesseract
    except ImportError as exc:
        raise RuntimeError("本机 Tesseract OCR 需要 Pillow 与 pytesseract。") from exc
    return pytesseract.image_to_string(Image.open(io.BytesIO(raw)), lang=OCR_LANG).strip()


def _guess_mime(raw: bytes) -> str:
    if raw[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if raw[:4] == b"RIFF":
        return "image/webp"
    if raw[:4] == b"%PDF":
        return "application/pdf"
    return "image/png"
