from __future__ import annotations

import base64
import hashlib
import io
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

from app.intake_models import MaterialInput, PageBlock, ParsedMaterial
from app.ocr_client import ocr_image


_SUPPORTED_EXTENSIONS = {".txt", ".md", ".docx", ".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
_BLOCK_MAX_CHARS = 1200


def parse_material(item: MaterialInput) -> ParsedMaterial:
    """解析文本、Word、PDF 或扫描图片，并保留逻辑页码和原文块定位。"""

    suffix = Path(item.file_name).suffix.lower()
    if suffix and suffix not in _SUPPORTED_EXTENSIONS:
        raise ValueError(f"不支持的材料类型：{suffix}")

    raw = item.text.encode("utf-8") if item.text.strip() else _decode_base64(item.content_base64)
    digest = hashlib.sha256(raw).hexdigest()
    material_id = digest[:16]
    warnings: list[str] = []

    if item.text.strip() or suffix in {"", ".txt", ".md"}:
        pages = [item.text or raw.decode("utf-8")]
        method = "text"
    elif suffix == ".docx":
        pages = _read_docx_pages(raw)
        method = "docx_xml"
    elif suffix == ".pdf":
        pages, used_ocr, ocr_backend = _read_pdf_pages(raw)
        method = "ocr" if used_ocr else "pdf_text"
        if used_ocr:
            warnings.append(f"部分 PDF 页面未包含文本层，已使用 OCR（后端：{ocr_backend}）。")
    else:
        text, ocr_backend = _ocr_image(raw)
        pages = [text]
        method = "ocr"
        warnings.append(f"图片材料已使用 OCR（后端：{ocr_backend}）。")

    blocks: list[PageBlock] = []
    for page_no, page_text in enumerate(pages, start=1):
        for block_no, text in enumerate(_split_blocks(page_text), start=1):
            blocks.append(
                PageBlock(
                    material_id=material_id,
                    file_name=item.file_name,
                    material_type=item.material_type,
                    page_no=page_no,
                    block_no=block_no,
                    text=text,
                    extraction_method=method,
                )
            )
    if not blocks:
        raise ValueError(f"材料未提取到可用文本：{item.file_name}")

    return ParsedMaterial(
        material_id=material_id,
        file_name=item.file_name,
        material_type=item.material_type,
        sha256=digest,
        size_bytes=len(raw),
        page_count=max(len(pages), 1),
        blocks=blocks,
        warnings=warnings,
    )


def _decode_base64(value: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except ValueError as exc:
        raise ValueError("content_base64 不是合法的 Base64 内容") from exc


def _read_docx_pages(raw: bytes) -> list[str]:
    """读取 docx XML；Word 分页依赖排版，这里只对显式分页符保留逻辑页码。"""

    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with ZipFile(io.BytesIO(raw)) as archive:
        xml = archive.read("word/document.xml")
    root = ET.fromstring(xml)
    pages: list[list[str]] = [[]]
    for paragraph in root.findall(".//w:p", namespace):
        parts: list[str] = []
        page_break = False
        for node in paragraph.iter():
            tag = node.tag.rsplit("}", 1)[-1]
            if tag == "t" and node.text:
                parts.append(node.text)
            elif tag == "tab":
                parts.append("\t")
            elif tag in {"br", "lastRenderedPageBreak"}:
                if tag == "lastRenderedPageBreak" or node.attrib.get(f"{{{namespace['w']}}}type") == "page":
                    page_break = True
                else:
                    parts.append("\n")
        text = "".join(parts).strip()
        if text:
            pages[-1].append(text)
        if page_break:
            pages.append([])
    normalized = ["\n".join(page).strip() for page in pages]
    return [page for page in normalized if page] or [""]


def _read_pdf_pages(raw: bytes) -> tuple[list[str], bool, str]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("解析 PDF 需要安装 pypdf。") from exc

    reader = PdfReader(io.BytesIO(raw))
    pages: list[str] = []
    used_ocr = False
    backend = ""
    for page_index, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if not text:
            text, backend = _ocr_pdf_page(raw, page_index)
            used_ocr = True
        pages.append(text)
    return pages, used_ocr, backend


def _ocr_pdf_page(raw: bytes, page_index: int) -> tuple[str, str]:
    """用 PyMuPDF 把单页渲染成 2 倍图，再交给可切换的 OCR 后端。"""

    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("扫描 PDF OCR 需要 pymupdf。") from exc

    with fitz.open(stream=raw, filetype="pdf") as document:
        pixmap = document.load_page(page_index).get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        return ocr_image(pixmap.tobytes("png"))


def _ocr_image(raw: bytes) -> tuple[str, str]:
    return ocr_image(raw)


def _split_blocks(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [re.sub(r"\s+", " ", item).strip() for item in re.split(r"\n{1,}", normalized)]
    blocks: list[str] = []
    current = ""
    for paragraph in filter(None, paragraphs):
        if current and len(current) + len(paragraph) + 1 > _BLOCK_MAX_CHARS:
            blocks.append(current)
            current = ""
        while len(paragraph) > _BLOCK_MAX_CHARS:
            if current:
                blocks.append(current)
                current = ""
            blocks.append(paragraph[:_BLOCK_MAX_CHARS])
            paragraph = paragraph[_BLOCK_MAX_CHARS:]
        current = f"{current}\n{paragraph}".strip()
    if current:
        blocks.append(current)
    return blocks
