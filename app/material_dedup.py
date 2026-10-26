from __future__ import annotations

import hashlib
import re
from itertools import combinations

from app.embedding_client import cosine_similarity, embed_texts
from app.intake_models import DuplicateCheck, ParsedMaterial


SIMHASH_THRESHOLD = 0.90
VECTOR_THRESHOLD = 0.92
LCS_THRESHOLD = 0.88
_LCS_MAX_CHARS = 800


def check_material_duplicates(materials: list[ParsedMaterial]) -> list[DuplicateCheck]:
    """依次执行 SHA256、SimHash、向量相似度和最长公共子序列四级去重。"""

    if len(materials) < 2:
        return []
    texts = [_normalize(material.full_text) for material in materials]
    vectors = embed_texts(texts)
    results: list[DuplicateCheck] = []
    for left_index, right_index in combinations(range(len(materials)), 2):
        left = materials[left_index]
        right = materials[right_index]
        sha_equal = left.sha256 == right.sha256
        if sha_equal:
            simhash_score = vector_score = lcs_score = 1.0
        else:
            simhash_score = _simhash_similarity(texts[left_index], texts[right_index])
            vector_score = cosine_similarity(vectors[left_index], vectors[right_index])
            # LCS 是 O(n*m)，仅对前三层判定为候选相似的材料继续精核。
            lcs_score = (
                _lcs_similarity(texts[left_index], texts[right_index])
                if simhash_score >= 0.65 or vector_score >= 0.65
                else 0.0
            )
        level, conclusion = _classify_duplicate(sha_equal, simhash_score, vector_score, lcs_score)
        results.append(
            DuplicateCheck(
                left_material_id=left.material_id,
                right_material_id=right.material_id,
                sha256_equal=sha_equal,
                simhash_similarity=round(simhash_score, 4),
                vector_similarity=round(vector_score, 4),
                lcs_similarity=round(lcs_score, 4),
                conclusion=conclusion,
                matched_level=level,
            )
        )
    return results


def _classify_duplicate(
    sha_equal: bool,
    simhash_score: float,
    vector_score: float,
    lcs_score: float,
) -> tuple[str, str]:
    if sha_equal:
        return "sha256", "重复"
    if simhash_score >= SIMHASH_THRESHOLD:
        return "simhash", "高度相似"
    if vector_score >= VECTOR_THRESHOLD:
        return "vector", "高度相似"
    if lcs_score >= LCS_THRESHOLD:
        return "lcs", "高度相似"
    return "none", "不重复"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _simhash_similarity(left: str, right: str, bits: int = 64) -> float:
    left_hash = _simhash(left, bits)
    right_hash = _simhash(right, bits)
    distance = (left_hash ^ right_hash).bit_count()
    return 1.0 - distance / bits


def _simhash(text: str, bits: int) -> int:
    tokens = _text_tokens(text)
    weights = [0] * bits
    for token in tokens:
        digest = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        for index in range(bits):
            weights[index] += 1 if digest & (1 << index) else -1
    result = 0
    for index, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << index
    return result


def _text_tokens(text: str) -> list[str]:
    if not text:
        return [""]
    chinese_bigrams = [text[index : index + 2] for index in range(max(len(text) - 1, 1))]
    latin_words = re.findall(r"[a-z0-9_]+", text)
    return chinese_bigrams + latin_words


def _lcs_similarity(left: str, right: str) -> float:
    """计算截断文本的真实 LCS 比例，限制长度以避免 O(n*m) 内存和时间失控。"""

    left = left[:_LCS_MAX_CHARS]
    right = right[:_LCS_MAX_CHARS]
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    if len(left) < len(right):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for left_char in left:
        current = [0]
        for index, right_char in enumerate(right, start=1):
            if left_char == right_char:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return 2.0 * previous[-1] / (len(left) + len(right))
