"""重建受理材料向量集合。

schema 变更后（例如加入 BM25 稀疏字段）必须重建，否则旧集合没有 ``sparse`` 字段，
检索会自动降级为纯稠密召回。

用法：
    python scripts/intake_collection.py            # 查看当前 schema
    python scripts/intake_collection.py --recreate # 删除后按新 schema 重建
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.intake_integrations import (  # noqa: E402
    MILVUS_COLLECTION,
    intake_vector_store,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="受理材料向量集合管理")
    parser.add_argument("--recreate", action="store_true", help="删除并重建集合（会清空已入库向量）")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not intake_vector_store.configured:
        print("未配置 MILVUS_ENDPOINT / MILVUS_TOKEN / 百炼向量 key，跳过。")
        raise SystemExit(1)

    client = intake_vector_store._get_client()
    if args.recreate:
        intake_vector_store.drop()
        print(f"已删除集合 {MILVUS_COLLECTION}")
    if not client.has_collection(collection_name=MILVUS_COLLECTION):
        intake_vector_store._ensure_collection(client)
        print(f"已按当前 schema 创建集合 {MILVUS_COLLECTION}")

    detail = client.describe_collection(collection_name=MILVUS_COLLECTION)
    fields = [field.get("name") for field in detail.get("fields", [])]
    print(f"字段：{fields}")
    print(f"BM25 稀疏检索：{'可用' if 'sparse' in fields else '不可用（旧 schema，需 --recreate）'}")
