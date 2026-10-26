"""重建证据检索后端（Zilliz 集合 / Elasticsearch 索引）。

schema 变更后必须重建，否则旧集合缺 ``sparse`` 字段会降级为纯稠密召回，
旧索引缺 ``text`` 字段则 BM25 无法命中。

用法：
    python scripts/intake_collection.py                 # 查看两边现状
    python scripts/intake_collection.py --target es     # 只看 Elasticsearch
    python scripts/intake_collection.py --recreate      # 两边都按当前 schema 重建
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.es_client import ES_INDEX, elasticsearch_evidence_store  # noqa: E402
from app.intake_integrations import (  # noqa: E402
    MILVUS_COLLECTION,
    intake_vector_store,
)


def _report_zilliz(recreate: bool) -> None:
    if not intake_vector_store.configured:
        print("Zilliz：未配置 MILVUS_ENDPOINT / MILVUS_TOKEN / 百炼向量 key，跳过")
        return
    client = intake_vector_store._get_client()
    if recreate and client.has_collection(collection_name=MILVUS_COLLECTION):
        intake_vector_store.drop()
        print(f"Zilliz：已删除集合 {MILVUS_COLLECTION}")
    if not client.has_collection(collection_name=MILVUS_COLLECTION):
        intake_vector_store._ensure_collection(client)
        print(f"Zilliz：已按当前 schema 创建集合 {MILVUS_COLLECTION}")
    detail = client.describe_collection(collection_name=MILVUS_COLLECTION)
    fields = [field.get("name") for field in detail.get("fields", [])]
    print(f"Zilliz：字段 {fields}")
    print(f"Zilliz：BM25 稀疏检索 {'可用' if 'sparse' in fields else '不可用（需 --recreate）'}")


def _report_elasticsearch(recreate: bool) -> None:
    if not elasticsearch_evidence_store.configured:
        print("Elasticsearch：未配置 ES_URL，跳过")
        return
    if recreate:
        elasticsearch_evidence_store.drop()
        print(f"Elasticsearch：已删除索引 {ES_INDEX}")
    status = elasticsearch_evidence_store.status()
    if not status.get("index_exists"):
        elasticsearch_evidence_store._ensure_index()
        print(f"Elasticsearch：已按当前 mapping 创建索引 {ES_INDEX}")
    print(f"Elasticsearch：集群状态 {status.get('cluster_status')}，分词器 {elasticsearch_evidence_store.analyzer or status.get('analyzer')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="证据检索后端集合/索引管理")
    parser.add_argument("--recreate", action="store_true", help="删除后按当前 schema 重建（会清空已入库数据）")
    parser.add_argument("--target", choices=["zilliz", "es", "all"], default="all")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if args.target in {"zilliz", "all"}:
        _report_zilliz(args.recreate)
    if args.target in {"es", "all"}:
        _report_elasticsearch(args.recreate)
