#!/usr/bin/env python3
"""Index sample-data/*.txt into Chroma (+ BM25) for local retrieval eval.

Designed to run inside the document-processor container:

  docker cp sample-data/. insurance-rag-document-processor:/tmp/sample-data
  docker cp eval/index_sample_for_eval.py insurance-rag-document-processor:/tmp/index_sample_for_eval.py
  docker exec -w /home/site/wwwroot insurance-rag-document-processor \\
    python /tmp/index_sample_for_eval.py --docs /tmp/sample-data
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

# Azure Functions site root
sys.path.insert(0, "/home/site/wwwroot")

from services.bm25_index_service import Bm25IndexService
from services.embedding_service import get_embedding_service
from services.vector_store_service import VectorStoreService


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=Path, default=Path("/tmp/sample-data"))
    parser.add_argument("--clear", action="store_true", default=True)
    args = parser.parse_args()

    files = sorted(args.docs.glob("*.txt"))
    if not files:
        print(f"No .txt files in {args.docs}", file=sys.stderr)
        return 1

    store = VectorStoreService()
    if args.clear:
        store.clear_collection()
        Bm25IndexService.get().clear()

    embedder = get_embedding_service()
    total = 0
    for path in files:
        content = path.read_text(encoding="utf-8")
        pieces = chunk_text(content)
        if not pieces:
            continue
        doc_id = uuid.uuid5(uuid.NAMESPACE_URL, path.name)
        ids = [str(uuid.uuid5(doc_id, f"{path.name}:{i}")) for i in range(len(pieces))]
        metadatas = [
            {
                "document_id": str(doc_id),
                "file_name": path.name,
                "page_number": 1,
                "chunk_index": i,
            }
            for i in range(len(pieces))
        ]
        embeddings = embedder.embed_texts(pieces)
        store.upsert_chunks(
            document_id=doc_id,
            file_name=path.name,
            ids=ids,
            documents=pieces,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        total += len(pieces)
        print(f"indexed {path.name}: {len(pieces)} chunks")

    print(f"done: {total} chunks from {len(files)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
