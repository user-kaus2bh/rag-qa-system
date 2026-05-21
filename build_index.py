import sys
import json
import glob
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_PROCESSED_DIR,
    VECTORSTORE_DIR,
    EMBEDDING_MODEL,
    FAISS_INDEX_FILE,
    FAISS_METADATA_FILE,
    TOP_K_RETRIEVAL,
)
from src.ingestion.chunker import TextChunker
from src.embeddings.vector_store import VectorStore


def load_latest_chunks(processed_dir: Path) -> list:
    pattern = str(processed_dir / "chunks_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No chunk files found in {processed_dir}. Run python ingest.py first."
        )
    latest = files[-1]
    print(f"  Loading chunks from: {Path(latest).name}")
    chunker = TextChunker()
    return chunker.load_chunks(latest)


def build(
    chunks_path: Path | None,
    index_path: Path,
    metadata_path: Path,
    embedding_model: str,
    batch_size: int,
) -> dict:
    print(f"\n{'='*52}")
    print(f"  RAG — Build Vector Index")
    print(f"{'='*52}")
    print(f"  Embedding model : {embedding_model}")
    print(f"  Index output    : {index_path}")
    print(f"{'='*52}\n")

    print("[1/3] Loading chunks...")
    if chunks_path:
        from src.ingestion.chunker import TextChunker
        chunks = TextChunker().load_chunks(chunks_path)
        print(f"  Loaded {len(chunks)} chunks from {chunks_path.name}")
    else:
        chunks = load_latest_chunks(DATA_PROCESSED_DIR)
        print(f"  Loaded {len(chunks)} chunks")

    print("\n[2/3] Encoding and indexing...")
    store = VectorStore(
        embedding_model=embedding_model,
        index_path=index_path,
        metadata_path=metadata_path,
    )
    store.build(chunks, batch_size=batch_size)

    print("\n[3/3] Saving index...")
    store.save()

    print(f"\n[Verify] Running a test search...")
    test_query = "What is this document about?"
    results = store.search(test_query, top_k=3)
    print(f"  Query: '{test_query}'")
    for r in results:
        preview = r.chunk.content[:80].replace("\n", " ")
        print(f"  [{r.rank}] score={r.score:.4f} | {preview}...")

    summary = {
        "built_at": datetime.utcnow().isoformat(),
        "embedding_model": embedding_model,
        "total_chunks": len(chunks),
        "total_vectors": store.total_vectors,
        "dimension": store.dimension,
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
    }

    print(f"\n{'='*52}")
    print(f"  Done.")
    print(f"  Chunks embedded : {len(chunks)}")
    print(f"  Vectors stored  : {store.total_vectors}")
    print(f"  Dimension       : {store.dimension}")
    print(f"{'='*52}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="RAG QA System — Build FAISS Vector Index",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--chunks",
        type=str,
        default=None,
        help="Path to a specific chunks JSON file. Defaults to the latest in data/processed/",
    )
    parser.add_argument(
        "--index",
        type=str,
        default=str(FAISS_INDEX_FILE),
        help="Output path for the FAISS index file",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=str(FAISS_METADATA_FILE),
        help="Output path for the metadata JSON file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=EMBEDDING_MODEL,
        help="Sentence-transformers model name to use for embeddings",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for encoding chunks",
    )

    args = parser.parse_args()

    build(
        chunks_path=Path(args.chunks) if args.chunks else None,
        index_path=Path(args.index),
        metadata_path=Path(args.metadata),
        embedding_model=args.model,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
