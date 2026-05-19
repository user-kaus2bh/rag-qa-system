import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_MIN_LENGTH,
    SUPPORTED_EXTENSIONS,
)
from src.ingestion.loaders import DocumentLoader
from src.ingestion.chunker import TextChunker, ChunkStrategy


def ingest(
    input_path: Path,
    output_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    min_length: int,
    strategy: str,
    recursive: bool,
) -> dict:
    loader = DocumentLoader(supported_extensions=SUPPORTED_EXTENSIONS)
    chunker = TextChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_length=min_length,
        strategy=ChunkStrategy(strategy),
    )

    print(f"\n{'='*52}")
    print(f"  RAG Ingestion Pipeline")
    print(f"{'='*52}")
    print(f"  Input   : {input_path}")
    print(f"  Output  : {output_dir}")
    print(f"  Strategy: {strategy}")
    print(f"  Chunk   : size={chunk_size}  overlap={chunk_overlap}  min={min_length}")
    print(f"{'='*52}\n")

    print("[1/3] Loading documents...")
    if input_path.is_dir():
        documents = loader.load_directory(input_path, recursive=recursive)
    else:
        documents = loader.load(input_path)

    if not documents:
        print("  No documents loaded. Check the input path and file types.")
        return {}

    total_chars = sum(len(d.content) for d in documents)
    print(f"  Loaded {len(documents)} document page(s) | {total_chars:,} chars total\n")

    print("[2/3] Chunking text...")
    chunks = chunker.chunk_documents(documents)
    print(f"  Produced {len(chunks)} chunks\n")

    print("[3/3] Saving output...")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    output_file = output_dir / f"chunks_{timestamp}.json"
    chunker.save_chunks(chunks, output_file)

    manifest_file = output_dir / "manifest.json"
    manifest = {
        "created_at": datetime.utcnow().isoformat(),
        "chunks_file": str(output_file.name),
        "total_documents": len(documents),
        "total_chunks": len(chunks),
        "total_chars": total_chars,
        "config": {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "min_chunk_length": min_length,
            "strategy": strategy,
        },
        "sources": list({d.source for d in documents}),
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*52}")
    print(f"  Done.")
    print(f"  Documents : {len(documents)}")
    print(f"  Chunks    : {len(chunks)}")
    print(f"  Output    : {output_file}")
    print(f"{'='*52}\n")

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="RAG QA System — Document Ingestion Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DATA_RAW_DIR),
        help="Path to a file or directory of documents to ingest",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DATA_PROCESSED_DIR),
        help="Directory where chunked JSON output will be saved",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help="Maximum characters per chunk",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=CHUNK_OVERLAP,
        help="Overlap characters between consecutive chunks",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=CHUNK_MIN_LENGTH,
        help="Minimum characters for a chunk to be kept",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="recursive",
        choices=["recursive", "sentence", "paragraph", "fixed"],
        help="Chunking strategy to use",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Disable recursive directory scanning",
    )

    args = parser.parse_args()

    ingest(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_length=args.min_length,
        strategy=args.strategy,
        recursive=not args.no_recursive,
    )


if __name__ == "__main__":
    main()
