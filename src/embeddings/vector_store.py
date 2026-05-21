import json
import time
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

from src.ingestion.chunker import Chunk


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    rank: int

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "score": round(float(self.score), 6),
            "chunk_id": self.chunk.chunk_id,
            "content": self.chunk.content,
            "source": self.chunk.source,
            "page": self.chunk.page,
            "file_type": self.chunk.file_type,
            "metadata": self.chunk.metadata,
        }


class VectorStore:
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        index_path: Optional[str | Path] = None,
        metadata_path: Optional[str | Path] = None,
    ):
        self.embedding_model_name = embedding_model
        self.index_path = Path(index_path) if index_path else None
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self._model = None
        self._index = None
        self._chunks: List[Chunk] = []

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"  Loading embedding model: {self.embedding_model_name}")
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    def build(self, chunks: List[Chunk], batch_size: int = 64) -> None:
        import faiss

        if not chunks:
            raise ValueError("Cannot build index from empty chunk list")

        print(f"\n  Building FAISS index from {len(chunks)} chunks...")
        t0 = time.time()

        texts = [c.content for c in chunks]
        embeddings = self.encode(texts, batch_size=batch_size, show_progress=True)

        index = faiss.IndexFlatIP(self.dimension)
        index.add(embeddings)

        self._index = index
        self._chunks = chunks

        elapsed = time.time() - t0
        print(f"  Index built: {index.ntotal} vectors | dim={self.dimension} | {elapsed:.1f}s")

    def save(self, index_path: Optional[str | Path] = None, metadata_path: Optional[str | Path] = None) -> None:
        import faiss

        if self._index is None or not self._chunks:
            raise RuntimeError("No index to save. Call build() first.")

        idx_path = Path(index_path) if index_path else self.index_path
        meta_path = Path(metadata_path) if metadata_path else self.metadata_path

        if idx_path is None or meta_path is None:
            raise ValueError("Provide index_path and metadata_path either in __init__ or save()")

        idx_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(idx_path))

        metadata = {
            "embedding_model": self.embedding_model_name,
            "dimension": self.dimension,
            "total_vectors": self._index.ntotal,
            "chunks": [c.to_dict() for c in self._chunks],
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False)

        print(f"  Saved index  → {idx_path}")
        print(f"  Saved metadata → {meta_path}")

    def load(self, index_path: Optional[str | Path] = None, metadata_path: Optional[str | Path] = None) -> None:
        import faiss

        idx_path = Path(index_path) if index_path else self.index_path
        meta_path = Path(metadata_path) if metadata_path else self.metadata_path

        if idx_path is None or meta_path is None:
            raise ValueError("Provide index_path and metadata_path either in __init__ or load()")

        if not idx_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {idx_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        print(f"  Loading FAISS index from {idx_path}...")
        self._index = faiss.read_index(str(idx_path))

        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.embedding_model_name = metadata["embedding_model"]
        self._chunks = [Chunk.from_dict(c) for c in metadata["chunks"]]

        print(f"  Loaded {self._index.ntotal} vectors | {len(self._chunks)} chunks")

    def is_loaded(self) -> bool:
        return self._index is not None and len(self._chunks) > 0

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[SearchResult]:
        if not self.is_loaded():
            raise RuntimeError("Index not loaded. Call build() or load() first.")

        query_vec = self.encode([query], show_progress=False)
        top_k_clamped = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_vec, top_k_clamped)

        results: List[SearchResult] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:
                continue
            if float(score) < score_threshold:
                continue
            results.append(
                SearchResult(
                    chunk=self._chunks[idx],
                    score=float(score),
                    rank=rank + 1,
                )
            )

        return results

    def search_batch(
        self,
        queries: List[str],
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[List[SearchResult]]:
        if not self.is_loaded():
            raise RuntimeError("Index not loaded. Call build() or load() first.")

        query_vecs = self.encode(queries, show_progress=False)
        top_k_clamped = min(top_k, self._index.ntotal)
        scores_batch, indices_batch = self._index.search(query_vecs, top_k_clamped)

        all_results: List[List[SearchResult]] = []
        for scores, indices in zip(scores_batch, indices_batch):
            results: List[SearchResult] = []
            for rank, (score, idx) in enumerate(zip(scores, indices)):
                if idx == -1:
                    continue
                if float(score) < score_threshold:
                    continue
                results.append(
                    SearchResult(
                        chunk=self._chunks[idx],
                        score=float(score),
                        rank=rank + 1,
                    )
                )
            all_results.append(results)

        return all_results

    @property
    def total_chunks(self) -> int:
        return len(self._chunks)

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal if self._index else 0
