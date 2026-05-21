import sys
import json
import tempfile
import numpy as np
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.chunker import Chunk
from src.embeddings.vector_store import VectorStore, SearchResult
from src.retrieval.retriever import Retriever


def make_chunk(content: str, chunk_id: str = None, source: str = "test.txt", page: int = 1) -> Chunk:
    return Chunk(
        chunk_id=chunk_id or f"id_{hash(content) % 100000:05d}",
        content=content,
        source=source,
        file_type="txt",
        chunk_index=0,
        total_chunks=1,
        page=page,
        total_pages=1,
        strategy="recursive",
        metadata={"filename": source, "page": page},
    )


SAMPLE_CHUNKS = [
    make_chunk("Python is a high-level programming language known for its simplicity.", "c001"),
    make_chunk("Machine learning is a subset of artificial intelligence.", "c002"),
    make_chunk("FAISS is a library for efficient similarity search on dense vectors.", "c003"),
    make_chunk("Neural networks are inspired by the human brain structure.", "c004"),
    make_chunk("Retrieval-Augmented Generation combines retrieval with language models.", "c005"),
    make_chunk("NumPy provides multi-dimensional array support in Python.", "c006"),
    make_chunk("Transformers use attention mechanisms to process sequential data.", "c007"),
    make_chunk("Vector embeddings map text to a continuous high-dimensional space.", "c008"),
]


@pytest.fixture(scope="module")
def built_store(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("vectorstore")
    store = VectorStore(
        embedding_model="all-MiniLM-L6-v2",
        index_path=tmp / "index.faiss",
        metadata_path=tmp / "metadata.json",
    )
    store.build(SAMPLE_CHUNKS)
    return store


@pytest.fixture(scope="module")
def saved_store(built_store, tmp_path_factory):
    built_store.save()
    return built_store


class TestVectorStore:
    def test_build_loads_index(self, built_store):
        assert built_store.is_loaded()

    def test_total_vectors_matches_chunks(self, built_store):
        assert built_store.total_vectors == len(SAMPLE_CHUNKS)

    def test_total_chunks_matches(self, built_store):
        assert built_store.total_chunks == len(SAMPLE_CHUNKS)

    def test_dimension_is_positive(self, built_store):
        assert built_store.dimension > 0

    def test_search_returns_results(self, built_store):
        results = built_store.search("what is machine learning", top_k=3)
        assert len(results) > 0

    def test_search_result_type(self, built_store):
        results = built_store.search("Python programming", top_k=2)
        for r in results:
            assert isinstance(r, SearchResult)
            assert isinstance(r.chunk, Chunk)
            assert isinstance(r.score, float)
            assert isinstance(r.rank, int)

    def test_search_top_k_respected(self, built_store):
        results = built_store.search("neural network deep learning", top_k=3)
        assert len(results) <= 3

    def test_search_scores_descending(self, built_store):
        results = built_store.search("vector embeddings similarity", top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_ranks_ascending(self, built_store):
        results = built_store.search("retrieval augmented generation", top_k=4)
        ranks = [r.rank for r in results]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_semantic_relevance(self, built_store):
        results = built_store.search("FAISS vector similarity search library", top_k=3)
        top_content = results[0].chunk.content.lower()
        assert any(word in top_content for word in ["faiss", "similarity", "vector", "search"])

    def test_search_with_score_threshold(self, built_store):
        results = built_store.search("quantum physics thermodynamics", top_k=5, score_threshold=0.99)
        assert all(r.score >= 0.99 for r in results)

    def test_search_empty_results_on_high_threshold(self, built_store):
        results = built_store.search("some query", top_k=5, score_threshold=1.0)
        assert len(results) == 0

    def test_encode_returns_normalized_vectors(self, built_store):
        vecs = built_store.encode(["hello world"], show_progress=False)
        norm = np.linalg.norm(vecs[0])
        assert abs(norm - 1.0) < 1e-5

    def test_build_raises_on_empty_chunks(self, tmp_path):
        store = VectorStore(
            embedding_model="all-MiniLM-L6-v2",
            index_path=tmp_path / "idx.faiss",
            metadata_path=tmp_path / "meta.json",
        )
        with pytest.raises(ValueError, match="empty"):
            store.build([])

    def test_search_raises_if_not_loaded(self, tmp_path):
        store = VectorStore(embedding_model="all-MiniLM-L6-v2")
        with pytest.raises(RuntimeError, match="not loaded"):
            store.search("query")

    def test_save_creates_files(self, saved_store):
        assert saved_store.index_path.exists()
        assert saved_store.metadata_path.exists()

    def test_load_restores_index(self, saved_store):
        store2 = VectorStore(
            embedding_model="all-MiniLM-L6-v2",
            index_path=saved_store.index_path,
            metadata_path=saved_store.metadata_path,
        )
        store2.load()
        assert store2.is_loaded()
        assert store2.total_vectors == saved_store.total_vectors
        assert store2.total_chunks == saved_store.total_chunks

    def test_load_search_matches_original(self, saved_store):
        store2 = VectorStore(
            embedding_model="all-MiniLM-L6-v2",
            index_path=saved_store.index_path,
            metadata_path=saved_store.metadata_path,
        )
        store2.load()
        r1 = saved_store.search("machine learning AI", top_k=1)
        r2 = store2.search("machine learning AI", top_k=1)
        assert r1[0].chunk.chunk_id == r2[0].chunk.chunk_id

    def test_metadata_file_contains_expected_keys(self, saved_store):
        with open(saved_store.metadata_path) as f:
            meta = json.load(f)
        assert "embedding_model" in meta
        assert "dimension" in meta
        assert "total_vectors" in meta
        assert "chunks" in meta
        assert len(meta["chunks"]) == len(SAMPLE_CHUNKS)

    def test_search_batch_returns_list_of_lists(self, built_store):
        queries = ["Python language", "neural networks brain"]
        results = built_store.search_batch(queries, top_k=2)
        assert len(results) == 2
        assert all(isinstance(r, list) for r in results)

    def test_load_missing_index_raises(self, tmp_path):
        store = VectorStore(
            embedding_model="all-MiniLM-L6-v2",
            index_path=tmp_path / "nonexistent.faiss",
            metadata_path=tmp_path / "nonexistent.json",
        )
        with pytest.raises(FileNotFoundError):
            store.load()


class TestRetriever:
    def test_retrieve_returns_results(self, built_store):
        retriever = Retriever(built_store, top_k=3)
        results = retriever.retrieve("what is Python")
        assert len(results) > 0

    def test_retrieve_top_k_respected(self, built_store):
        retriever = Retriever(built_store, top_k=2)
        results = retriever.retrieve("machine learning")
        assert len(results) <= 2

    def test_retrieve_deduplication(self, built_store):
        retriever = Retriever(built_store, top_k=5, deduplicate=True)
        results = retriever.retrieve("Python programming language")
        ids = [r.chunk.chunk_id for r in results]
        assert len(ids) == len(set(ids))

    def test_retrieve_ranks_are_sequential(self, built_store):
        retriever = Retriever(built_store, top_k=4)
        results = retriever.retrieve("vector embeddings")
        assert [r.rank for r in results] == list(range(1, len(results) + 1))

    def test_format_context_not_empty(self, built_store):
        retriever = Retriever(built_store, top_k=3)
        results = retriever.retrieve("retrieval augmented generation")
        ctx = retriever.format_context(results)
        assert len(ctx) > 0
        assert "[Source 1]" in ctx

    def test_format_context_includes_filename(self, built_store):
        retriever = Retriever(built_store, top_k=2)
        results = retriever.retrieve("neural network")
        ctx = retriever.format_context(results, include_metadata=True)
        assert "File:" in ctx

    def test_format_context_empty_results(self, built_store):
        retriever = Retriever(built_store, top_k=3)
        ctx = retriever.format_context([])
        assert "No relevant context" in ctx

    def test_retrieve_batch(self, built_store):
        retriever = Retriever(built_store, top_k=2)
        all_results = retriever.retrieve_batch(["Python", "deep learning"])
        assert len(all_results) == 2
        assert all(len(r) <= 2 for r in all_results)
