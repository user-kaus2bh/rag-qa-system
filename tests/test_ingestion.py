import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.loaders import DocumentLoader, RawDocument
from src.ingestion.chunker import TextChunker, ChunkStrategy, Chunk


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_txt(tmp_dir):
    path = tmp_dir / "sample.txt"
    path.write_text(
        "Artificial intelligence is transforming industries worldwide.\n\n"
        "Machine learning models learn from vast datasets.\n\n"
        "Natural language processing enables machines to understand human text.\n\n"
        "Deep learning uses neural networks with many layers.\n\n"
        "Transformers are a key architecture in modern NLP systems.",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def sample_md(tmp_dir):
    path = tmp_dir / "sample.md"
    path.write_text(
        "# Introduction to RAG\n\n"
        "Retrieval-Augmented Generation combines retrieval with generation.\n\n"
        "## Vector Search\n\n"
        "Embeddings map text to dense vector space for similarity search.\n\n"
        "## LLM Integration\n\n"
        "Large language models generate answers grounded in retrieved context.",
        encoding="utf-8",
    )
    return path


class TestDocumentLoader:
    def test_load_txt_returns_raw_document(self, sample_txt):
        loader = DocumentLoader()
        docs = loader.load(sample_txt)
        assert len(docs) == 1
        assert isinstance(docs[0], RawDocument)
        assert docs[0].file_type == "txt"
        assert "Artificial intelligence" in docs[0].content

    def test_load_md_returns_markdown_type(self, sample_md):
        loader = DocumentLoader()
        docs = loader.load(sample_md)
        assert len(docs) == 1
        assert docs[0].file_type == "markdown"

    def test_load_unsupported_raises(self, tmp_dir):
        bad = tmp_dir / "file.xyz"
        bad.write_text("data")
        loader = DocumentLoader()
        with pytest.raises(ValueError, match="Unsupported file type"):
            loader.load(bad)

    def test_load_missing_file_raises(self):
        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/file.txt")

    def test_load_directory(self, tmp_dir, sample_txt, sample_md):
        loader = DocumentLoader()
        docs = loader.load_directory(tmp_dir)
        assert len(docs) == 2

    def test_metadata_populated(self, sample_txt):
        loader = DocumentLoader()
        docs = loader.load(sample_txt)
        assert "loaded_at" in docs[0].metadata
        assert "file_size_bytes" in docs[0].metadata
        assert docs[0].metadata["filename"] == "sample.txt"

    def test_doc_id_is_stable(self, sample_txt):
        loader = DocumentLoader()
        docs1 = loader.load(sample_txt)
        docs2 = loader.load(sample_txt)
        assert docs1[0].doc_id == docs2[0].doc_id

    def test_clean_text_strips_extra_whitespace(self, tmp_dir):
        path = tmp_dir / "messy.txt"
        path.write_text("hello   world\n\n\n\nextra   spaces  here\r\n", encoding="utf-8")
        loader = DocumentLoader()
        docs = loader.load(path)
        assert "\n\n\n" not in docs[0].content
        assert "  " not in docs[0].content


class TestTextChunker:
    def test_recursive_chunking_produces_chunks(self, sample_txt):
        loader = DocumentLoader()
        docs = loader.load(sample_txt)
        chunker = TextChunker(chunk_size=100, chunk_overlap=20, min_chunk_length=10)
        chunks = chunker.chunk_documents(docs)
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_no_chunk_exceeds_chunk_size_by_too_much(self, sample_txt):
        loader = DocumentLoader()
        docs = loader.load(sample_txt)
        chunk_size = 100
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=20, min_chunk_length=10)
        chunks = chunker.chunk_documents(docs)
        for chunk in chunks:
            assert chunk.char_count <= chunk_size * 2.5

    def test_min_length_filter_applied(self, tmp_dir):
        path = tmp_dir / "short.txt"
        path.write_text("Hi.\n\nThis is a much longer paragraph with enough content to pass the filter.", encoding="utf-8")
        loader = DocumentLoader()
        docs = loader.load(path)
        chunker = TextChunker(chunk_size=200, chunk_overlap=20, min_chunk_length=30)
        chunks = chunker.chunk_documents(docs)
        assert all(chunk.char_count >= 30 for chunk in chunks)

    def test_chunk_metadata_is_correct(self, sample_txt):
        loader = DocumentLoader()
        docs = loader.load(sample_txt)
        chunker = TextChunker(chunk_size=200, chunk_overlap=30, min_chunk_length=10)
        chunks = chunker.chunk_documents(docs)
        for chunk in chunks:
            assert chunk.source == str(sample_txt)
            assert chunk.file_type == "txt"
            assert chunk.chunk_index >= 0
            assert chunk.total_chunks == len(chunks)
            assert chunk.chunk_id != ""

    def test_chunk_ids_are_unique(self, sample_txt):
        loader = DocumentLoader()
        docs = loader.load(sample_txt)
        chunker = TextChunker(chunk_size=150, chunk_overlap=20, min_chunk_length=10)
        chunks = chunker.chunk_documents(docs)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_sentence_strategy(self, sample_txt):
        loader = DocumentLoader()
        docs = loader.load(sample_txt)
        chunker = TextChunker(chunk_size=200, chunk_overlap=30, strategy=ChunkStrategy.SENTENCE)
        chunks = chunker.chunk_documents(docs)
        assert len(chunks) > 0

    def test_paragraph_strategy(self, sample_txt):
        loader = DocumentLoader()
        docs = loader.load(sample_txt)
        chunker = TextChunker(chunk_size=300, chunk_overlap=30, strategy=ChunkStrategy.PARAGRAPH)
        chunks = chunker.chunk_documents(docs)
        assert len(chunks) > 0

    def test_fixed_strategy(self, sample_txt):
        loader = DocumentLoader()
        docs = loader.load(sample_txt)
        chunker = TextChunker(chunk_size=100, chunk_overlap=10, strategy=ChunkStrategy.FIXED)
        chunks = chunker.chunk_documents(docs)
        assert all(chunk.char_count <= 100 for chunk in chunks)

    def test_save_and_load_chunks(self, tmp_dir, sample_txt):
        loader = DocumentLoader()
        docs = loader.load(sample_txt)
        chunker = TextChunker(chunk_size=200, chunk_overlap=30)
        chunks = chunker.chunk_documents(docs)
        output_path = tmp_dir / "chunks.json"
        chunker.save_chunks(chunks, output_path)
        assert output_path.exists()
        loaded = chunker.load_chunks(output_path)
        assert len(loaded) == len(chunks)
        assert loaded[0].chunk_id == chunks[0].chunk_id
        assert loaded[0].content == chunks[0].content

    def test_overlap_raises_on_bad_config(self):
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            TextChunker(chunk_size=100, chunk_overlap=100)

    def test_token_estimate_positive(self, sample_txt):
        loader = DocumentLoader()
        docs = loader.load(sample_txt)
        chunker = TextChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk_documents(docs)
        assert all(c.token_estimate > 0 for c in chunks)

    def test_to_dict_and_from_dict_roundtrip(self, sample_txt):
        loader = DocumentLoader()
        docs = loader.load(sample_txt)
        chunker = TextChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk_documents(docs)
        for chunk in chunks:
            d = chunk.to_dict()
            assert isinstance(d, dict)
            restored = Chunk.from_dict(d)
            assert restored.chunk_id == chunk.chunk_id
            assert restored.content == chunk.content
