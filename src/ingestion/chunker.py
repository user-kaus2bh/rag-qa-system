import re
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

from .loaders import RawDocument


class ChunkStrategy(str, Enum):
    RECURSIVE = "recursive"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    FIXED = "fixed"


@dataclass
class Chunk:
    chunk_id: str
    content: str
    source: str
    file_type: str
    chunk_index: int
    total_chunks: int
    page: Optional[int] = None
    total_pages: Optional[int] = None
    strategy: str = "recursive"
    char_count: int = 0
    token_estimate: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.char_count = len(self.content)
        self.token_estimate = max(1, self.char_count // 4)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "source": self.source,
            "file_type": self.file_type,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "page": self.page,
            "total_pages": self.total_pages,
            "strategy": self.strategy,
            "char_count": self.char_count,
            "token_estimate": self.token_estimate,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(**data)


class TextChunker:
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_length: int = 50,
        strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length
        self.strategy = strategy

        self._recursive_separators = [
            "\n\n",
            "\n",
            ". ",
            "! ",
            "? ",
            "; ",
            ", ",
            " ",
            "",
        ]

    def chunk_documents(self, documents: List[RawDocument]) -> List[Chunk]:
        all_chunks: List[Chunk] = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        return all_chunks

    def chunk_document(self, document: RawDocument) -> List[Chunk]:
        strategy_map = {
            ChunkStrategy.RECURSIVE: self._recursive_split,
            ChunkStrategy.SENTENCE: self._sentence_split,
            ChunkStrategy.PARAGRAPH: self._paragraph_split,
            ChunkStrategy.FIXED: self._fixed_split,
        }

        splitter = strategy_map[self.strategy]
        raw_chunks = splitter(document.content)
        raw_chunks = [c for c in raw_chunks if len(c.strip()) >= self.min_chunk_length]

        total = len(raw_chunks)
        chunks: List[Chunk] = []

        for idx, text in enumerate(raw_chunks):
            chunk_id = hashlib.md5(
                f"{document.source}:{document.page}:{idx}:{text[:32]}".encode()
            ).hexdigest()[:16]

            chunk = Chunk(
                chunk_id=chunk_id,
                content=text.strip(),
                source=document.source,
                file_type=document.file_type,
                chunk_index=idx,
                total_chunks=total,
                page=document.page,
                total_pages=document.total_pages,
                strategy=self.strategy.value,
                metadata={
                    **document.metadata,
                    "doc_id": document.doc_id,
                    "chunk_index": idx,
                    "total_chunks": total,
                },
            )
            chunks.append(chunk)

        return chunks

    def _recursive_split(self, text: str) -> List[str]:
        return self._split_recursive(text, self._recursive_separators)

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        separator = ""
        for sep in separators:
            if sep and sep in text:
                separator = sep
                break

        if not separator:
            return self._fixed_split(text)

        splits = text.split(separator)
        chunks: List[str] = []
        current_parts: List[str] = []
        current_length = 0

        for split in splits:
            split_len = len(split) + len(separator)

            if current_length + split_len > self.chunk_size and current_parts:
                chunk_text = separator.join(current_parts).strip()
                if chunk_text:
                    if len(chunk_text) > self.chunk_size:
                        sub_chunks = self._split_recursive(chunk_text, separators[1:])
                        chunks.extend(sub_chunks)
                    else:
                        chunks.append(chunk_text)

                overlap_parts: List[str] = []
                overlap_length = 0
                for part in reversed(current_parts):
                    part_len = len(part) + len(separator)
                    if overlap_length + part_len > self.chunk_overlap:
                        break
                    overlap_parts.insert(0, part)
                    overlap_length += part_len

                current_parts = overlap_parts
                current_length = overlap_length

            current_parts.append(split)
            current_length += split_len

        if current_parts:
            final_text = separator.join(current_parts).strip()
            if final_text:
                if len(final_text) > self.chunk_size:
                    sub_chunks = self._split_recursive(final_text, separators[1:])
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(final_text)

        return chunks

    def _sentence_split(self, text: str) -> List[str]:
        sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
        sentences = sentence_endings.split(text)

        chunks: List[str] = []
        current_sentences: List[str] = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sent_len = len(sentence)

            if current_length + sent_len > self.chunk_size and current_sentences:
                chunks.append(" ".join(current_sentences))

                overlap_sents: List[str] = []
                overlap_length = 0
                for s in reversed(current_sentences):
                    if overlap_length + len(s) > self.chunk_overlap:
                        break
                    overlap_sents.insert(0, s)
                    overlap_length += len(s)

                current_sentences = overlap_sents
                current_length = sum(len(s) for s in current_sentences)

            current_sentences.append(sentence)
            current_length += sent_len

        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return chunks

    def _paragraph_split(self, text: str) -> List[str]:
        paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]

        chunks: List[str] = []
        current_paras: List[str] = []
        current_length = 0

        for para in paragraphs:
            para_len = len(para)

            if para_len > self.chunk_size:
                if current_paras:
                    chunks.append("\n\n".join(current_paras))
                    current_paras = []
                    current_length = 0
                sub_chunks = self._recursive_split(para)
                chunks.extend(sub_chunks)
                continue

            if current_length + para_len > self.chunk_size and current_paras:
                chunks.append("\n\n".join(current_paras))
                current_paras = []
                current_length = 0

            current_paras.append(para)
            current_length += para_len

        if current_paras:
            chunks.append("\n\n".join(current_paras))

        return chunks

    def _fixed_split(self, text: str) -> List[str]:
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def save_chunks(self, chunks: List[Chunk], output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in chunks], f, ensure_ascii=False, indent=2)
        print(f"  Saved {len(chunks)} chunks to {output_path}")

    def load_chunks(self, input_path: str | Path) -> List[Chunk]:
        input_path = Path(input_path)
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [Chunk.from_dict(d) for d in data]
