from typing import List, Optional
from src.embeddings.vector_store import VectorStore, SearchResult


class Retriever:
    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 5,
        score_threshold: float = 0.0,
        deduplicate: bool = True,
    ):
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.deduplicate = deduplicate

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        k = top_k or self.top_k
        results = self.vector_store.search(
            query=query,
            top_k=k * 2 if self.deduplicate else k,
            score_threshold=self.score_threshold,
        )

        if self.deduplicate:
            results = self._deduplicate(results)

        results = results[:k]

        for i, r in enumerate(results):
            r.rank = i + 1

        return results

    def retrieve_batch(self, queries: List[str], top_k: Optional[int] = None) -> List[List[SearchResult]]:
        k = top_k or self.top_k
        all_results = self.vector_store.search_batch(
            queries=queries,
            top_k=k * 2 if self.deduplicate else k,
            score_threshold=self.score_threshold,
        )

        final: List[List[SearchResult]] = []
        for results in all_results:
            if self.deduplicate:
                results = self._deduplicate(results)
            results = results[:k]
            for i, r in enumerate(results):
                r.rank = i + 1
            final.append(results)

        return final

    def _deduplicate(self, results: List[SearchResult]) -> List[SearchResult]:
        seen_ids: set = set()
        unique: List[SearchResult] = []
        for result in results:
            if result.chunk.chunk_id not in seen_ids:
                seen_ids.add(result.chunk.chunk_id)
                unique.append(result)
        return unique

    def format_context(self, results: List[SearchResult], include_metadata: bool = True) -> str:
        if not results:
            return "No relevant context found."

        parts: List[str] = []
        for result in results:
            chunk = result.chunk
            header_parts = [f"[Source {result.rank}]"]
            if include_metadata:
                filename = chunk.metadata.get("filename", chunk.source.split("/")[-1])
                header_parts.append(f"File: {filename}")
                if chunk.page:
                    header_parts.append(f"Page: {chunk.page}")
                header_parts.append(f"Score: {result.score:.4f}")
            header = " | ".join(header_parts)
            parts.append(f"{header}\n{chunk.content}")

        return "\n\n---\n\n".join(parts)
