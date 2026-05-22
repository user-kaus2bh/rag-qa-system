import time
from dataclasses import dataclass, field
from typing import List, Optional, Iterator

from src.llm.client import LLMClient, LLMResponse
from src.llm.prompts import (
    Message,
    build_rag_prompt,
    build_condense_prompt,
    NO_CONTEXT_RESPONSE,
)
from src.retrieval.retriever import Retriever
from src.embeddings.vector_store import SearchResult


@dataclass
class RAGResponse:
    answer: str
    question: str
    standalone_question: str
    sources: List[SearchResult]
    context: str
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def source_files(self) -> List[str]:
        seen = set()
        files = []
        for s in self.sources:
            fname = s.chunk.metadata.get("filename", s.chunk.source.split("/")[-1])
            if fname not in seen:
                seen.add(fname)
                files.append(fname)
        return files

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "standalone_question": self.standalone_question,
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "source_files": self.source_files,
            "retrieval_time_ms": round(self.retrieval_time_ms, 2),
            "generation_time_ms": round(self.generation_time_ms, 2),
            "total_time_ms": round(self.total_time_ms, 2),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


class RAGChain:
    def __init__(
        self,
        retriever: Retriever,
        llm_client: LLMClient,
        top_k: int = 5,
        score_threshold: float = 0.0,
        use_conversation_memory: bool = True,
        max_history_turns: int = 6,
        condense_questions: bool = True,
    ):
        self.retriever = retriever
        self.llm = llm_client
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.use_conversation_memory = use_conversation_memory
        self.max_history_turns = max_history_turns
        self.condense_questions = condense_questions
        self._history: List[Message] = []

    def ask(self, question: str) -> RAGResponse:
        t_start = time.time()

        standalone_question = self._resolve_question(question)

        t_retrieval_start = time.time()
        results = self.retriever.retrieve(standalone_question, top_k=self.top_k)
        retrieval_ms = (time.time() - t_retrieval_start) * 1000

        if not results:
            answer = NO_CONTEXT_RESPONSE
            generation_ms = 0.0
            input_tokens = 0
            output_tokens = 0
        else:
            context = self.retriever.format_context(results)
            rag_prompt = build_rag_prompt(context=context, question=question)

            t_gen_start = time.time()
            history = self._get_history()
            llm_response: LLMResponse = self.llm.complete(
                user_message=rag_prompt,
                history=history,
            )
            generation_ms = (time.time() - t_gen_start) * 1000
            answer = llm_response.content
            input_tokens = llm_response.input_tokens
            output_tokens = llm_response.output_tokens

        context_text = self.retriever.format_context(results) if results else ""
        total_ms = (time.time() - t_start) * 1000

        if self.use_conversation_memory:
            self._history.append(Message(role="user", content=question))
            self._history.append(Message(role="assistant", content=answer))

        return RAGResponse(
            answer=answer,
            question=question,
            standalone_question=standalone_question,
            sources=results,
            context=context_text,
            retrieval_time_ms=retrieval_ms,
            generation_time_ms=generation_ms,
            total_time_ms=total_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def stream(self, question: str):
        standalone_question = self._resolve_question(question)
        results = self.retriever.retrieve(standalone_question, top_k=self.top_k)

        if not results:
            yield NO_CONTEXT_RESPONSE, results
            return

        context = self.retriever.format_context(results)
        rag_prompt = build_rag_prompt(context=context, question=question)
        history = self._get_history()

        full_answer = ""
        for chunk in self.llm.stream(user_message=rag_prompt, history=history):
            full_answer += chunk
            yield chunk, results

        if self.use_conversation_memory:
            self._history.append(Message(role="user", content=question))
            self._history.append(Message(role="assistant", content=full_answer))

    def _resolve_question(self, question: str) -> str:
        if not self.condense_questions or not self._history:
            return question

        history = self._get_history()
        if not history:
            return question

        condense_prompt = build_condense_prompt(
            chat_history=history,
            question=question,
        )
        try:
            response = self.llm.complete(user_message=condense_prompt)
            condensed = response.content.strip()
            return condensed if condensed else question
        except Exception:
            return question

    def _get_history(self) -> List[Message]:
        max_messages = self.max_history_turns * 2
        return self._history[-max_messages:]

    def clear_history(self) -> None:
        self._history = []

    @property
    def history(self) -> List[Message]:
        return list(self._history)

    @property
    def turn_count(self) -> int:
        return len(self._history) // 2
