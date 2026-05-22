import sys
from pathlib import Path
from typing import List, Iterator
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.prompts import (
    Message, build_rag_prompt, build_condense_prompt,
    format_chat_history, SYSTEM_PROMPT, NO_CONTEXT_RESPONSE,
    RAG_PROMPT_TEMPLATE, CONDENSE_QUESTION_TEMPLATE,
)
from src.llm.client import LLMClient, LLMResponse
from src.embeddings.vector_store import SearchResult
from src.ingestion.chunker import Chunk
from src.pipeline.rag_chain import RAGChain, RAGResponse


def make_chunk(content: str, chunk_id: str = "c001", page: int = 1) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        content=content,
        source="test.pdf",
        file_type="pdf",
        chunk_index=0,
        total_chunks=3,
        page=page,
        total_pages=5,
        strategy="recursive",
        metadata={"filename": "test.pdf", "page": page},
    )


def make_search_result(content: str, score: float = 0.85, rank: int = 1, chunk_id: str = "c001") -> SearchResult:
    return SearchResult(
        chunk=make_chunk(content, chunk_id=chunk_id),
        score=score,
        rank=rank,
    )


def make_mock_llm(response_text: str = "This is a test answer.") -> LLMClient:
    mock = MagicMock(spec=LLMClient)
    mock.complete.return_value = LLMResponse(
        content=response_text,
        model="claude-3-5-sonnet-20241022",
        input_tokens=100,
        output_tokens=50,
        stop_reason="end_turn",
    )
    mock.stream.return_value = iter([response_text])
    return mock


def make_mock_retriever(results: List[SearchResult] = None):
    from src.retrieval.retriever import Retriever
    mock = MagicMock(spec=Retriever)
    mock.retrieve.return_value = results or [
        make_search_result("The refund policy allows returns within 30 days.", score=0.88, rank=1, chunk_id="c001"),
        make_search_result("Items must be unused and in original packaging.", score=0.75, rank=2, chunk_id="c002"),
    ]
    mock.format_context.return_value = (
        "[Source 1] | File: test.pdf | Page: 1 | Score: 0.8800\n"
        "The refund policy allows returns within 30 days.\n\n---\n\n"
        "[Source 2] | File: test.pdf | Page: 1 | Score: 0.7500\n"
        "Items must be unused and in original packaging."
    )
    return mock


class TestMessage:
    def test_to_dict(self):
        m = Message(role="user", content="hello")
        assert m.to_dict() == {"role": "user", "content": "hello"}

    def test_assistant_role(self):
        m = Message(role="assistant", content="hi there")
        assert m.role == "assistant"


class TestPrompts:
    def test_build_rag_prompt_contains_context(self):
        prompt = build_rag_prompt(context="some context", question="what is this?")
        assert "some context" in prompt
        assert "what is this?" in prompt

    def test_build_rag_prompt_contains_template_structure(self):
        prompt = build_rag_prompt(context="ctx", question="q")
        assert "CONTEXT:" in prompt
        assert "QUESTION:" in prompt

    def test_build_condense_prompt_contains_question(self):
        history = [
            Message(role="user", content="What is RAG?"),
            Message(role="assistant", content="RAG stands for Retrieval-Augmented Generation."),
        ]
        prompt = build_condense_prompt(history, "Tell me more about it")
        assert "Tell me more about it" in prompt
        assert "What is RAG?" in prompt

    def test_build_condense_prompt_empty_history(self):
        prompt = build_condense_prompt([], "standalone question")
        assert "standalone question" in prompt

    def test_format_chat_history_limits_turns(self):
        messages = [Message(role="user", content=f"q{i}") for i in range(20)]
        result = format_chat_history(messages, max_turns=3)
        assert len(result) <= 6

    def test_format_chat_history_returns_dicts(self):
        messages = [Message(role="user", content="hello")]
        result = format_chat_history(messages)
        assert isinstance(result[0], dict)
        assert "role" in result[0]
        assert "content" in result[0]

    def test_system_prompt_not_empty(self):
        assert len(SYSTEM_PROMPT) > 50

    def test_no_context_response_not_empty(self):
        assert len(NO_CONTEXT_RESPONSE) > 10


class TestLLMResponse:
    def test_total_tokens(self):
        r = LLMResponse(content="hi", model="m", input_tokens=100, output_tokens=50)
        assert r.total_tokens == 150

    def test_total_tokens_zero(self):
        r = LLMResponse(content="hi", model="m")
        assert r.total_tokens == 0


class TestLLMClient:
    def test_invalid_provider_raises(self):
        with pytest.raises(ValueError, match="Unsupported provider"):
            LLMClient(provider="fakeprovider")

    def test_default_anthropic_model(self):
        client = LLMClient(provider="anthropic")
        assert "claude" in client.model

    def test_default_openai_model(self):
        client = LLMClient(provider="openai")
        assert "gpt" in client.model

    def test_custom_model_respected(self):
        client = LLMClient(provider="anthropic", model="claude-3-haiku-20240307")
        assert client.model == "claude-3-haiku-20240307"

    def test_client_lazy_loaded(self):
        client = LLMClient(provider="anthropic")
        assert client._client is None

    def test_missing_api_key_raises(self):
        import os
        original = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            client = LLMClient(provider="anthropic")
            with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
                _ = client.client
        finally:
            if original:
                os.environ["ANTHROPIC_API_KEY"] = original


class TestRAGResponse:
    def test_source_files_deduped(self):
        results = [
            make_search_result("text a", chunk_id="c001"),
            make_search_result("text b", chunk_id="c002"),
        ]
        response = RAGResponse(
            answer="answer", question="q", standalone_question="q",
            sources=results, context="ctx",
            retrieval_time_ms=10.0, generation_time_ms=200.0, total_time_ms=210.0,
        )
        assert response.source_files == ["test.pdf"]

    def test_to_dict_has_required_keys(self):
        response = RAGResponse(
            answer="answer", question="q", standalone_question="q",
            sources=[], context="",
            retrieval_time_ms=10.0, generation_time_ms=200.0, total_time_ms=210.0,
        )
        d = response.to_dict()
        for key in ["question", "answer", "sources", "source_files", "retrieval_time_ms", "generation_time_ms"]:
            assert key in d


class TestRAGChain:
    def test_ask_returns_rag_response(self):
        retriever = make_mock_retriever()
        llm = make_mock_llm("The refund period is 30 days.")
        chain = RAGChain(retriever=retriever, llm_client=llm)
        response = chain.ask("What is the refund policy?")
        assert isinstance(response, RAGResponse)
        assert response.answer == "The refund period is 30 days."

    def test_ask_calls_retriever(self):
        retriever = make_mock_retriever()
        llm = make_mock_llm()
        chain = RAGChain(retriever=retriever, llm_client=llm)
        chain.ask("test question")
        retriever.retrieve.assert_called_once()

    def test_ask_calls_llm(self):
        retriever = make_mock_retriever()
        llm = make_mock_llm()
        chain = RAGChain(retriever=retriever, llm_client=llm)
        chain.ask("test question")
        llm.complete.assert_called_once()

    def test_memory_grows_with_turns(self):
        retriever = make_mock_retriever()
        llm = make_mock_llm()
        chain = RAGChain(retriever=retriever, llm_client=llm, use_conversation_memory=True)
        chain.ask("question 1")
        chain.ask("question 2")
        assert chain.turn_count == 2
        assert len(chain.history) == 4

    def test_clear_history_resets_memory(self):
        retriever = make_mock_retriever()
        llm = make_mock_llm()
        chain = RAGChain(retriever=retriever, llm_client=llm)
        chain.ask("question 1")
        chain.clear_history()
        assert chain.turn_count == 0
        assert len(chain.history) == 0

    def test_no_results_returns_no_context_response(self):
        retriever = make_mock_retriever(results=[])
        retriever.retrieve.return_value = []
        llm = make_mock_llm()
        chain = RAGChain(retriever=retriever, llm_client=llm)
        response = chain.ask("obscure question")
        assert NO_CONTEXT_RESPONSE in response.answer

    def test_response_has_timing(self):
        retriever = make_mock_retriever()
        llm = make_mock_llm()
        chain = RAGChain(retriever=retriever, llm_client=llm)
        response = chain.ask("test")
        assert response.retrieval_time_ms >= 0
        assert response.generation_time_ms >= 0
        assert response.total_time_ms >= 0

    def test_response_sources_populated(self):
        retriever = make_mock_retriever()
        llm = make_mock_llm()
        chain = RAGChain(retriever=retriever, llm_client=llm)
        response = chain.ask("test question")
        assert len(response.sources) > 0

    def test_stream_yields_tokens(self):
        retriever = make_mock_retriever()
        llm = make_mock_llm("streaming answer")
        chain = RAGChain(retriever=retriever, llm_client=llm)
        tokens = []
        for token, _ in chain.stream("question"):
            tokens.append(token)
        assert len(tokens) > 0
        assert "".join(tokens) == "streaming answer"

    def test_stream_no_results_yields_no_context(self):
        retriever = make_mock_retriever(results=[])
        retriever.retrieve.return_value = []
        llm = make_mock_llm()
        chain = RAGChain(retriever=retriever, llm_client=llm)
        tokens = []
        for token, _ in chain.stream("obscure question"):
            tokens.append(token)
        assert NO_CONTEXT_RESPONSE in "".join(tokens)

    def test_memory_disabled(self):
        retriever = make_mock_retriever()
        llm = make_mock_llm()
        chain = RAGChain(retriever=retriever, llm_client=llm, use_conversation_memory=False)
        chain.ask("question 1")
        assert chain.turn_count == 0

    def test_condense_skipped_on_first_turn(self):
        retriever = make_mock_retriever()
        llm = make_mock_llm()
        chain = RAGChain(retriever=retriever, llm_client=llm, condense_questions=True)
        chain.ask("first question")
        assert llm.complete.call_count == 1

    def test_history_property_returns_copy(self):
        retriever = make_mock_retriever()
        llm = make_mock_llm()
        chain = RAGChain(retriever=retriever, llm_client=llm)
        chain.ask("q1")
        history = chain.history
        history.clear()
        assert len(chain.history) > 0
