import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    FAISS_INDEX_FILE,
    FAISS_METADATA_FILE,
    EMBEDDING_MODEL,
    LLM_PROVIDER,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    TOP_K_RETRIEVAL,
    SIMILARITY_THRESHOLD,
    CONTEXT_WINDOW_TURNS,
)
from src.embeddings.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.llm.client import LLMClient
from src.pipeline.rag_chain import RAGChain


def build_chain(
    index_path: Path,
    metadata_path: Path,
    provider: str,
    model: str,
    top_k: int,
    temperature: float,
) -> RAGChain:
    print(f"\n{'='*52}")
    print(f"  RAG QA System — Interactive CLI")
    print(f"{'='*52}")
    print(f"  LLM      : {provider} / {model}")
    print(f"  Top-K    : {top_k}")
    print(f"  Index    : {index_path}")
    print(f"{'='*52}\n")

    print("[1/2] Loading vector index...")
    store = VectorStore(
        embedding_model=EMBEDDING_MODEL,
        index_path=index_path,
        metadata_path=metadata_path,
    )
    store.load()
    print(f"  {store.total_vectors} vectors loaded\n")

    print("[2/2] Initialising LLM client...")
    llm = LLMClient(
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=LLM_MAX_TOKENS,
    )
    print(f"  Model: {llm.model}\n")

    retriever = Retriever(
        vector_store=store,
        top_k=top_k,
        score_threshold=SIMILARITY_THRESHOLD,
    )

    chain = RAGChain(
        retriever=retriever,
        llm_client=llm,
        top_k=top_k,
        use_conversation_memory=True,
        max_history_turns=CONTEXT_WINDOW_TURNS,
        condense_questions=True,
    )

    return chain


def run_single(chain: RAGChain, question: str, json_out: bool) -> None:
    print(f"\nQuestion: {question}\n")
    print("Answer:")
    print("-" * 48)

    response = chain.ask(question)
    print(response.answer)

    print("-" * 48)
    print(f"\nSources used:")
    for s in response.sources:
        fname = s.chunk.metadata.get("filename", s.chunk.source)
        page = f" | page {s.chunk.page}" if s.chunk.page else ""
        print(f"  [{s.rank}] {fname}{page} (score: {s.score:.4f})")

    print(f"\nTiming: retrieval={response.retrieval_time_ms:.0f}ms | generation={response.generation_time_ms:.0f}ms | total={response.total_time_ms:.0f}ms")
    print(f"Tokens: input={response.input_tokens} | output={response.output_tokens}")

    if json_out:
        out_path = Path("query_result.json")
        with open(out_path, "w") as f:
            json.dump(response.to_dict(), f, indent=2)
        print(f"\nFull result saved to {out_path}")


def run_interactive(chain: RAGChain) -> None:
    print("\nReady. Type your question and press Enter.")
    print("Commands: 'clear' to reset memory | 'history' to show turns | 'quit' to exit\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue

        if question.lower() == "quit":
            print("Goodbye.")
            break

        if question.lower() == "clear":
            chain.clear_history()
            print("Conversation memory cleared.\n")
            continue

        if question.lower() == "history":
            if not chain.history:
                print("No conversation history yet.\n")
            else:
                for i, m in enumerate(chain.history):
                    label = "You" if m.role == "user" else "Assistant"
                    preview = m.content[:120].replace("\n", " ")
                    print(f"  [{i+1}] {label}: {preview}")
                print()
            continue

        print("\nAssistant: ", end="", flush=True)
        for token, _ in chain.stream(question):
            print(token, end="", flush=True)
        print()

        sources = chain.retriever.retrieve(question, top_k=3)
        if sources:
            print("\nSources:")
            for s in sources:
                fname = s.chunk.metadata.get("filename", s.chunk.source)
                page = f" | page {s.chunk.page}" if s.chunk.page else ""
                print(f"  [{s.rank}] {fname}{page} (score: {s.score:.4f})")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="RAG QA System — Query CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--question", "-q", type=str, default=None,
                        help="Single question to answer (non-interactive mode)")
    parser.add_argument("--index", type=str, default=str(FAISS_INDEX_FILE),
                        help="Path to FAISS index file")
    parser.add_argument("--metadata", type=str, default=str(FAISS_METADATA_FILE),
                        help="Path to metadata JSON file")
    parser.add_argument("--provider", type=str, default=LLM_PROVIDER,
                        choices=["anthropic", "openai"], help="LLM provider")
    parser.add_argument("--model", type=str, default=LLM_MODEL,
                        help="LLM model name")
    parser.add_argument("--top-k", type=int, default=TOP_K_RETRIEVAL,
                        help="Number of chunks to retrieve")
    parser.add_argument("--temperature", type=float, default=LLM_TEMPERATURE,
                        help="LLM temperature")
    parser.add_argument("--json", action="store_true",
                        help="Save result as JSON (only in single-question mode)")

    args = parser.parse_args()

    chain = build_chain(
        index_path=Path(args.index),
        metadata_path=Path(args.metadata),
        provider=args.provider,
        model=args.model,
        top_k=args.top_k,
        temperature=args.temperature,
    )

    if args.question:
        run_single(chain, args.question, json_out=args.json)
    else:
        run_interactive(chain)


if __name__ == "__main__":
    main()
