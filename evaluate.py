import sys
import argparse
from pathlib import Path
from datetime import datetime

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
)
from src.embeddings.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.llm.client import LLMClient
from src.pipeline.rag_chain import RAGChain
from src.evaluation.metrics import RAGEvaluator, load_eval_samples


def main():
    parser = argparse.ArgumentParser(
        description="RAG QA System — Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--samples", type=str, default="data/eval_samples.json",
                        help="Path to evaluation samples JSON file")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save evaluation report JSON")
    parser.add_argument("--provider", type=str, default=LLM_PROVIDER)
    parser.add_argument("--model", type=str, default=LLM_MODEL)
    parser.add_argument("--top-k", type=int, default=TOP_K_RETRIEVAL)
    args = parser.parse_args()

    print(f"\n{'='*52}")
    print(f"  RAG QA System — Evaluation")
    print(f"{'='*52}")
    print(f"  Samples  : {args.samples}")
    print(f"  Provider : {args.provider} / {args.model}")
    print(f"  Top-K    : {args.top_k}")
    print(f"{'='*52}\n")

    print("[1/3] Loading index...")
    store = VectorStore(
        embedding_model=EMBEDDING_MODEL,
        index_path=FAISS_INDEX_FILE,
        metadata_path=FAISS_METADATA_FILE,
    )
    store.load()

    retriever = Retriever(vector_store=store, top_k=args.top_k, score_threshold=0.0)
    llm = LLMClient(provider=args.provider, model=args.model,
                    temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS)
    chain = RAGChain(retriever=retriever, llm_client=llm, top_k=args.top_k,
                     use_conversation_memory=False)

    print("[2/3] Loading evaluation samples...")
    samples = load_eval_samples(args.samples)
    print(f"  Loaded {len(samples)} sample(s)\n")

    print("[3/3] Running evaluation...")
    evaluator = RAGEvaluator(rag_chain=chain)
    report = evaluator.evaluate(samples)
    report.print_summary()

    output_path = args.output or f"eval_report_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.json"
    report.save(output_path)


if __name__ == "__main__":
    main()
