import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class EvalSample:
    question: str
    expected_answer: str
    context_docs: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "expected_answer": self.expected_answer,
            "context_docs": self.context_docs,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EvalSample":
        return cls(
            question=d["question"],
            expected_answer=d.get("expected_answer", ""),
            context_docs=d.get("context_docs", []),
        )


@dataclass
class EvalResult:
    question: str
    expected_answer: str
    generated_answer: str
    retrieved_contexts: List[str]
    faithfulness_score: float
    answer_relevance_score: float
    context_recall_score: float
    retrieval_time_ms: float
    generation_time_ms: float

    @property
    def overall_score(self) -> float:
        return round(
            (self.faithfulness_score + self.answer_relevance_score + self.context_recall_score) / 3,
            4,
        )

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "expected_answer": self.expected_answer,
            "generated_answer": self.generated_answer,
            "faithfulness_score": self.faithfulness_score,
            "answer_relevance_score": self.answer_relevance_score,
            "context_recall_score": self.context_recall_score,
            "overall_score": self.overall_score,
            "retrieval_time_ms": round(self.retrieval_time_ms, 2),
            "generation_time_ms": round(self.generation_time_ms, 2),
        }


@dataclass
class EvalReport:
    results: List[EvalResult]
    model: str
    embedding_model: str
    top_k: int
    evaluated_at: str = ""

    @property
    def avg_faithfulness(self) -> float:
        if not self.results:
            return 0.0
        return round(sum(r.faithfulness_score for r in self.results) / len(self.results), 4)

    @property
    def avg_answer_relevance(self) -> float:
        if not self.results:
            return 0.0
        return round(sum(r.answer_relevance_score for r in self.results) / len(self.results), 4)

    @property
    def avg_context_recall(self) -> float:
        if not self.results:
            return 0.0
        return round(sum(r.context_recall_score for r in self.results) / len(self.results), 4)

    @property
    def avg_overall(self) -> float:
        if not self.results:
            return 0.0
        return round(sum(r.overall_score for r in self.results) / len(self.results), 4)

    @property
    def avg_retrieval_ms(self) -> float:
        if not self.results:
            return 0.0
        return round(sum(r.retrieval_time_ms for r in self.results) / len(self.results), 2)

    @property
    def avg_generation_ms(self) -> float:
        if not self.results:
            return 0.0
        return round(sum(r.generation_time_ms for r in self.results) / len(self.results), 2)

    def to_dict(self) -> dict:
        return {
            "summary": {
                "model": self.model,
                "embedding_model": self.embedding_model,
                "top_k": self.top_k,
                "evaluated_at": self.evaluated_at,
                "total_questions": len(self.results),
                "avg_faithfulness": self.avg_faithfulness,
                "avg_answer_relevance": self.avg_answer_relevance,
                "avg_context_recall": self.avg_context_recall,
                "avg_overall_score": self.avg_overall,
                "avg_retrieval_ms": self.avg_retrieval_ms,
                "avg_generation_ms": self.avg_generation_ms,
            },
            "results": [r.to_dict() for r in self.results],
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"  Evaluation report saved to {path}")

    def print_summary(self) -> None:
        print(f"\n{'='*52}")
        print(f"  Evaluation Summary")
        print(f"{'='*52}")
        print(f"  Model             : {self.model}")
        print(f"  Embedding model   : {self.embedding_model}")
        print(f"  Top-K             : {self.top_k}")
        print(f"  Questions         : {len(self.results)}")
        print(f"{'─'*52}")
        print(f"  Faithfulness      : {self.avg_faithfulness:.4f}")
        print(f"  Answer Relevance  : {self.avg_answer_relevance:.4f}")
        print(f"  Context Recall    : {self.avg_context_recall:.4f}")
        print(f"  Overall Score     : {self.avg_overall:.4f}")
        print(f"{'─'*52}")
        print(f"  Avg retrieval     : {self.avg_retrieval_ms:.0f}ms")
        print(f"  Avg generation    : {self.avg_generation_ms:.0f}ms")
        print(f"{'='*52}\n")


class RAGEvaluator:
    def __init__(self, rag_chain, llm_judge=None):
        self.chain = rag_chain
        self.llm_judge = llm_judge or rag_chain.llm

    def evaluate(self, samples: List[EvalSample]) -> EvalReport:
        from datetime import datetime

        print(f"\n  Evaluating {len(samples)} question(s)...\n")
        results: List[EvalResult] = []

        for i, sample in enumerate(samples, 1):
            print(f"  [{i}/{len(samples)}] {sample.question[:60]}...")
            self.chain.clear_history()

            response = self.chain.ask(sample.question)

            retrieved_contexts = [r.chunk.content for r in response.sources]

            faithfulness = self._score_faithfulness(
                answer=response.answer,
                contexts=retrieved_contexts,
            )
            relevance = self._score_answer_relevance(
                question=sample.question,
                answer=response.answer,
            )
            recall = self._score_context_recall(
                expected_answer=sample.expected_answer,
                contexts=retrieved_contexts,
            )

            results.append(
                EvalResult(
                    question=sample.question,
                    expected_answer=sample.expected_answer,
                    generated_answer=response.answer,
                    retrieved_contexts=retrieved_contexts,
                    faithfulness_score=faithfulness,
                    answer_relevance_score=relevance,
                    context_recall_score=recall,
                    retrieval_time_ms=response.retrieval_time_ms,
                    generation_time_ms=response.generation_time_ms,
                )
            )

            print(f"         faithfulness={faithfulness:.3f} relevance={relevance:.3f} recall={recall:.3f}")

        return EvalReport(
            results=results,
            model=self.chain.llm.model,
            embedding_model=self.chain.retriever.vector_store.embedding_model_name,
            top_k=self.chain.top_k,
            evaluated_at=datetime.utcnow().isoformat(),
        )

    def _score_faithfulness(self, answer: str, contexts: List[str]) -> float:
        if not contexts or not answer.strip():
            return 0.0

        context_text = "\n\n".join(contexts)
        prompt = (
            f"You are an evaluation judge. Given the context passages and an answer, "
            f"score how faithful the answer is to the context on a scale of 0.0 to 1.0.\n\n"
            f"A score of 1.0 means every claim in the answer is directly supported by the context.\n"
            f"A score of 0.0 means the answer contains claims not present in the context.\n\n"
            f"CONTEXT:\n{context_text}\n\n"
            f"ANSWER:\n{answer}\n\n"
            f"Respond with ONLY a decimal number between 0.0 and 1.0. Nothing else."
        )
        return self._get_score(prompt)

    def _score_answer_relevance(self, question: str, answer: str) -> float:
        if not answer.strip():
            return 0.0

        prompt = (
            f"You are an evaluation judge. Score how relevant this answer is to the question "
            f"on a scale of 0.0 to 1.0.\n\n"
            f"A score of 1.0 means the answer directly and completely addresses the question.\n"
            f"A score of 0.0 means the answer is completely off-topic.\n\n"
            f"QUESTION:\n{question}\n\n"
            f"ANSWER:\n{answer}\n\n"
            f"Respond with ONLY a decimal number between 0.0 and 1.0. Nothing else."
        )
        return self._get_score(prompt)

    def _score_context_recall(self, expected_answer: str, contexts: List[str]) -> float:
        if not contexts or not expected_answer.strip():
            return 0.0

        context_text = "\n\n".join(contexts)
        prompt = (
            f"You are an evaluation judge. Given the expected answer and retrieved context passages, "
            f"score how much of the expected answer's information is present in the context "
            f"on a scale of 0.0 to 1.0.\n\n"
            f"A score of 1.0 means all information needed to produce the expected answer is in the context.\n"
            f"A score of 0.0 means none of the relevant information was retrieved.\n\n"
            f"EXPECTED ANSWER:\n{expected_answer}\n\n"
            f"RETRIEVED CONTEXT:\n{context_text}\n\n"
            f"Respond with ONLY a decimal number between 0.0 and 1.0. Nothing else."
        )
        return self._get_score(prompt)

    def _get_score(self, prompt: str) -> float:
        try:
            response = self.llm_judge.complete(user_message=prompt)
            text = response.content.strip()
            score = float(text.split()[0].rstrip(".,"))
            return max(0.0, min(1.0, score))
        except Exception:
            return 0.5


def load_eval_samples(path: str | Path) -> List[EvalSample]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [EvalSample.from_dict(d) for d in data]
