from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Message:
    role: str
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


SYSTEM_PROMPT = """You are a precise and reliable question-answering assistant. You answer questions strictly based on the provided context passages retrieved from the user's documents.

Rules you must follow:
1. Answer ONLY using information present in the provided context.
2. If the context does not contain enough information to answer, say: "I could not find relevant information in the provided documents to answer this question."
3. Always cite which source (e.g. [Source 1], [Source 2]) your answer draws from.
4. Never fabricate facts, statistics, names, or dates that are not in the context.
5. Keep answers concise, accurate, and well-structured.
6. If multiple sources support the answer, mention all of them."""


RAG_PROMPT_TEMPLATE = """Use the following retrieved context passages to answer the question.

CONTEXT:
{context}

QUESTION:
{question}

Provide a clear, accurate answer based solely on the context above. Cite sources using [Source N] notation."""


CONDENSE_QUESTION_TEMPLATE = """Given the conversation history below and a follow-up question, rewrite the follow-up question to be a standalone question that contains all necessary context to search for relevant documents.

Conversation history:
{chat_history}

Follow-up question: {question}

Standalone question:"""


NO_CONTEXT_RESPONSE = "I could not find relevant information in the provided documents to answer this question."


def build_rag_prompt(context: str, question: str) -> str:
    return RAG_PROMPT_TEMPLATE.format(context=context, question=question)


def build_condense_prompt(chat_history: List[Message], question: str) -> str:
    history_text = "\n".join(
        f"{m.role.capitalize()}: {m.content}" for m in chat_history
    )
    return CONDENSE_QUESTION_TEMPLATE.format(
        chat_history=history_text,
        question=question,
    )


def format_chat_history(messages: List[Message], max_turns: int = 6) -> List[dict]:
    recent = messages[-(max_turns * 2):]
    return [m.to_dict() for m in recent]
