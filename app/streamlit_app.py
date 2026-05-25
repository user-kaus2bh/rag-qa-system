import sys
import time
import glob
import tempfile
import shutil
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    VECTORSTORE_DIR,
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
    SUPPORTED_EXTENSIONS,
)

st.set_page_config(
    page_title="RAG-QA System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300&display=swap');

:root {
    --bg: #0f1117;
    --surface: #161b27;
    --surface2: #1e2535;
    --border: #2a3347;
    --accent: #4f8ef7;
    --accent2: #7c5cbf;
    --success: #34c98a;
    --warning: #f5a623;
    --text: #e8eaf0;
    --text-muted: #7a8499;
    --user-bubble: #1a2744;
    --ai-bubble: #161b27;
    --radius: 12px;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.stApp { background-color: var(--bg); }

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding: 1.5rem 2rem 2rem 2rem;
    max-width: 1400px;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

.sidebar-logo {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: var(--text);
    letter-spacing: -0.02em;
    margin-bottom: 0.2rem;
}
.sidebar-tagline {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 1.5rem;
}

/* ── Stat cards ── */
.stat-row { display: flex; gap: 0.6rem; margin-bottom: 1.2rem; }
.stat-card {
    flex: 1;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0.7rem 0.8rem;
    text-align: center;
}
.stat-num {
    font-family: 'DM Mono', monospace;
    font-size: 1.3rem;
    font-weight: 500;
    color: var(--accent);
    line-height: 1;
}
.stat-label {
    font-size: 0.65rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.3rem;
}

/* ── Header area ── */
.page-header {
    display: flex;
    align-items: baseline;
    gap: 1rem;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--border);
}
.page-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: var(--text);
    letter-spacing: -0.03em;
    line-height: 1;
}
.page-subtitle {
    font-size: 0.82rem;
    color: var(--text-muted);
}

/* ── Status badge ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.25rem 0.75rem;
    border-radius: 100px;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.status-ready {
    background: rgba(52, 201, 138, 0.12);
    color: var(--success);
    border: 1px solid rgba(52, 201, 138, 0.25);
}
.status-notready {
    background: rgba(245, 166, 35, 0.12);
    color: var(--warning);
    border: 1px solid rgba(245, 166, 35, 0.25);
}

/* ── Chat messages ── */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 1.2rem;
    padding-bottom: 1rem;
}

.message-user {
    display: flex;
    justify-content: flex-end;
    gap: 0.75rem;
    align-items: flex-start;
}
.message-ai {
    display: flex;
    justify-content: flex-start;
    gap: 0.75rem;
    align-items: flex-start;
}

.bubble {
    max-width: 70%;
    padding: 0.9rem 1.1rem;
    border-radius: var(--radius);
    line-height: 1.65;
    font-size: 0.9rem;
}
.bubble-user {
    background: var(--user-bubble);
    border: 1px solid rgba(79, 142, 247, 0.2);
    border-bottom-right-radius: 4px;
    color: var(--text);
}
.bubble-ai {
    background: var(--ai-bubble);
    border: 1px solid var(--border);
    border-bottom-left-radius: 4px;
    color: var(--text);
}

.avatar {
    width: 30px;
    height: 30px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    flex-shrink: 0;
    margin-top: 2px;
}
.avatar-user {
    background: rgba(79, 142, 247, 0.15);
    border: 1px solid rgba(79, 142, 247, 0.3);
}
.avatar-ai {
    background: rgba(124, 92, 191, 0.15);
    border: 1px solid rgba(124, 92, 191, 0.3);
}

/* ── Source citations ── */
.sources-container {
    margin-top: 0.7rem;
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
}
.source-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.2rem 0.6rem;
    background: rgba(79, 142, 247, 0.08);
    border: 1px solid rgba(79, 142, 247, 0.2);
    border-radius: 100px;
    font-size: 0.68rem;
    font-family: 'DM Mono', monospace;
    color: var(--accent);
    cursor: default;
}

/* ── Timing bar ── */
.timing-bar {
    display: flex;
    gap: 1rem;
    margin-top: 0.6rem;
    padding-top: 0.6rem;
    border-top: 1px solid var(--border);
}
.timing-item {
    display: flex;
    align-items: center;
    gap: 0.3rem;
    font-size: 0.68rem;
    font-family: 'DM Mono', monospace;
    color: var(--text-muted);
}
.timing-dot {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: var(--accent);
}

/* ── Upload zone ── */
.upload-hint {
    text-align: center;
    padding: 1.5rem;
    border: 2px dashed var(--border);
    border-radius: var(--radius);
    color: var(--text-muted);
    font-size: 0.82rem;
    margin-bottom: 1rem;
}

/* ── Streamlit overrides ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(79, 142, 247, 0.15) !important;
}

.stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

.stButton > button[kind="secondary"] {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
}

div[data-testid="stFileUploader"] {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0.5rem;
}

.stSlider > div { color: var(--text) !important; }
.stSelectbox > div > div {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

div[data-testid="stExpander"] {
    background: var(--surface2);
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

.stSpinner > div { border-top-color: var(--accent) !important; }

hr { border-color: var(--border) !important; }

.stAlert {
    background: var(--surface2) !important;
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
}

p, li, span { color: var(--text); }
label { color: var(--text-muted) !important; font-size: 0.78rem !important; }

.section-heading {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-muted);
    margin: 1.2rem 0 0.6rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border);
}
</style>
""", unsafe_allow_html=True)


def init_session_state():
    defaults = {
        "messages": [],
        "chain": None,
        "index_loaded": False,
        "total_chunks": 0,
        "total_docs": 0,
        "top_k": TOP_K_RETRIEVAL,
        "temperature": LLM_TEMPERATURE,
        "provider": LLM_PROVIDER,
        "model": LLM_MODEL,
        "processing": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def check_index_exists() -> bool:
    return FAISS_INDEX_FILE.exists() and FAISS_METADATA_FILE.exists()


@st.cache_resource(show_spinner=False)
def load_chain(provider: str, model: str, temperature: float, top_k: int):
    from src.embeddings.vector_store import VectorStore
    from src.retrieval.retriever import Retriever
    from src.llm.client import LLMClient
    from src.pipeline.rag_chain import RAGChain

    store = VectorStore(
        embedding_model=EMBEDDING_MODEL,
        index_path=FAISS_INDEX_FILE,
        metadata_path=FAISS_METADATA_FILE,
    )
    store.load()

    retriever = Retriever(
        vector_store=store,
        top_k=top_k,
        score_threshold=0.0,
        deduplicate=True,
    )

    llm = LLMClient(
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=LLM_MAX_TOKENS,
    )

    chain = RAGChain(
        retriever=retriever,
        llm_client=llm,
        top_k=top_k,
        use_conversation_memory=True,
        max_history_turns=CONTEXT_WINDOW_TURNS,
        condense_questions=True,
    )

    return chain, store.total_vectors, store.total_chunks


def run_ingestion_and_indexing(uploaded_files) -> bool:
    from src.ingestion.loaders import DocumentLoader
    from src.ingestion.chunker import TextChunker
    from src.embeddings.vector_store import VectorStore

    try:
        DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
        DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for f in uploaded_files:
            dest = DATA_RAW_DIR / f.name
            with open(dest, "wb") as out:
                out.write(f.getbuffer())
            saved_paths.append(dest)

        loader = DocumentLoader(supported_extensions=SUPPORTED_EXTENSIONS)
        chunker = TextChunker()

        all_docs = []
        for path in saved_paths:
            docs = loader.load(path)
            all_docs.extend(docs)

        if not all_docs:
            st.error("No text could be extracted from the uploaded files.")
            return False

        chunks = chunker.chunk_documents(all_docs)

        from datetime import datetime
        out_path = DATA_PROCESSED_DIR / f"chunks_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.json"
        chunker.save_chunks(chunks, out_path)

        store = VectorStore(
            embedding_model=EMBEDDING_MODEL,
            index_path=FAISS_INDEX_FILE,
            metadata_path=FAISS_METADATA_FILE,
        )
        store.build(chunks)
        store.save()

        st.session_state.total_chunks = len(chunks)
        st.session_state.total_docs = len(saved_paths)
        st.session_state.index_loaded = True

        load_chain.clear()
        return True

    except Exception as e:
        st.error(f"Error during processing: {e}")
        return False


def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="sidebar-logo">🔍 RAG-QA</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-tagline">Retrieval-Augmented Generation</div>', unsafe_allow_html=True)

        index_ready = check_index_exists()
        if index_ready:
            st.markdown('<span class="status-badge status-ready">● Index Ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-notready">● No Index</span>', unsafe_allow_html=True)

        if st.session_state.total_chunks > 0:
            st.markdown(f"""
            <div class="stat-row">
                <div class="stat-card">
                    <div class="stat-num">{st.session_state.total_docs}</div>
                    <div class="stat-label">Docs</div>
                </div>
                <div class="stat-card">
                    <div class="stat-num">{st.session_state.total_chunks}</div>
                    <div class="stat-label">Chunks</div>
                </div>
                <div class="stat-card">
                    <div class="stat-num">{len(st.session_state.messages) // 2}</div>
                    <div class="stat-label">Turns</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-heading">Upload Documents</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Drop files here",
            type=["pdf", "txt", "docx", "md"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded:
            if st.button("Process & Index", use_container_width=True):
                with st.spinner("Ingesting and building index..."):
                    success = run_ingestion_and_indexing(uploaded)
                if success:
                    st.success(f"Indexed {st.session_state.total_chunks} chunks from {st.session_state.total_docs} file(s)")
                    st.rerun()

        st.markdown('<div class="section-heading">Model Settings</div>', unsafe_allow_html=True)

        provider = st.selectbox(
            "Provider",
            ["gemini", "anthropic", "openai"],
            index=["gemini", "anthropic", "openai"].index(st.session_state.provider)
            if st.session_state.provider in ["gemini", "anthropic", "openai"] else 0,
            label_visibility="visible",
        )

        model_defaults = {
            "gemini": "gemini-2.0-flash",
            "anthropic": "claude-3-5-sonnet-20241022",
            "openai": "gpt-4o-mini",
        }

        model = st.text_input(
            "Model",
            value=model_defaults.get(provider, st.session_state.model),
        )

        top_k = st.slider("Sources to retrieve (Top-K)", 1, 10, st.session_state.top_k)
        temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.temperature, step=0.05)

        if (provider != st.session_state.provider or
                model != st.session_state.model or
                top_k != st.session_state.top_k or
                temperature != st.session_state.temperature):
            st.session_state.provider = provider
            st.session_state.model = model
            st.session_state.top_k = top_k
            st.session_state.temperature = temperature
            load_chain.clear()

        st.markdown('<div class="section-heading">Session</div>', unsafe_allow_html=True)

        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.chain:
                st.session_state.chain.clear_history()
            st.rerun()


def render_message(role: str, content: str, sources=None, timing=None):
    if role == "user":
        st.markdown(f"""
        <div class="message-user">
            <div class="bubble bubble-user">{content}</div>
            <div class="avatar avatar-user">👤</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        sources_html = ""
        if sources:
            chips = ""
            for s in sources:
                fname = s.chunk.metadata.get("filename", s.chunk.source.split("/")[-1])
                page = f" p.{s.chunk.page}" if s.chunk.page else ""
                chips += f'<span class="source-chip">📄 {fname}{page} · {s.score:.2f}</span>'
            sources_html = f'<div class="sources-container">{chips}</div>'

        timing_html = ""
        if timing:
            timing_html = f"""
            <div class="timing-bar">
                <div class="timing-item"><div class="timing-dot"></div>retrieval {timing.get("retrieval", 0):.0f}ms</div>
                <div class="timing-item"><div class="timing-dot"></div>generation {timing.get("generation", 0):.0f}ms</div>
                <div class="timing-item"><div class="timing-dot"></div>{timing.get("input_tokens", 0)}+{timing.get("output_tokens", 0)} tokens</div>
            </div>
            """

        st.markdown(f"""
        <div class="message-ai">
            <div class="avatar avatar-ai">🤖</div>
            <div class="bubble bubble-ai">
                {content}
                {sources_html}
                {timing_html}
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_chat():
    col_chat, col_context = st.columns([3, 1])

    with col_chat:
        st.markdown("""
        <div class="page-header">
            <div class="page-title">Document Q&A</div>
            <div class="page-subtitle">Ask anything about your uploaded documents</div>
        </div>
        """, unsafe_allow_html=True)

        if not check_index_exists():
            st.markdown("""
            <div class="upload-hint">
                📂 No documents indexed yet.<br>
                Upload files using the sidebar to get started.
            </div>
            """, unsafe_allow_html=True)
            return

        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            render_message(
                role=msg["role"],
                content=msg["content"],
                sources=msg.get("sources"),
                timing=msg.get("timing"),
            )
        st.markdown('</div>', unsafe_allow_html=True)

        with st.container():
            question = st.chat_input("Ask a question about your documents...")

        if question:
            st.session_state.messages.append({"role": "user", "content": question})

            try:
                chain, vectors, chunks = load_chain(
                    provider=st.session_state.provider,
                    model=st.session_state.model,
                    temperature=st.session_state.temperature,
                    top_k=st.session_state.top_k,
                )
                st.session_state.total_chunks = chunks

                with st.spinner("Thinking..."):
                    response = chain.ask(question)

                timing = {
                    "retrieval": response.retrieval_time_ms,
                    "generation": response.generation_time_ms,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                }

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.answer,
                    "sources": response.sources,
                    "timing": timing,
                })

            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Error: {str(e)}",
                    "sources": None,
                    "timing": None,
                })

            st.rerun()

    with col_context:
        if st.session_state.messages:
            last_ai = next(
                (m for m in reversed(st.session_state.messages) if m["role"] == "assistant"),
                None
            )
            if last_ai and last_ai.get("sources"):
                st.markdown('<div class="section-heading">Retrieved Sources</div>', unsafe_allow_html=True)
                for s in last_ai["sources"]:
                    fname = s.chunk.metadata.get("filename", s.chunk.source.split("/")[-1])
                    page = f" — Page {s.chunk.page}" if s.chunk.page else ""
                    with st.expander(f"[{s.rank}] {fname}{page} · {s.score:.3f}"):
                        st.markdown(f"""
                        <div style="font-size:0.8rem; color: var(--text-muted); font-family: 'DM Mono', monospace; line-height: 1.6; white-space: pre-wrap;">{s.chunk.content}</div>
                        """, unsafe_allow_html=True)


def main():
    init_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
