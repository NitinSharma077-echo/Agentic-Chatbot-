import json
import time

import requests
import streamlit as st

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 4px;
    }
    .badge-green  { background:#d4edda; color:#155724; }
    .badge-yellow { background:#fff3cd; color:#856404; }
    .badge-red    { background:#f8d7da; color:#721c24; }
    .badge-blue   { background:#cce5ff; color:#004085; }
    .badge-gray   { background:#e2e3e5; color:#383d41; }
    .source-box {
        background: #f8f9fa;
        border-left: 3px solid #6c757d;
        padding: 8px 12px;
        margin: 6px 0;
        border-radius: 4px;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state defaults ─────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = "default-session"
if "api_key" not in st.session_state:
    st.session_state.api_key = "dev-secret-key-change-this"


# ── Helpers ───────────────────────────────────────────────────────────────────

def headers() -> dict:
    return {"X-API-Key": st.session_state.api_key}


def badge(text: str, color: str) -> str:
    return f'<span class="badge badge-{color}">{text}</span>'


def intent_color(intent: str) -> str:
    return {"qa": "blue", "summary": "green", "comparison": "yellow", "out_of_scope": "red", "chat": "blue"}.get(intent, "gray")


def confidence_color(conf: str) -> str:
    return {"high": "green", "medium": "yellow", "low": "red", "no_context": "red", "out_of_scope": "red"}.get(conf, "gray")


def guard_color(status: str) -> str:
    return {"passed": "green", "warning": "yellow", "blocked": "red", "not_checked": "gray"}.get(status, "gray")


def render_sources(sources: list) -> None:
    if not sources:
        return
    with st.expander(f"📄 {len(sources)} source(s) cited", expanded=False):
        for i, src in enumerate(sources, 1):
            score = src.get("similarity_score")
            score_str = f"{score:.4f}" if score is not None else "N/A"
            page = src.get("page_number") or "N/A"
            st.markdown(
                f'<div class="source-box">'
                f'<b>Source {i}</b> — {src.get("file_name","?")} '
                f'| Page: {page} | Score: {score_str}<br>'
                f'<small>{src.get("content_preview","")[:300]}</small>'
                f"</div>",
                unsafe_allow_html=True,
            )


def stream_chat(question: str, session_id: str, top_k: int, mode: str = "rag"):
    """
    Calls POST /chat/stream and yields SSE event dicts.
    """
    url = f"{BACKEND_URL}/chat/stream"
    payload = {"question": question, "session_id": session_id, "top_k": top_k, "mode": mode}

    with requests.post(url, json=payload, headers=headers(), stream=True, timeout=120) as resp:
        if resp.status_code != 200:
            raise RuntimeError(f"Backend error {resp.status_code}: {resp.text}")

        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data: "):
                continue
            try:
                event = json.loads(line[6:])
                yield event
            except json.JSONDecodeError:
                continue


def upload_document(file) -> dict:
    url = f"{BACKEND_URL}/documents/upload"
    resp = requests.post(url, files={"file": (file.name, file, file.type)}, headers=headers(), timeout=60)
    resp.raise_for_status()
    return resp.json()


def get_stats() -> dict:
    resp = requests.get(f"{BACKEND_URL}/documents/stats", headers=headers(), timeout=10)
    resp.raise_for_status()
    return resp.json()


def reset_vector_store() -> dict:
    resp = requests.delete(f"{BACKEND_URL}/documents/reset", headers=headers(), timeout=10)
    resp.raise_for_status()
    return resp.json()


def clear_memory(session_id: str) -> dict:
    resp = requests.delete(f"{BACKEND_URL}/chat/memory/{session_id}", headers=headers(), timeout=10)
    resp.raise_for_status()
    return resp.json()


def health_check() -> dict:
    resp = requests.get(f"{BACKEND_URL}/health/", timeout=5)
    resp.raise_for_status()
    return resp.json()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    # Backend health
    try:
        h = health_check()
        st.success(f"Backend: {h.get('status','?').upper()}")
    except Exception:
        st.error("Backend unreachable — start uvicorn first")

    st.divider()

    # API key & session
    st.session_state.api_key = st.text_input(
        "API Key", value=st.session_state.api_key, type="password"
    )
    st.session_state.session_id = st.text_input(
        "Session ID", value=st.session_state.session_id
    )
    mode = st.radio(
        "Chat mode",
        options=["rag", "chat"],
        format_func=lambda x: "📚 RAG — document search" if x == "rag" else "💬 General chat",
        horizontal=True,
    )
    top_k = st.slider("Top-K chunks", min_value=1, max_value=10, value=4, disabled=(mode == "chat"))

    st.divider()

    # Document upload
    st.subheader("📤 Upload Document")
    uploaded_file = st.file_uploader(
        "PDF, DOCX, TXT, MD (max 10 MB)",
        type=["pdf", "docx", "txt", "md"],
        label_visibility="collapsed",
    )
    if uploaded_file and st.button("Upload & Index", use_container_width=True):
        with st.spinner("Uploading and indexing..."):
            try:
                result = upload_document(uploaded_file)
                st.success(
                    f"Indexed {result['indexed_chunk_count']} chunks "
                    f"from **{result['original_file_name']}**"
                )
            except Exception as e:
                st.error(f"Upload failed: {e}")

    st.divider()

    # Vector store stats
    st.subheader("📊 Vector Store")
    if st.button("Refresh Stats", use_container_width=True):
        try:
            stats = get_stats()
            st.metric("Total chunks", stats["total_chunks"])
            st.metric("Unique documents", stats["unique_document_count"])
            if stats["documents"]:
                for doc in stats["documents"]:
                    st.markdown(f"- **{doc['file_name']}** ({doc['chunk_count']} chunks)")
        except Exception as e:
            st.error(f"Could not load stats: {e}")

    if st.button("Reset Vector Store", type="secondary", use_container_width=True):
        try:
            reset_vector_store()
            st.warning("Vector store cleared.")
        except Exception as e:
            st.error(f"Reset failed: {e}")

    st.divider()

    # Memory controls
    st.subheader("🧹 Memory")
    if st.button("Clear Chat Memory", use_container_width=True):
        try:
            clear_memory(st.session_state.session_id)
            st.session_state.messages = []
            st.success("Memory cleared.")
            st.rerun()
        except Exception as e:
            st.error(f"Clear failed: {e}")


# ── Main chat area ────────────────────────────────────────────────────────────

st.title("🤖 Enterprise RAG Knowledge Assistant")
st.caption("Ask questions about your uploaded company documents.")

# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            m = msg["meta"]
            st.markdown(
                badge(m.get("intent", "qa"), intent_color(m.get("intent", "qa")))
                + badge(m.get("confidence", "?"), confidence_color(m.get("confidence", "?")))
                + badge(m.get("guard_status", "?"), guard_color(m.get("guard_status", "?"))),
                unsafe_allow_html=True,
            )
            if m.get("was_rewritten"):
                st.caption(f"Rewritten query: *{m.get('rewritten_question', '')}*")
            render_sources(m.get("sources", []))

# Chat input
placeholder = "Ask a question about your documents..." if mode == "rag" else "Chat with the AI assistant..."
question = st.chat_input(placeholder)

if question:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Stream assistant response
    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        status_placeholder = st.empty()

        full_answer = ""
        meta = {}
        sources = []
        error_occurred = False

        try:
            status_placeholder.caption("Thinking...")

            for event in stream_chat(question, st.session_state.session_id, top_k, mode):
                etype = event.get("type")

                if etype == "start":
                    meta["intent"] = event.get("intent", mode)
                    meta["route_reason"] = event.get("route_reason", "")
                    meta["was_rewritten"] = event.get("was_rewritten", False)
                    meta["rewritten_question"] = event.get("rewritten_question", question)
                    label = "Generating..." if mode == "chat" else f"Intent: **{meta['intent']}** — retrieving documents..."
                    status_placeholder.caption(label)

                elif etype == "metadata":
                    sources = event.get("sources", [])
                    meta["confidence"] = event.get("confidence", "basic")
                    status_placeholder.caption(
                        f"Intent: **{meta['intent']}** | "
                        f"Confidence: **{meta['confidence']}** | "
                        f"Sources: {len(sources)} — generating..."
                    )

                elif etype == "chunk":
                    full_answer += event.get("content", "")
                    answer_placeholder.markdown(full_answer + "▌")

                elif etype == "correction":
                    full_answer = event.get("content", full_answer)
                    answer_placeholder.markdown(full_answer + "▌")

                elif etype == "done":
                    meta["guard_status"] = event.get("guard_status", "not_checked")
                    meta["guard_reason"] = event.get("guard_reason")
                    meta["sources"] = sources
                    answer_placeholder.markdown(full_answer)
                    status_placeholder.empty()

                elif etype == "error":
                    full_answer = event.get("content", "An error occurred.")
                    answer_placeholder.markdown(full_answer)
                    status_placeholder.empty()
                    error_occurred = True
                    break

        except Exception as e:
            full_answer = f"Connection error: {e}"
            answer_placeholder.error(full_answer)
            error_occurred = True

        if not error_occurred and meta:
            st.markdown(
                badge(meta.get("intent", "qa"), intent_color(meta.get("intent", "qa")))
                + badge(meta.get("confidence", "?"), confidence_color(meta.get("confidence", "?")))
                + badge(meta.get("guard_status", "?"), guard_color(meta.get("guard_status", "?"))),
                unsafe_allow_html=True,
            )
            if meta.get("was_rewritten"):
                st.caption(f"Rewritten query: *{meta.get('rewritten_question', '')}*")
            render_sources(sources)

    # Save to chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer,
        "meta": meta,
    })
