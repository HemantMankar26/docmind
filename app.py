import streamlit as st
from src.loader import load_and_chunk_pdfs
from src.embedder import build_vectorstore, load_vectorstore
from src.chain import build_chain
import os
import shutil

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind – AI Document Chatbot",
    page_icon="🧠",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; color: #ffffff; }
    .chat-bubble-user {
        background: #1e3a5f;
        border-radius: 12px;
        padding: 12px 16px;
        margin: 8px 0;
        text-align: right;
    }
    .chat-bubble-bot {
        background: #1a1f2e;
        border-left: 3px solid #4f8ef7;
        border-radius: 12px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .source-box {
        background: #12161f;
        border: 1px solid #2a2f3e;
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 0.8em;
        color: #8892a4;
        margin-top: 6px;
    }
    .stButton>button {
        background: #4f8ef7;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 600;
    }
    .stButton>button:hover { background: #3a7ae0; }
    h1 { color: #4f8ef7 !important; }
    .upload-section {
        background: #1a1f2e;
        border-radius: 12px;
        padding: 20px;
        border: 1px dashed #2a3a5e;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = False

# ── Header ───────────────────────────────────────────────────
st.markdown("# 🧠 DocMind")
st.markdown("##### AI-powered document chatbot — upload PDFs, ask anything.")
st.divider()

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Setup")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get a free key at console.groq.com"
    )

    st.markdown("---")
    st.markdown("## 📂 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files and api_key:
        if st.button("🚀 Process Documents", use_container_width=True):
            with st.spinner("Reading and indexing your documents..."):
                # Save uploads
                upload_dir = "uploads"
                if os.path.exists(upload_dir):
                    shutil.rmtree(upload_dir)
                os.makedirs(upload_dir)

                for f in uploaded_files:
                    with open(os.path.join(upload_dir, f.name), "wb") as out:
                        out.write(f.read())

                # Load, chunk, embed
                chunks = load_and_chunk_pdfs(upload_dir)
                vectorstore = build_vectorstore(chunks)
                st.session_state.chain = build_chain(vectorstore, api_key)
                st.session_state.docs_loaded = True
                st.session_state.chat_history = []

            st.success(f"✅ {len(uploaded_files)} doc(s) indexed — {len(chunks)} chunks")

    elif uploaded_files and not api_key:
        st.warning("⚠️ Enter your Groq API key first.")

    st.markdown("---")

    if st.session_state.docs_loaded:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("---")
    st.markdown("""
    **How it works:**
    1. Add your Groq API key
    2. Upload PDF documents
    3. Click Process
    4. Ask anything!

    ---
    **Stack:** LangChain • FAISS • HuggingFace • Groq • Streamlit

    Built by [Hemant Mankar](https://linkedin.com/in/hemant-mankar-3204a7225)
    """)

# ── Main chat area ───────────────────────────────────────────
if not st.session_state.docs_loaded:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align:center; padding: 60px 0; color: #555;'>
            <div style='font-size: 4em;'>📄</div>
            <h3 style='color: #4f8ef7;'>No documents loaded yet</h3>
            <p>Add your Groq API key and upload PDFs from the sidebar to get started.</p>
            <br>
            <p style='font-size:0.85em; color:#444;'>
                Supports multiple PDFs • Remembers conversation context • Cites sources
            </p>
        </div>
        """, unsafe_allow_html=True)
else:
    # Chat history display
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""
            <div style='text-align:center; padding:30px; color:#555;'>
                <p>✅ Documents ready! Ask your first question below.</p>
            </div>
            """, unsafe_allow_html=True)

        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class='chat-bubble-user'>
                    <strong>You</strong><br>{msg['content']}
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='chat-bubble-bot'>
                    <strong>🧠 DocMind</strong><br>{msg['content']}
                </div>""", unsafe_allow_html=True)

                if msg.get("sources"):
                    with st.expander("📎 View Sources", expanded=False):
                        for i, src in enumerate(msg["sources"], 1):
                            st.markdown(f"""
                            <div class='source-box'>
                                <strong>Source {i}:</strong> {src['source']} — Page {src['page']}<br>
                                <em>{src['snippet']}</em>
                            </div>""", unsafe_allow_html=True)

    st.divider()

    # Input
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            "Ask a question",
            placeholder="What is this document about? Summarize key points...",
            label_visibility="collapsed",
            key="user_input"
        )
    with col2:
        send = st.button("Send ➤", use_container_width=True)

    if send and user_input.strip():
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

        with st.spinner("Thinking..."):
            result = st.session_state.chain.invoke({
                "question": user_input,
                "chat_history": [
                    (m["content"], st.session_state.chat_history[i+1]["content"])
                    for i, m in enumerate(st.session_state.chat_history[:-1])
                    if m["role"] == "user" and i + 1 < len(st.session_state.chat_history)
                ]
            })

            answer = result.get("answer", "I couldn't find an answer in the documents.")
            source_docs = result.get("source_documents", [])

            sources = []
            seen = set()
            for doc in source_docs:
                key = (doc.metadata.get("source", "Unknown"), doc.metadata.get("page", "?"))
                if key not in seen:
                    seen.add(key)
                    sources.append({
                        "source": os.path.basename(doc.metadata.get("source", "Unknown")),
                        "page": doc.metadata.get("page", "?"),
                        "snippet": doc.page_content[:200] + "..."
                    })

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
        st.rerun()
