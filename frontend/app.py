import os
import requests
import streamlit as st

# ============================= PAGE CONFIG =============================
st.set_page_config(
    page_title="Chat with Energy Documents",
    page_icon="🧠",
    layout="wide",
)

# ============================= SIDEBAR SETTINGS =============================
with st.sidebar:
    st.header("⚙️ Settings")
    API_BASE = st.text_input("Backend URL", value="http://localhost:8000", help="Your FastAPI backend URL")

    theme = st.radio("🌓 Theme", ["🌙 Dark", "☀️ Light"], horizontal=True)

    show_sources = st.checkbox("Show Sources", value=False, help="Toggle whether to show source snippets below answers.")

    st.markdown("---")
    st.markdown("**Endpoints:** `/upload`, `/query`, `/docs-list`")

    if st.button("🔄 Refresh Document List"):
        try:
            r = requests.get(f"{API_BASE}/docs-list", timeout=30)
            st.session_state["docs"] = r.json().get("documents", [])
            st.success("Document list updated!")
        except Exception as e:
            st.error(f"Failed to refresh docs: {e}")

# Apply theme styling
if theme == "🌙 Dark":
    st.markdown("""
        <style>
        body, .stApp { background-color: #0E1117; color: #FAFAFA; }
        div[data-testid="stSidebar"] { background-color: #1E1E26; color: #FAFAFA; }
        .stButton>button { background-color: #1F77B4; color: white; border-radius: 10px; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .stButton>button { background-color: #4F46E5; color: white; border-radius: 10px; }
        </style>
    """, unsafe_allow_html=True)

# ============================= APP TITLE =============================
st.markdown(
    "<h1 style='text-align:center;'>📚Chat with Energy Documents</h1>",
    unsafe_allow_html=True,
)
st.caption("Ask questions directly from your uploaded research papers. Built with 💡 FastAPI + Streamlit + FAISS RAG")

# ============================= TABS =============================
tabs = st.tabs(["📤 Upload & Index", "❓ Ask a Question", "📄 Indexed Documents"])

# -------------------- TAB 1: Upload PDFs --------------------
with tabs[0]:
    st.subheader("📥 Upload Your Paper")
    files = st.file_uploader("Select PDF(s) to upload", type=["pdf"], accept_multiple_files=True)

    if st.button("🚀 Index Files") and files:
        with st.spinner("Processing & indexing your documents..."):
            try:
                files_payload = [("files", (f.name, f.getvalue(), "application/pdf")) for f in files]
                r = requests.post(f"{API_BASE}/upload", files=files_payload, timeout=300)
                if r.status_code == 200:
                    data = r.json()
                    st.success(f"✅ Indexed {data.get('indexed', 0)} text chunks successfully!")
                else:
                    st.error(f"Error: {r.text}")
            except Exception as e:
                st.error(f"Upload failed: {e}")

# -------------------- TAB 2: Ask Questions --------------------
with tabs[1]:
    st.subheader("💬 Ask a Question About Your Uploaded Papers")

    question = st.text_input("Type your question below 👇", placeholder="e.g., What is the main contribution of this paper?")
    top_k = st.slider("🔍 Top-K Chunks to Retrieve", 1, 10, 5)

    if st.button("💡 Get Answer") and question.strip():
        with st.spinner("Analyzing documents and generating answer..."):
            try:
                response = requests.post(f"{API_BASE}/query", data={"q": question, "top_k": top_k}, timeout=120)

                if response.status_code == 200:
                    data = response.json()
                    st.markdown("### 🧠 **Answer**")
                    st.write(data.get("answer", "No answer generated."))

                    if show_sources and "sources" in data:
                        st.markdown("---")
                        st.markdown("### 📎 **Sources (optional)**")
                        for s in data["sources"]:
                            st.markdown(f"- **{s['doc_title']}**, p.{s['page']} — score: {s['score']:.3f}")
                            st.caption(s.get("text", "")[:300] + "...")

                else:
                    st.error(f"Server Error: {response.text}")

            except Exception as e:
                st.error(f"Failed to query backend: {e}")

# -------------------- TAB 3: List Documents --------------------
with tabs[2]:
    st.subheader("📑 Indexed Documents")

    docs = st.session_state.get("docs", [])
    if not docs:
        try:
            r = requests.get(f"{API_BASE}/docs-list", timeout=30)
            docs = r.json().get("documents", [])
        except Exception as e:
            st.error(f"Error loading docs: {e}")

    if docs:
        for d in docs:
            st.markdown(
                f"**📘 {d['title']}** — {d['pages']} pages — added *{d['created_at']}*"
            )
    else:
        st.info("No documents found. Try uploading some PDFs first.")
