import streamlit as st
import tempfile
from process import load_and_embed_pdf
from qa_chain import create_qa_chain
from graph_runner import build_langgraph_flow

st.set_page_config("📄 RAG PDF QA Agent", layout="centered")
st.title("📄 RAG-based PDF QA Agent with Chat Memory")

# Initialize session state
if "graph" not in st.session_state:
    st.session_state.graph = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_uploaded_file_name" not in st.session_state:
    st.session_state.last_uploaded_file_name = None

# Sidebar uploader
uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    if uploaded_file.name != st.session_state.last_uploaded_file_name:
        st.session_state.last_uploaded_file_name = uploaded_file.name
        st.session_state.chat_history = []

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        db = load_and_embed_pdf(tmp_path)
        qa_chain = create_qa_chain(db)
        graph = build_langgraph_flow(qa_chain)

        st.session_state.graph = graph
        st.success("✅ PDF processed and vector store created successfully!")

if st.sidebar.button("Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

if st.session_state.graph:
    for pair in st.session_state.chat_history:
        st.chat_message("user").markdown(pair["question"])
        st.chat_message("assistant").markdown(pair["answer"])
        if "sources" in pair and pair["sources"]:
            with st.expander("📄 Sources", expanded=False):
                for i, doc in enumerate(pair["sources"]):
                    st.markdown(f"**Source {i+1}:** {doc.page_content[:500]}...")

    user_input = st.chat_input("Ask a question about the PDF")
    if user_input:
        state = {"question": user_input}
        result = st.session_state.graph.invoke(state)
        answer = result["answer"]
        sources = result.get("source_documents", [])

        st.session_state.chat_history.append({
            "question": user_input,
            "answer": answer,
            "sources": sources
        })

        st.chat_message("user").markdown(user_input)
        st.chat_message("assistant").markdown(answer)
        if sources:
            with st.expander("📄 Sources", expanded=False):
                for i, doc in enumerate(sources):
                    st.markdown(f"**Source {i+1}:** {doc.page_content[:500]}...")
else:
    st.info("📂 Please upload a PDF file to get started.")