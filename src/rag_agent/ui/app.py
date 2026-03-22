"""
app.py
======
Streamlit user interface for the Deep Learning RAG Interview Prep Agent.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import streamlit as st
from langchain_core.messages import HumanMessage

from rag_agent.agent.graph import get_compiled_graph
from rag_agent.config import get_settings
from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.vectorstore.store import VectorStoreManager


@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    return VectorStoreManager()


@st.cache_resource
def get_chunker() -> DocumentChunker:
    return DocumentChunker()


@st.cache_resource
def get_graph():
    return get_compiled_graph()


def initialise_session_state() -> None:
    defaults = {
        "chat_history": [],
        "ingested_documents": [],
        "selected_document": None,
        "last_ingestion_result": None,
        "thread_id": "default-session",
        "topic_filter": None,
        "difficulty_filter": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def _refresh_documents(store: VectorStoreManager) -> None:
    st.session_state["ingested_documents"] = store.list_documents()


def render_ingestion_panel(
    store: VectorStoreManager,
    chunker: DocumentChunker,
) -> None:
    st.sidebar.header("📂 Corpus Ingestion")

    uploaded_files = st.sidebar.file_uploader(
        "Upload study materials",
        type=["pdf", "md"],
        accept_multiple_files=True,
    )

    ingest_clicked = st.sidebar.button(
        "Ingest Documents",
        disabled=not uploaded_files,
        use_container_width=True,
    )

    if ingest_clicked and uploaded_files:
        total_ingested = 0
        total_skipped = 0
        total_errors = 0
        error_details: list[str] = []

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            for uploaded_file in uploaded_files:
                saved_path = tmp_path / uploaded_file.name
                saved_path.write_bytes(uploaded_file.getbuffer())

                try:
                    chunks = chunker.chunk_file(saved_path)
                    result = store.ingest(chunks)

                    total_ingested += result.ingested
                    total_skipped += result.skipped
                    total_errors += len(result.errors)

                    if result.errors:
                        error_details.extend([str(err) for err in result.errors])

                except Exception as exc:
                    total_errors += 1
                    error_details.append(f"{uploaded_file.name}: {exc}")

        st.session_state["last_ingestion_result"] = {
            "ingested": total_ingested,
            "skipped": total_skipped,
            "errors": total_errors,
            "details": error_details,
        }

        _refresh_documents(store)

        if total_ingested > 0:
            st.sidebar.success(
                f"{total_ingested} chunks added, {total_skipped} duplicates skipped"
            )
        elif total_skipped > 0 and total_errors == 0:
            st.sidebar.warning(
                f"No new chunks added. {total_skipped} duplicates were skipped."
            )

        if total_errors > 0:
            st.sidebar.error(f"{total_errors} ingestion errors occurred.")
            with st.sidebar.expander("Error details"):
                for detail in error_details:
                    st.write(detail)

    if not st.session_state["ingested_documents"]:
        _refresh_documents(store)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Ingested Documents")

    docs = st.session_state["ingested_documents"]

    if not docs:
        st.sidebar.info("No documents ingested yet.")
        return

    for idx, doc in enumerate(docs):
        col1, col2 = st.sidebar.columns([4, 1])
        with col1:
            st.caption(
                f"**{doc['source']}**  \nTopic: {doc['topic']}  \nChunks: {doc['chunk_count']}"
            )
        with col2:
            if st.button("🗑", key=f"delete_doc_{idx}", help="Remove document"):
                deleted = store.delete_document(doc["source"])
                if deleted:
                    st.sidebar.success(f"Removed {doc['source']}")
                _refresh_documents(store)
                if st.session_state["selected_document"] == doc["source"]:
                    st.session_state["selected_document"] = None
                st.rerun()


def render_corpus_stats(store: VectorStoreManager) -> None:
    stats = store.get_collection_stats()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Corpus Stats")
    st.sidebar.metric("Total Chunks", stats["total_chunks"])

    topics = stats.get("topics", [])
    if topics:
        st.sidebar.write("Topics:", ", ".join(topics))
    else:
        st.sidebar.write("Topics: None yet")

    if stats.get("bonus_topics_present"):
        st.sidebar.success("✅ Bonus topics present")
    else:
        st.sidebar.warning("⚠️ No bonus topics yet")


def render_document_viewer(store: VectorStoreManager) -> None:
    st.subheader("📄 Document Viewer")

    docs = st.session_state["ingested_documents"]
    if not docs:
        st.info("Ingest documents using the sidebar to view content here.")
        return

    options = [doc["source"] for doc in docs]

    if (
        st.session_state["selected_document"] is None
        or st.session_state["selected_document"] not in options
    ):
        st.session_state["selected_document"] = options[0]

    selected_source = st.selectbox(
        "Select document",
        options=options,
        index=options.index(st.session_state["selected_document"]),
    )
    st.session_state["selected_document"] = selected_source

    chunks = store.get_document_chunks(selected_source)

    st.caption(f"Chunks: {len(chunks)}")

    viewer_container = st.container(height=420)
    with viewer_container:
        for idx, chunk in enumerate(chunks, start=1):
            meta = chunk.metadata
            st.markdown(
                f"**Chunk {idx}**  \n"
                f"`{meta.topic}` | `{meta.difficulty}` | `{meta.type}`"
            )
            st.write(chunk.chunk_text)
            st.markdown("---")


def render_chat_interface(graph) -> None:
    st.subheader("💬 Interview Prep Chat")

    docs = st.session_state["ingested_documents"]
    topics = sorted({doc["topic"] for doc in docs if doc.get("topic")})

    col_topic, col_diff = st.columns(2)
    with col_topic:
        topic_value = st.selectbox(
            "Topic",
            options=["All"] + topics,
            index=0,
        )
        st.session_state["topic_filter"] = None if topic_value == "All" else topic_value

    with col_diff:
        diff_value = st.selectbox(
            "Difficulty",
            options=["All", "beginner", "intermediate", "advanced"],
            index=0,
        )
        st.session_state["difficulty_filter"] = (
            None if diff_value == "All" else diff_value
        )

    chat_container = st.container(height=400)
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    with st.expander("📎 Sources"):
                        for source in message["sources"]:
                            st.caption(source)
                if message.get("no_context_found"):
                    st.warning("⚠️ No relevant content found in corpus.")

    query = st.chat_input("Ask about a deep learning topic...")

    if query:
        st.session_state.chat_history.append(
            {"role": "user", "content": query}
        )

        graph_input = {
            "messages": [HumanMessage(content=query)],
            "topic_filter": st.session_state["topic_filter"],
            "difficulty_filter": st.session_state["difficulty_filter"],
        }
        config = {
            "configurable": {
                "thread_id": st.session_state["thread_id"],
            }
        }

        try:
            result = graph.invoke(graph_input, config=config)
            response = result["final_response"]

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": response.answer,
                    "sources": response.sources,
                    "no_context_found": response.no_context_found,
                }
            )
        except Exception as exc:
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": f"An error occurred while generating a response: {exc}",
                    "sources": [],
                    "no_context_found": False,
                }
            )

        st.rerun()


def main() -> None:
    settings = get_settings()

    st.set_page_config(
        page_title=settings.app_title,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(f"🧠 {settings.app_title}")
    st.caption(
        "RAG-powered interview preparation — built with LangChain, LangGraph, and ChromaDB"
    )

    initialise_session_state()

    store = get_vector_store()
    chunker = get_chunker()
    graph = get_graph()

    render_ingestion_panel(store, chunker)
    render_corpus_stats(store)

    viewer_col, chat_col = st.columns([1, 1], gap="large")

    with viewer_col:
        render_document_viewer(store)

    with chat_col:
        render_chat_interface(graph)


if __name__ == "__main__":
    main()