"""
nodes.py
========
LangGraph node functions for the RAG interview preparation agent.

Each function in this module is a node in the agent state graph.
Nodes receive the current AgentState, perform their operation,
and return a dict of state fields to update.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages

from rag_agent.agent.prompts import (
    QUESTION_GENERATION_PROMPT,
    SYSTEM_PROMPT,
)
from rag_agent.agent.state import AgentResponse, AgentState, RetrievedChunk
from rag_agent.config import LLMFactory, get_settings
from rag_agent.vectorstore.store import VectorStoreManager


# ---------------------------------------------------------------------------
# Node: Query Rewriter
# ---------------------------------------------------------------------------


def query_rewrite_node(state: AgentState) -> dict:
    """
    Rewrite the user's query to maximise retrieval effectiveness.

    Natural language questions are often poorly suited for vector
    similarity search. This node rephrases the query into a form
    that produces better embedding matches against the corpus.

    Example
    -------
    Input:  "I'm confused about how LSTMs remember things long-term"
    Output: "LSTM long-term memory cell state forget gate mechanism"

    Interview talking point: query rewriting is a production RAG pattern
    that significantly improves retrieval recall. It acknowledges that
    users do not phrase queries the way documents are written.

    Parameters
    ----------
    state : AgentState
        Current graph state. Reads: messages (for context).

    Returns
    -------
    dict
        Updates: original_query, rewritten_query.
    """
    # TODO: implement
    # 1. Extract the latest HumanMessage from state.messages as original_query
    # 2. Build a short prompt instructing the LLM to rewrite for vector search
    #    Keep the rewriting prompt lightweight — this adds latency
    # 3. Call llm.invoke() with the rewrite prompt
    # 4. Return {"original_query": original_query, "rewritten_query": rewritten}
    #
    # Fallback: if rewriting fails (API error, timeout), return the original
    # query unchanged so the graph continues gracefully
    original_query = ""

    for message in reversed(state.messages):
        if isinstance(message, HumanMessage):
            original_query = str(message.content).strip()
            break

    if not original_query:
        return {"original_query": "", "rewritten_query": ""}

    llm = LLMFactory(get_settings()).create()

    rewrite_prompt = (
        "Rewrite the following user query for semantic vector search over a deep "
        "learning study corpus. Preserve the technical meaning, remove filler "
        "language, and output only the rewritten query.\n\n"
        f"User query: {original_query}"
    )

    try:
        rewritten = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You rewrite user questions into concise retrieval queries "
                        "for a deep learning RAG system."
                    )
                ),
                HumanMessage(content=rewrite_prompt),
            ]
        )

        rewritten_query = str(rewritten.content).strip()
        if not rewritten_query:
            rewritten_query = original_query

        return {
            "original_query": original_query,
            "rewritten_query": rewritten_query,
        }
    except Exception:
        return {
            "original_query": original_query,
            "rewritten_query": original_query,
        }


# ---------------------------------------------------------------------------
# Node: Retriever
# ---------------------------------------------------------------------------


def retrieval_node(state: AgentState) -> dict:
    """
    Retrieve relevant chunks from ChromaDB based on the rewritten query.

    Sets the no_context_found flag if no chunks meet the similarity
    threshold. This flag is checked by generation_node to trigger
    the hallucination guard.

    Interview talking point: separating retrieval into its own node
    makes it independently testable and replaceable — you could swap
    ChromaDB for Pinecone or Weaviate by changing only this node.

    Parameters
    ----------
    state : AgentState
        Current graph state.
        Reads: rewritten_query, topic_filter, difficulty_filter.

    Returns
    -------
    dict
        Updates: retrieved_chunks, no_context_found.
    """
    # TODO: implement
    # 1. Instantiate VectorStoreManager (consider caching this)
    # 2. manager.query(
    #        query_text=state.rewritten_query,
    #        topic_filter=state.topic_filter,
    #        difficulty_filter=state.difficulty_filter
    #    )
    # 3. If result is empty: return {"retrieved_chunks": [], "no_context_found": True}
    # 4. Otherwise: return {"retrieved_chunks": chunks, "no_context_found": False}
    manager = VectorStoreManager()

    query_text = state.rewritten_query or state.original_query or ""

    chunks = manager.query(
        query_text=query_text,
        topic_filter=getattr(state, "topic_filter", None),
        difficulty_filter=getattr(state, "difficulty_filter", None),
    )

    if not chunks:
        return {"retrieved_chunks": [], "no_context_found": True}

    return {"retrieved_chunks": chunks, "no_context_found": False}


# ---------------------------------------------------------------------------
# Node: Generator
# ---------------------------------------------------------------------------


def generation_node(state: AgentState) -> dict:
    """
    Generate the final response using retrieved chunks as context.

    Implements the hallucination guard: if no_context_found is True,
    returns a clear "no relevant context" message rather than allowing
    the LLM to answer from parametric memory.

    Implements token-aware conversation memory trimming: when the
    message history approaches max_context_tokens, the oldest
    non-system messages are removed.

    Interview talking point: the hallucination guard is the most
    commonly asked about production RAG pattern. Interviewers want
    to know how you prevent the model from confidently making up
    information when the retrieval step finds nothing relevant.

    Parameters
    ----------
    state : AgentState
        Current graph state.
        Reads: retrieved_chunks, no_context_found, messages,
               original_query, topic_filter.

    Returns
    -------
    dict
        Updates: final_response, messages (with new AIMessage appended).
    """
    settings = get_settings()
    llm = LLMFactory(settings).create()

    # ---- Hallucination Guard -----------------------------------------------
    if state.no_context_found:
        no_context_message = (
            "I was unable to find relevant information in the corpus for your query. "
            "This may mean the topic is not yet covered in the study material, or "
            "your query may need to be rephrased. Please try a more specific "
            "deep learning topic such as 'LSTM forget gate' or 'CNN pooling layers'."
        )
        response = AgentResponse(
            answer=no_context_message,
            sources=[],
            confidence=0.0,
            no_context_found=True,
            rewritten_query=state.rewritten_query,
        )
        return {
            "final_response": response,
            "messages": [AIMessage(content=no_context_message)],
        }

    # ---- Build Context from Retrieved Chunks --------------------------------
    # TODO: implement
    # 1. Format retrieved chunks into a context string with citations
    #    Each chunk should appear as: "[SOURCE: topic | file]\n{chunk_text}\n"
    # 2. Calculate average confidence score from chunk scores
    # 3. Build the full prompt:
    #    - SystemMessage with SYSTEM_PROMPT
    #    - Context message with formatted chunks
    #    - Trimmed conversation history (trim to max_context_tokens)
    #    - HumanMessage with original_query
    # 4. llm.invoke(messages)
    # 5. Construct AgentResponse with answer, sources (list of citations), confidence
    # 6. Append AIMessage to messages
    # 7. Return {"final_response": response, "messages": [new_ai_message]}
    formatted_chunks: list[str] = []
    sources: list[str] = []
    scores: list[float] = []

    for chunk in state.retrieved_chunks:
        citation = f"[SOURCE: {chunk.metadata.topic} | {chunk.metadata.source}]"
        formatted_chunks.append(f"{citation}\n{chunk.chunk_text}\n")
        sources.append(citation)
        scores.append(float(chunk.score))

    context_block = "\n".join(formatted_chunks)
    confidence = sum(scores) / len(scores) if scores else 0.0

    trimmed_history = trim_messages(
        state.messages,
        max_tokens=settings.max_context_tokens,
        token_counter=len,
        strategy="last",
        include_system=False,
        start_on="human",
        allow_partial=False,
    )

    prompt_messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                "Use only the retrieved context below to answer the user's question. "
                "If the answer is not supported by the context, say so.\n\n"
                f"Retrieved context:\n{context_block}"
            )
        ),
        *trimmed_history,
        HumanMessage(content=state.original_query),
    ]

    llm_response = llm.invoke(prompt_messages)
    answer_text = str(llm_response.content).strip()

    response = AgentResponse(
        answer=answer_text,
        sources=sources,
        confidence=confidence,
        no_context_found=False,
        rewritten_query=state.rewritten_query,
    )

    new_ai_message = AIMessage(content=answer_text)

    return {
        "final_response": response,
        "messages": [new_ai_message],
    }


# ---------------------------------------------------------------------------
# Routing Function
# ---------------------------------------------------------------------------


def should_retry_retrieval(state: AgentState) -> str:
    """
    Conditional edge function: decide whether to retry retrieval or generate.

    Called by the graph after retrieval_node. If no context was found,
    the graph routes back to query_rewrite_node for one retry with a
    broader query before triggering the hallucination guard.

    Interview talking point: conditional edges in LangGraph enable
    agentic behaviour — the graph makes decisions about its own
    execution path rather than following a fixed sequence.

    Parameters
    ----------
    state : AgentState
        Current graph state. Reads: no_context_found, retrieved_chunks.

    Returns
    -------
    str
        "generate" — proceed to generation_node.
        "end"      — skip generation, return no_context response directly.

    Notes
    -----
    Retry logic should be limited to one attempt to prevent infinite loops.
    Track retry count in AgentState if implementing retry behaviour.
    """
    # TODO: implement
    # Simple version: if no_context_found → "end", else → "generate"
    # Advanced version: track retry count, allow one retry with broader query
    if state.no_context_found:
        return "end"
    return "generate"
