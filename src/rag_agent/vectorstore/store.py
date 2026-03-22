"""
store.py
========
ChromaDB vector store management.

Handles all interactions with the persistent ChromaDB collection:
initialisation, ingestion, duplicate detection, and retrieval.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import chromadb
from loguru import logger

from rag_agent.agent.state import (
    ChunkMetadata,
    DocumentChunk,
    IngestionResult,
    RetrievedChunk,
)
from rag_agent.config import EmbeddingFactory, Settings, get_settings


class VectorStoreManager:
    """
    Manages the ChromaDB persistent vector store for the corpus.result.errored

    All corpus ingestion and retrieval operations pass through this class.
    It is the single point of contact between the application and ChromaDB.

    Parameters
    ----------
    settings : Settings, optional
        Application settings. Uses get_settings() singleton if not provided.

    Example
    -------
    >>> manager = VectorStoreManager()
    >>> result = manager.ingest(chunks)
    >>> print(f"Ingested: {result.ingested}, Skipped: {result.skipped}")
    >>>
    >>> chunks = manager.query("explain the vanishing gradient problem", k=4)
    >>> for chunk in chunks:
    ...     print(chunk.to_citation(), chunk.score)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._embeddings = EmbeddingFactory(self._settings).create()
        self._client = None
        self._collection = None
        self._initialise()

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def _initialise(self) -> None:
        """
        Create or connect to the persistent ChromaDB client and collection.

        Creates the chroma_db_path directory if it does not exist.
        Uses PersistentClient so data survives between application restarts.

        Called automatically during __init__. Should not be called directly.

        Raises
        ------
        RuntimeError
            If ChromaDB cannot be initialised at the configured path.
        """
        # TODO: implement
        # 1. Ensure Path(self._settings.chroma_db_path).mkdir(parents=True, exist_ok=True)
        # 2. chromadb.PersistentClient(path=self._settings.chroma_db_path)
        # 3. client.get_or_create_collection(
        #        name=self._settings.chroma_collection_name,
        #        metadata={"hnsw:space": "cosine"}   # cosine similarity
        #    )
        # 4. Log successful initialisation with collection name and item count
        try:
            db_path = Path(self._settings.chroma_db_path)
            db_path.mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(path=str(db_path))
            self._collection = self._client.get_or_create_collection(
                name=self._settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            item_count = self._collection.count()
            logger.info(
                "Initialised ChromaDB collection '{}' at '{}' with {} items.",
                self._settings.chroma_collection_name,
                str(db_path),
                item_count,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialise ChromaDB at "
                f"'{self._settings.chroma_db_path}': {exc}"
            ) from exc

    # -----------------------------------------------------------------------
    # Duplicate Detection
    # -----------------------------------------------------------------------

    @staticmethod
    def generate_chunk_id(source: str, chunk_text: str) -> str:
        """
        Generate a deterministic chunk ID from source filename and content.

        Using a content hash ensures two uploads of the same file produce
        the same IDs, making duplicate detection reliable regardless of
        filename changes.

        Parameters
        ----------
        source : str
            The source filename (e.g. 'lstm.md').
        chunk_text : str
            The full text content of the chunk.

        Returns
        -------
        str
            A 16-character hex string derived from SHA-256 of the inputs.
        """
        content = f"{source}::{chunk_text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def check_duplicate(self, chunk_id: str) -> bool:
        """
        Check whether a chunk with this ID already exists in the collection.

        Parameters
        ----------
        chunk_id : str
            The deterministic chunk ID to check.

        Returns
        -------
        bool
            True if the chunk already exists (duplicate). False otherwise.

        Interview talking point: content-addressed deduplication is more
        robust than filename-based deduplication because it detects identical
        content even when files are renamed or re-uploaded.
        """
        # TODO: implement
        # self._collection.get(ids=[chunk_id])
        # Return True if the result contains the ID, False otherwise
        result = self._collection.get(ids=[chunk_id])
        ids = result.get("ids", []) if result else []
        return chunk_id in ids

    # -----------------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------------

    def ingest(self, chunks: list[DocumentChunk]) -> IngestionResult:
        """
        Embed and store a list of DocumentChunks in ChromaDB.

        Checks each chunk for duplicates before embedding. Skips duplicates
        silently and records the count in the returned IngestionResult.

        Parameters
        ----------
        chunks : list[DocumentChunk]
            Prepared chunks with text and metadata. Use DocumentChunker
            to produce these from raw files.

        Returns
        -------
        IngestionResult
            Summary with counts of ingested, skipped, and errored chunks.

        Notes
        -----
        Embeds in batches of 100 to avoid memory issues with large corpora.
        Uses upsert (not add) so re-ingestion of modified content updates
        existing chunks rather than raising an error.

        Interview talking point: batch processing with a configurable
        batch size is a production pattern that prevents OOM errors when
        ingesting large document sets.
        """
        # TODO: implement
        # result = IngestionResult()
        # For each chunk:
        #   - check_duplicate(chunk.chunk_id) → if True, result.skipped += 1, continue
        #   - embed chunk.chunk_text using self._embeddings.embed_documents([chunk.chunk_text])
        #   - self._collection.upsert(
        #         ids=[chunk.chunk_id],
        #         embeddings=[embedding],
        #         documents=[chunk.chunk_text],
        #         metadatas=[chunk.metadata.to_dict()]
        #     )
        #   - result.ingested += 1
        # Log summary and return result
        result = IngestionResult()

        if not chunks:
            logger.info("No chunks provided for ingestion.")
            return result

        batch_size = 100
        pending_chunks: list[DocumentChunk] = []

        def flush_batch(batch: list[DocumentChunk]) -> None:
            if not batch:
                return

            texts = [chunk.chunk_text for chunk in batch]
            ids = [chunk.chunk_id for chunk in batch]
            metadatas = [chunk.metadata.to_dict() for chunk in batch]

            embeddings = self._embeddings.embed_documents(texts)

            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

        for chunk in chunks:
            try:
                if self.check_duplicate(chunk.chunk_id):
                    result.skipped += 1
                    continue

                pending_chunks.append(chunk)

                if len(pending_chunks) >= batch_size:
                    flush_batch(pending_chunks)
                    result.ingested += len(pending_chunks)
                    pending_chunks = []

            except Exception as exc:
                result.errors += 1
                logger.exception(
                    "Failed preparing chunk '{}' from source '{}': {}",
                    getattr(chunk, "chunk_id", "unknown"),
                    getattr(chunk.metadata, "source", "unknown"),
                    exc,
                )

        try:
            if pending_chunks:
                flush_batch(pending_chunks)
                result.ingested += len(pending_chunks)
        except Exception as exc:
            result.errors += len(pending_chunks)
            logger.exception("Failed ingesting final batch: {}", exc)

        logger.info(
            "Ingestion complete. Ingested: {}, Skipped: {}, Errored: {}",
            result.ingested,
            result.skipped,
            result.errors,
        )
        return result


    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        k: int | None = None,
        topic_filter: str | None = None,
        difficulty_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Applies similarity threshold filtering — chunks below
        settings.similarity_threshold are excluded from results.

        Parameters
        ----------
        query_text : str
            The user query or rewritten query to retrieve against.
        k : int, optional
            Number of chunks to retrieve. Defaults to settings.retrieval_k.
        topic_filter : str, optional
            Restrict retrieval to a specific topic (e.g. 'LSTM').
            Maps to ChromaDB where-filter on metadata.topic.
        difficulty_filter : str, optional
            Restrict retrieval to a difficulty level.
            Maps to ChromaDB where-filter on metadata.difficulty.

        Returns
        -------
        list[RetrievedChunk]
            Chunks sorted by similarity score descending.
            Empty list if no chunks meet the similarity threshold.

        Interview talking point: returning an empty list (not hallucinating)
        when no relevant context exists is the hallucination guard. This is
        a critical production RAG pattern — the system must know what it
        does not know.
        """
        # TODO: implement
        # k = k or self._settings.retrieval_k
        # Build where_filter dict from topic_filter and difficulty_filter if provided
        # Embed query_text using self._embeddings.embed_query(query_text)
        # self._collection.query(
        #     query_embeddings=[query_embedding],
        #     n_results=k,
        #     where=where_filter,      # None if no filters
        #     include=["documents", "metadatas", "distances"]
        # )
        # Convert distances to similarity scores: score = 1 - distance (for cosine)
        # Filter out chunks below self._settings.similarity_threshold
        # Return list of RetrievedChunk objects sorted by score descending
        if not query_text or not query_text.strip():
            return []

        k = k or self._settings.retrieval_k

        where_filter = None
        filters: dict[str, str] = {}

        if topic_filter:
            filters["topic"] = topic_filter
        if difficulty_filter:
            filters["difficulty"] = difficulty_filter

        if filters:
            where_filter = filters

        query_embedding = self._embeddings.embed_query(query_text)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        retrieved_chunks: list[RetrievedChunk] = []

        for chunk_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            score = 1 - float(distance)

            if score < self._settings.similarity_threshold:
                continue

            chunk_metadata = ChunkMetadata(**metadata)
            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    chunk_text=document,
                    metadata=chunk_metadata,
                    score=score,
                )
            )

        retrieved_chunks.sort(key=lambda chunk: chunk.score, reverse=True)
        return retrieved_chunks

    # -----------------------------------------------------------------------
    # Corpus Inspection
    # -----------------------------------------------------------------------

    def list_documents(self) -> list[dict]:
        results = self._collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", []) if results else []

        grouped: dict[str, dict] = {}

        for metadata in metadatas:
            source = metadata.get("source", "unknown")
            topic = metadata.get("topic", "unknown")

            if source not in grouped:
                grouped[source] = {
                    "source": source,
                    "topic": topic,
                    "chunk_count": 0,
                }

            grouped[source]["chunk_count"] += 1

        return list(grouped.values())

    def get_document_chunks(self, source: str) -> list[DocumentChunk]:
        results = self._collection.get(
            where={"source": source},
            include=["documents", "metadatas"],
        )

        documents = results.get("documents", []) if results else []
        metadatas = results.get("metadatas", []) if results else []
        ids = results.get("ids", []) if results else []

        chunks: list[DocumentChunk] = []

        for chunk_id, document, metadata in zip(ids, documents, metadatas):
            chunk_metadata = ChunkMetadata(**metadata)
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    chunk_text=document,
                    metadata=chunk_metadata,
                )
            )

        return chunks

    def delete_document(self, source: str) -> int:
        results = self._collection.get(where={"source": source})
        ids = results.get("ids", []) if results else []

        if ids:
            self._collection.delete(ids=ids)

        return len(ids)

    def get_collection_stats(self) -> dict:
        results = self._collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", []) if results else []

        topics = set()
        bonus = False

        for m in metadatas:
            topic = m.get("topic")
            if topic:
                topics.add(topic)
            if m.get("is_bonus"):
                bonus = True

        return {
            "total_chunks": len(metadatas),
            "topics": list(topics),
            "bonus_topics_present": bonus,
        }