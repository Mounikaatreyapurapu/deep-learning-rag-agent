"""
chunker.py
==========
Corpus document parsing and chunk creation utilities.
"""

from __future__ import annotations

from pathlib import Path

from rag_agent.agent.state import ChunkMetadata, DocumentChunk
from rag_agent.vectorstore.store import VectorStoreManager


class DocumentChunker:
    """
    Converts .md and .pdf source files into DocumentChunk objects.
    """

    def chunk_file(self, file_path: str | Path) -> list[DocumentChunk]:
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()

        if suffix == ".md":
            return self._chunk_markdown(file_path)
        elif suffix == ".pdf":
            return self._chunk_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def _chunk_markdown(self, file_path: Path) -> list[DocumentChunk]:
        text = file_path.read_text(encoding="utf-8").strip()
        if not text:
            return []

        sections = self._split_markdown_sections(text)

        topic = self._infer_topic(file_path.name)
        difficulty = self._infer_difficulty(file_path.name)

        chunks: list[DocumentChunk] = []

        for idx, section in enumerate(sections):
            chunk_text = section.strip()
            if not chunk_text:
                continue

            metadata = ChunkMetadata(
                topic=topic,
                difficulty=difficulty,
                type="concept_explanation",
                source=file_path.name,
                related_topics=[],
                is_bonus=False,
            )

            chunk_id = VectorStoreManager.generate_chunk_id(
                source=file_path.name,
                chunk_text=chunk_text,
            )

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    chunk_text=chunk_text,
                    metadata=metadata,
                )
            )

        return chunks

    def _chunk_pdf(self, file_path: Path) -> list[DocumentChunk]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError(
                "pypdf is required to ingest PDF files."
            ) from exc

        reader = PdfReader(str(file_path))
        pages_text: list[str] = []

        for page in reader.pages:
            extracted = page.extract_text() or ""
            extracted = extracted.strip()
            if extracted:
                pages_text.append(extracted)

        full_text = "\n\n".join(pages_text).strip()
        if not full_text:
            return []

        text_chunks = self._split_text_into_chunks(
            full_text,
            min_words=100,
            max_words=300,
        )

        topic = self._infer_topic(file_path.name)
        difficulty = self._infer_difficulty(file_path.name)

        chunks: list[DocumentChunk] = []

        for idx, chunk_text in enumerate(text_chunks):
            metadata = ChunkMetadata(
                topic=topic,
                difficulty=difficulty,
                type="concept_explanation",
                source=file_path.name,
                related_topics=[],
                is_bonus=False,
            )

            chunk_id = VectorStoreManager.generate_chunk_id(
                source=file_path.name,
                chunk_text=chunk_text,
            )

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    chunk_text=chunk_text,
                    metadata=metadata,
                )
            )

        return chunks

    def _split_markdown_sections(self, text: str) -> list[str]:
        lines = text.splitlines()
        sections: list[str] = []
        current: list[str] = []

        for line in lines:
            if line.startswith("## ") and current:
                sections.append("\n".join(current).strip())
                current = [line]
            else:
                current.append(line)

        if current:
            sections.append("\n".join(current).strip())

        cleaned = [
            section for section in sections
            if len(section.split()) >= 30
        ]

        if cleaned:
            return cleaned

        return self._split_text_into_chunks(text, min_words=100, max_words=300)

    def _split_text_into_chunks(
        self,
        text: str,
        min_words: int = 100,
        max_words: int = 300,
    ) -> list[str]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: list[str] = []
        current_words: list[str] = []

        for para in paragraphs:
            para_words = para.split()

            if len(current_words) + len(para_words) <= max_words:
                current_words.extend(para_words)
            else:
                if current_words:
                    chunks.append(" ".join(current_words))
                current_words = para_words[:]

        if current_words:
            chunks.append(" ".join(current_words))

        merged: list[str] = []
        buffer = ""

        for chunk in chunks:
            if len(chunk.split()) < min_words and merged:
                merged[-1] = merged[-1] + "\n\n" + chunk
            else:
                merged.append(chunk)

        return merged

    def _infer_topic(self, filename: str) -> str:
        lower = filename.lower()
        if "ann" in lower:
            return "ANN"
        if "cnn" in lower:
            return "CNN"
        if "rnn" in lower:
            return "RNN"
        return "Unknown"

    def _infer_difficulty(self, filename: str) -> str:
        lower = filename.lower()
        if "beginner" in lower:
            return "beginner"
        if "advanced" in lower:
            return "advanced"
        return "intermediate"