from pathlib import Path
import traceback

from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.vectorstore.store import VectorStoreManager


def main():
    print("Starting corpus ingestion...")

    corpus_path = Path("data/corpus")
    if not corpus_path.exists():
        print("data/corpus folder not found")
        return

    files = list(corpus_path.glob("*.md")) + list(corpus_path.glob("*.pdf"))
    if not files:
        print("No .md or .pdf files found in data/corpus")
        return

    chunker = DocumentChunker()
    manager = VectorStoreManager()

    total_ingested = 0
    total_skipped = 0
    total_errored = 0

    for file_path in files:
        print(f"Ingesting {file_path.name} ...")
        try:
            chunks = chunker.chunk_file(file_path)
            result = manager.ingest(chunks)

            total_ingested += result.ingested
            total_skipped += result.skipped
            total_errored += len(result.errors)

            print(
                f"Done: ingested={result.ingested}, "
                f"skipped={result.skipped}, "
                f"errors={len(result.errors)}, "
                f"details={result.errors}"
            )
        except Exception as exc:
            total_errored += 1
            print(f"Failed on {file_path.name}: {repr(exc)}")
            traceback.print_exc()

    print(
        f"Finished. Total ingested={total_ingested}, "
        f"skipped={total_skipped}, errored={total_errored}"
    )


if __name__ == "__main__":
    main()