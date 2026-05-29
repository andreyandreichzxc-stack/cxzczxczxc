"""RAG pipeline — индексация документов и статей в Qdrant."""

from src.core.rag.chunker import chunk_text, ChunkConfig
from src.core.rag.document_store import (
    DocumentStore,
    DocumentRecord,
    get_document_store,
)
from src.core.rag.ingest import (
    ingest_file,
    ingest_directory,
    get_ingested_documents,
    delete_document,
    rebuild_index,
)

__all__ = [
    "chunk_text",
    "ChunkConfig",
    "DocumentStore",
    "DocumentRecord",
    "get_document_store",
    "ingest_file",
    "ingest_directory",
    "get_ingested_documents",
    "delete_document",
    "rebuild_index",
]
