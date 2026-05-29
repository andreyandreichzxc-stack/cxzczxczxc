"""Recursive character text chunker with overlap and sentence boundary awareness.

Splits long documents into overlapping chunks for embedding and retrieval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ChunkConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ")


# Matches sentence endings: .!? followed by space or end-of-string
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation boundaries.

    The regex splits on whitespace after .!? — the punctuation stays
    with the preceding fragment, the whitespace is the delimiter.
    Result: already-correct sentences like ["Hello.", "How are you?"]
    """
    parts = _SENTENCE_RE.split(text)
    return [p.strip() for p in parts if p.strip()] or [text]


def chunk_text(
    text: str,
    config: ChunkConfig | None = None,
) -> list[str]:
    """Split text into overlapping chunks using recursive character splitting.

    Strategy:
    1. Try splitting by the most semantic boundary (paragraphs → sentences → words).
    2. Merge chunks to reach target size, respecting sentence boundaries.
    3. Add overlap between adjacent chunks for context continuity.

    Args:
        text: The document text to chunk.
        config: Optional chunking configuration. Uses defaults if None.

    Returns:
        List of text chunks ready for embedding.
    """
    if config is None:
        config = ChunkConfig()
    if not text or not text.strip():
        return []

    # Step 1: split into paragraphs first
    paragraphs = text.split(config.separators[0])
    # Flatten empty paragraphs
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # Step 2: split long paragraphs into sentences
    raw_chunks: list[str] = []
    for para in paragraphs:
        if len(para) <= config.chunk_size:
            raw_chunks.append(para)
        else:
            raw_chunks.extend(_split_sentences(para))

    # Step 3: merge small chunks and split oversized ones
    chunks: list[str] = []
    current = ""

    for raw in raw_chunks:
        candidate = raw if not current else current + "\n" + raw
        if len(candidate) <= config.chunk_size:
            current = candidate
        else:
            # Current chunk is ready, save it
            if current:
                chunks.append(current.strip())
            # Handle the new raw chunk — it might still be too big
            if len(raw) > config.chunk_size:
                chunks.extend(_force_split(raw, config.chunk_size))
                current = ""
            else:
                current = raw

    if current:
        chunks.append(current.strip())

    # Step 4: add overlap between adjacent chunks
    if config.chunk_overlap > 0 and len(chunks) > 1:
        overlapped: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            curr = chunks[i]
            # Take suffix of previous chunk as prefix for current
            overlap_tokens = prev.split()
            if len(overlap_tokens) > config.chunk_overlap // 7:
                suffix = " ".join(overlap_tokens[-(config.chunk_overlap // 7) :])
                overlapped.append(suffix + " " + curr)
            else:
                overlapped.append(curr)
        return overlapped

    return chunks


def _force_split(text: str, max_len: int) -> list[str]:
    """Force-split text into chunks of max_len, trying word boundaries."""
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        wlen = len(word)
        if current_len + wlen + (1 if current else 0) > max_len:
            if current:
                chunks.append(" ".join(current))
            current = [word]
            current_len = wlen
        else:
            current.append(word)
            current_len += wlen + (1 if current_len > 0 else 0)

    if current:
        chunks.append(" ".join(current))

    return chunks if chunks else [text]
