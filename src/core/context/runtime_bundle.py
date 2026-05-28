from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.core.context.spec import ContextChunk


@dataclass
class RuntimeContextBundle:
    """Unified runtime context handed to prompt assembly."""

    memory_context: str = ""
    self_profile: str = ""
    contact_context: str = ""
    chunks: list[ContextChunk] = field(default_factory=list)
    source_trace: list[str] = field(default_factory=list)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def build_runtime_context(
    *,
    memory_context: str = "",
    self_profile: str = "",
    contact_context: str = "",
    chunks: list[ContextChunk] | None = None,
    extra_sources: list[str] | None = None,
) -> RuntimeContextBundle:
    """Build a deduplicated context bundle from already-loaded sources."""

    source_trace: list[str] = list(extra_sources or [])
    deduped_chunks: list[ContextChunk] = []
    seen: set[tuple[str, str]] = set()
    lines: list[str] = []

    if memory_context.strip():
        source_trace.append("memory_context")
        lines.append(memory_context.strip())
    if self_profile.strip():
        source_trace.append("self_profile")
    if contact_context.strip():
        source_trace.append("contact_context")

    for chunk in chunks or []:
        normalized = _normalize_text(chunk.text)
        key = (chunk.source, normalized)
        if not normalized or key in seen:
            continue
        seen.add(key)
        deduped_chunks.append(chunk)
        reason = f":{chunk.reason}" if chunk.reason else ""
        lines.append(f"[{chunk.source}{reason}] {chunk.text}")
        source_trace.append(chunk.source)

    return RuntimeContextBundle(
        memory_context="\n\n".join(lines),
        self_profile=self_profile,
        contact_context=contact_context,
        chunks=deduped_chunks,
        source_trace=sorted(set(source_trace)),
    )
