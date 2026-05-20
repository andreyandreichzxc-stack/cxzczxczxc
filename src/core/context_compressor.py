"""Lightweight context compression for the Maestro path only."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CompressionResult:
    compressed_context: str
    dropped_sections: list[str] = field(default_factory=list)
    source_counts: dict[str, int] = field(default_factory=dict)


def _compact_lines(text: str, *, max_lines: int, max_chars: int) -> tuple[str, int]:
    if not text:
        return "", 0
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    original = len(lines)
    if len(lines) > max_lines:
        head = lines[: max_lines // 2]
        tail = lines[-(max_lines - len(head)) :]
        lines = head + ["[...]"] + tail
    compact = "\n".join(lines)
    if len(compact) > max_chars:
        compact = compact[: max_chars - 1] + "…"
    return compact, max(0, original - len(lines))


def compress_maestro_context(
    *,
    history_block: str | None = None,
    memory_context: str | None = None,
    deep_memory: str | None = None,
    agent_outputs: list[str] | None = None,
    budget_chars: int = 9000,
) -> CompressionResult:
    """Compress noisy context while preserving safety/contact facts upstream."""
    dropped: list[str] = []
    counts: dict[str, int] = {}
    parts: list[str] = []

    memory, dropped_memory = _compact_lines(memory_context or "", max_lines=24, max_chars=3000)
    if memory:
        parts.append(memory)
    if dropped_memory:
        dropped.append("memory_context")
        counts["memory_lines_dropped"] = dropped_memory

    deep, dropped_deep = _compact_lines(deep_memory or "", max_lines=18, max_chars=2200)
    if deep:
        parts.append(deep)
    if dropped_deep:
        dropped.append("deep_memory")
        counts["deep_lines_dropped"] = dropped_deep

    history, dropped_history = _compact_lines(history_block or "", max_lines=18, max_chars=2200)
    if history:
        parts.append(history)
    if dropped_history:
        dropped.append("history_block")
        counts["history_lines_dropped"] = dropped_history

    outputs = agent_outputs or []
    if outputs:
        joined, dropped_outputs = _compact_lines(
            "\n\n".join(outputs), max_lines=24, max_chars=3000
        )
        if joined:
            parts.append(joined)
        if dropped_outputs:
            dropped.append("agent_outputs")
            counts["agent_output_lines_dropped"] = dropped_outputs

    combined = "\n\n".join(parts)
    if len(combined) > budget_chars:
        combined = combined[: budget_chars - 1] + "…"
        dropped.append("budget_tail")
    return CompressionResult(combined, dropped, counts)

