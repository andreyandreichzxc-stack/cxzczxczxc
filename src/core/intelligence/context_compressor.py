"""Adaptive 3-stage context compression using token-aware offload.

Replaces the old lightweight compressor with the adaptive offload pipeline
from ``src.core.context.context_offload``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


async def compress_maestro_context(
    history: list[dict],
    system_prompt: str = "",
    user_prompt: str = "",
    *,
    owner_id: int | None = None,
) -> tuple[str, str | None]:
    """Adaptive 3-stage context compression using token-aware offload.

    Returns (compressed_context_text, mermaid_graph | None).
    The mermaid graph can be injected as a lightweight alternative to raw history.
    """
    from src.core.context.context_offload import compress_context, CompressedContext

    if not history:
        return "", None

    try:
        compressed: CompressedContext = await compress_context(
            messages=history,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        # Save checkpoint for reuse
        if owner_id is not None:
            from src.core.context.offload_checkpoint import save_offload_state

            await save_offload_state(
                user_id=owner_id,
                messages=compressed.messages,
                mermaid_graph=compressed.mermaid_graph,
                drilldown_refs=compressed.drilldown_refs,
                tokens_saved=compressed.tokens_before - compressed.tokens_after,
            )

        # Build compressed context string
        parts: list[str] = []

        if compressed.mermaid_graph:
            parts.append("```mermaid")
            parts.append(compressed.mermaid_graph)
            parts.append("```")
            parts.append("")  # blank line separator

        # Remaining messages (critical facts + summary)
        for msg in compressed.messages:
            role = msg.get("role", "system")
            content = msg.get("content", "")
            if role == "system":
                parts.append(content)
            else:
                parts.append(f"[{role}]: {content}")

        context_text = "\n".join(parts)
        return context_text, compressed.mermaid_graph

    except Exception:
        logger.debug("Context offload failed, using raw history", exc_info=True)
        # Fallback: return raw history as text
        raw = "\n".join(
            f"[{m.get('role', '?')}]: {(m.get('content') or '')[:200]}"
            for m in history[-20:]  # safety: at most 20 messages in fallback
        )
        return raw, None


async def compress_context(history: list[dict], max_tokens: int = 4000) -> str:
    """Compress conversation turns into a single summary string.

    A lightweight compressor for conversation_context turns (list of
    ``{"role": …, "content": …}`` dicts).  Truncates to *max_tokens*
    approximate tokens (4 chars ≈ 1 token).
    """
    if not history:
        return ""
    # Simple approach: concatenate with labels, truncate to max_tokens
    lines: list[str] = []
    for turn in history[-20:]:  # last 20 turns
        role = turn.get("role", "?")
        content = turn.get("content", "")
        if content:
            lines.append(f"[{role}]: {content[:200]}")
    summary = "\n".join(lines)
    # Truncate to approximate token count (4 chars ≈ 1 token)
    max_chars = max_tokens * 4
    if len(summary) > max_chars:
        summary = summary[:max_chars] + "\n...[сжато]"
    return summary
