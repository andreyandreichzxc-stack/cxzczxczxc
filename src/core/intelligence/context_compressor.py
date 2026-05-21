"""Adaptive 3-stage context compression using token-aware offload.

Replaces the old lightweight compressor with the adaptive offload pipeline
from ``src.core.context.context_offload``.
"""

from __future__ import annotations

import logging
from typing import Any

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
            f"[{m.get('role', '?')}]: {m.get('content', '')[:200]}"
            for m in history[-20:]  # safety: at most 20 messages in fallback
        )
        return raw, None
