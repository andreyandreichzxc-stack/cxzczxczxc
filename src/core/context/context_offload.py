"""3-stage adaptive context offload: format → LLM summary → Mermaid graph.

Stage 1 (format, 50%+): Compress formatting — strip redundant newlines,
    shorten role labels, remove empty messages.
Stage 2 (summary, 75%+): Use LLM to summarise oldest messages into brief
    narrative, keeping critical facts intact.
Stage 3 (mermaid, 85%+): Convert conversation to Mermaid graph with
    drill-down reference IDs.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from src.core.context.token_tracker import (
    count_prompt_tokens,
    get_budget_stage,
    DEFAULT_MAX_TOKENS,
)

logger = logging.getLogger(__name__)


@dataclass
class CompressedContext:
    """Result of context compression."""

    messages: list[dict] = field(default_factory=list)
    mermaid_graph: str | None = None
    drilldown_refs: dict[str, int] = field(default_factory=dict)
    tokens_before: int = 0
    tokens_after: int = 0
    stage: str = "ok"


# Critical patterns — messages containing these NEVER get compressed away
CRITICAL_PATTERNS: list[str] = [
    r"(задач[ауие]|task|todo|сдела[тьйю]|напомни|дедлайн|срок)",
    r"(\d{1,2}[.:]\d{2})",  # time markers (13:00, 9.30)
    r"(\b[A-ZА-Я][a-zа-я]+\s[A-ZА-Я][a-zа-я]+\b)",  # potential names "Иван Петров"
    r"(обязательно|обязательств[ао]|договорились|обещал|promise)",
    r"(пароль|password|ключ|key|токен|token|секрет|secret)",
    r"(адрес|address|телефон|phone|почта|email)",
]


def _is_critical(message: dict) -> bool:
    """Check if a message contains critical information that must be preserved."""
    text = str(message.get("content", ""))
    for pattern in CRITICAL_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def _stage1_format(messages: list[dict]) -> list[dict]:
    """Stage 1: light formatting — remove double newlines, trim whitespace."""
    result = []
    for msg in messages:
        content = str(msg.get("content", ""))
        # Collapse 3+ newlines into 2
        content = re.sub(r"\n{3,}", "\n\n", content)
        # Trim trailing whitespace per line
        content = "\n".join(line.rstrip() for line in content.split("\n"))
        if content.strip():  # skip empty messages
            result.append({**msg, "content": content})
    return result


async def _stage2_summarise(
    messages: list[dict],
    critical_indices: set[int],
    session: Any = None,
    user: Any = None,
    max_summary_tokens: int = 500,
) -> list[dict]:
    """Stage 2: LLM summarise oldest non-critical messages.

    Requires ``session`` (DB session) and ``user`` (User model) to build
    a provider.  If either is missing the stage is silently skipped.
    """
    # Split: oldest 40% → summarise, rest → keep as-is
    split = max(1, len(messages) * 2 // 5)

    to_summarise = [
        m for i, m in enumerate(messages[:split]) if i not in critical_indices
    ]
    to_keep = [m for i, m in enumerate(messages[:split]) if i in critical_indices]

    if not to_summarise:
        return messages
    if session is None or user is None:
        logger.debug("Stage 2 skipped: no session/user provided")
        return messages

    try:
        from src.llm.base import ChatMessage, TaskType
        from src.llm.router import build_provider

        # Build summary text
        conversation_text = "\n".join(
            f"[{m.get('role', '?')}]: {m.get('content', '')[:500]}"
            for m in to_summarise[:20]  # max 20 messages in summary context
        )

        provider = await build_provider(
            session=session,
            user=user,
            purpose="background",
            task_type=TaskType.SUMMARIZE,
        )
        if provider:
            chat_messages: list[ChatMessage] = [
                ChatMessage(
                    role="system",
                    content=(
                        "Ты сжимаешь историю диалога. Сохрани: имена, договорённости, "
                        "даты, цифры, важные решения. Убери: приветствия, small talk, "
                        "повторы. Формат: 1-3 предложения, только суть."
                    ),
                ),
                ChatMessage(
                    role="user",
                    content=f"Сожми этот фрагмент диалога:\n\n{conversation_text}",
                ),
            ]
            summary = await provider.chat(chat_messages)
            if summary is None:
                summary = ""
            summary_msg = {
                "role": "system",
                "content": f"📋 [Саммари]: {summary.strip()}",
            }
            result = [summary_msg] + to_keep + messages[split:]
            return result
    except Exception:
        logger.debug("Stage 2 summary failed, keeping original", exc_info=True)

    return messages


def _stage3_mermaid(
    messages: list[dict],
    critical_indices: set[int],
) -> tuple[str | None, dict[str, int]]:
    """Stage 3: convert conversation to Mermaid graph with drill-down refs."""
    nodes: list[str] = []
    edges: list[str] = []
    refs: dict[str, int] = {}

    participants: dict[str, str] = {}  # role → display name
    node_counter = 0

    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = str(msg.get("content", ""))
        is_crit = i in critical_indices

        # Assign short label
        if role not in participants:
            participants[role] = role[:4].upper()

        node_id = f"N{node_counter}"
        label = content[:60].replace('"', "'").replace("\n", " ")
        if is_crit:
            label = f"⚠️ {label}"
        if len(content) > 60:
            label += "…"

        nodes.append(f'    {node_id}["{label}"]')
        refs[node_id] = i  # drill-down: node → message index

        if node_counter > 0:
            edges.append(f"    N{node_counter - 1} --> {node_id}")

        node_counter += 1

    if not nodes:
        return None, {}

    mermaid = "graph TD\n" + "\n".join(nodes) + "\n" + "\n".join(edges)
    return mermaid, refs


async def compress_context(
    messages: list[dict],
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    system_prompt: str = "",
    user_prompt: str = "",
    session: Any = None,
    user: Any = None,
) -> CompressedContext:
    """Adaptive 3-stage context compression.

    Args:
        messages: Conversation history list of {"role": str, "content": str}
        max_tokens: Model token budget
        system_prompt: Current system prompt (for budget calculation)
        user_prompt: Current user prompt (for budget calculation)
        session: Optional DB session for LLM provider (required for stage 2).
        user: Optional User model for LLM provider (required for stage 2).

    Returns:
        CompressedContext with compressed messages and optional Mermaid graph.
    """
    if not messages:
        return CompressedContext(messages=[], stage="ok")

    # Calculate current token usage
    total = count_prompt_tokens(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        history=messages,
    )
    stage, ratio = get_budget_stage(total, max_tokens)
    logger.info(
        "Context offload: %d tokens (%.0f%%), stage=%s", total, ratio * 100, stage
    )

    result = CompressedContext(
        messages=list(messages),  # copy
        tokens_before=total,
        stage=stage,
    )

    # Stage 1: Format (always applies if needed)
    if stage in ("format", "summary", "mermaid"):
        result.messages = _stage1_format(result.messages)

    # Identify critical messages AFTER Stage 1, so indices are not stale
    critical_indices: set[int] = {
        i for i, m in enumerate(result.messages) if _is_critical(m)
    }

    # Stage 2: LLM Summary
    if stage in ("summary", "mermaid"):
        result.messages = await _stage2_summarise(
            result.messages,
            critical_indices,
            session=session,
            user=user,
        )

    # Stage 3: Mermaid graph
    if stage == "mermaid":
        mermaid, refs = _stage3_mermaid(result.messages, critical_indices)
        result.mermaid_graph = mermaid
        result.drilldown_refs = refs
        # Replace message list with mermaid + critical messages only
        critical_msgs = [
            m for i, m in enumerate(result.messages) if i in critical_indices
        ]
        result.messages = critical_msgs

    # Recalculate tokens after compression
    result.tokens_after = count_prompt_tokens(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        history=result.messages,
    )

    logger.info(
        "Offload: %d → %d tokens (saved %d, stage=%s)",
        result.tokens_before,
        result.tokens_after,
        result.tokens_before - result.tokens_after,
        stage,
    )

    return result


__all__ = [
    "CompressedContext",
    "compress_context",
    "CRITICAL_PATTERNS",
]
