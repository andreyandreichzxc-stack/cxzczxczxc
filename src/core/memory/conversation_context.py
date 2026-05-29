"""Краткосрочная память диалога с control-bot: последние N ходов и последний
обсуждаемый контакт. Нужно чтобы «напиши ему», «в том же чате» правильно
резолвились без повторения имени. Persists compressed summaries to DB,
survives restarts."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from time import time

from src.config import settings

MAX_TURNS = (
    settings.context_max_turns
)  # порог сжатия: при превышении старые ходы сворачиваются в summary
_DEQUE_SAFETY_CAP = MAX_TURNS * 2  # запас для deque, чтобы не терять ходы до сжатия
LAST_PEER_TTL_SECONDS = 30 * 60  # 30 минут

_STALE_CTX_TTL = 3600  # 1 час — контексты с last_peer_at == 0 или старше удаляются


@dataclass
class _Ctx:
    turns: deque = field(default_factory=lambda: deque(maxlen=_DEQUE_SAFETY_CAP))
    compressed: str | None = None  # сжатая сводка старых ходов
    last_peer_id: int | None = None
    last_peer_name: str | None = None
    last_peer_at: float = 0.0
    last_purpose: str | None = None
    transcription_meta: dict | None = (
        None  # метаданные последней голосовой транскрипции
    )
    created_at: float = field(default_factory=time)


_STORE: dict[int, _Ctx] = {}
_ctx_lock = asyncio.Lock()


def _cleanup_stale_contexts() -> None:
    """Удаляет контексты, где created_at старше _STALE_CTX_TTL."""
    now = time()
    stale = [
        uid
        for uid, ctx in list(_STORE.items())
        if (now - ctx.created_at) > _STALE_CTX_TTL
    ]
    for uid in stale:
        del _STORE[uid]


async def _get(user_id: int) -> _Ctx:
    async with _ctx_lock:
        _cleanup_stale_contexts()
        ctx = _STORE.get(user_id)
        if ctx is None:
            ctx = _Ctx()
            _STORE[user_id] = ctx
            # Try to load recent summaries from DB on first access
            compressed = await load_recent_summaries(user_id)
            if compressed:
                ctx.compressed = "[Предыдущий диалог]\n" + compressed
        return ctx


def _quick_summarize(
    turns: list[tuple[float, str, str]],
) -> str:
    """Свернуть старые ходы в компактную текстовую сводку."""
    lines: list[str] = []
    for _ts, user_text, assistant_summary in turns:
        if user_text:
            lines.append(f"Владелец: {user_text[:200]}")
        if assistant_summary:
            lines.append(f"Я ответил: {assistant_summary[:200]}")
    if len(lines) > 30:
        lines = lines[:30] + ["…[и ещё]"]
    return "\n".join(lines)


async def add_turn(user_id: int, user_text: str, assistant_summary: str) -> None:
    ctx = await _get(user_id)
    user_text = (user_text or "").strip()
    assistant_summary = (assistant_summary or "").strip()
    if not user_text and not assistant_summary:
        return
    async with _ctx_lock:
        ctx.turns.append((time(), user_text[:400], assistant_summary[:400]))

        # Авто-сжатие: если ходов стало больше порога — сворачиваем старые
        if len(ctx.turns) > MAX_TURNS:
            turns_list = list(ctx.turns)
            old_turns = turns_list[:-10]  # все, кроме последних 10
            ctx.turns = deque(turns_list[-10:], maxlen=_DEQUE_SAFETY_CAP)
            ctx.compressed = f"[Предыдущий диалог]: {_quick_summarize(old_turns)}"
            # Persist compressed summary to DB (fire-and-forget)
            asyncio.ensure_future(_save_summary_to_db(user_id, ctx))


async def set_last_peer(user_id: int, peer_id: int, peer_name: str | None) -> None:
    ctx = await _get(user_id)
    async with _ctx_lock:
        ctx.last_peer_id = peer_id
        ctx.last_peer_name = peer_name
        ctx.last_peer_at = time()


async def get_last_peer(user_id: int) -> tuple[int, str | None] | None:
    ctx = await _get(user_id)
    if ctx.last_peer_id is None:
        return None
    if (time() - ctx.last_peer_at) > LAST_PEER_TTL_SECONDS:
        return None
    return ctx.last_peer_id, ctx.last_peer_name


async def get_recent_turns(
    user_id: int,
) -> list[tuple[str, str]]:
    """Returns recent turns (user_text, assistant_summary).

    Does **not** include the ``compressed`` summary — callers that need it
    should read ``ctx.compressed`` separately via ``_get()`` or use
    ``render_history_block()``.
    """
    ctx = await _get(user_id)
    rows: list[tuple[str, str]] = []
    for item in ctx.turns:
        if len(item) == 3:
            _, user_text, assistant_summary = item
            rows.append((user_text, assistant_summary))
        else:
            rows.append(item)
    return rows


async def get_recent_turn_count(user_id: int, max_age_seconds: int = 3600) -> int:
    now = time()
    count = 0
    for item in (await _get(user_id)).turns:
        if len(item) == 3:
            ts = item[0]
            if now - ts <= max_age_seconds:
                count += 1
        else:
            count += 1
    return count


async def set_last_purpose(user_id: int, purpose: str) -> None:
    """Запоминает последний purpose для context chaining."""
    ctx = await _get(user_id)
    async with _ctx_lock:
        ctx.last_purpose = purpose


async def get_last_purpose(user_id: int) -> str | None:
    """Возвращает последний purpose для context chaining."""
    ctx = await _get(user_id)
    return ctx.last_purpose


# ---------------------------------------------------------------------------
# Persistent conversation summaries (DB-backed, survives restarts)
# ---------------------------------------------------------------------------


async def _save_summary_to_db(user_id: int, ctx: _Ctx) -> None:
    """Fire-and-forget: persist compressed summary to DB."""
    if not ctx.compressed:
        return
    try:
        from src.db.session import get_session
        from src.db.models._messaging import ConversationSummary

        async with get_session() as session:
            summary = ConversationSummary(
                user_id=user_id,
                last_peer_id=ctx.last_peer_id,
                last_peer_name=ctx.last_peer_name,
                summary_text=ctx.compressed[:2000],
                turn_count=len(ctx.turns),
            )
            session.add(summary)
            await session.commit()
    except Exception:
        import logging

        logging.getLogger(__name__).debug(
            "Failed to persist conversation summary", exc_info=True
        )


async def load_recent_summaries(user_id: int) -> str | None:
    """Load recent summaries from DB and combine into one context string."""
    try:
        from datetime import datetime, timedelta, timezone

        from sqlalchemy import desc, select

        from src.db.session import get_session
        from src.db.models._messaging import ConversationSummary

        async with get_session() as session:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
            result = await session.execute(
                select(ConversationSummary)
                .where(
                    ConversationSummary.user_id == user_id,
                    ConversationSummary.created_at >= cutoff,
                )
                .order_by(desc(ConversationSummary.created_at))
                .limit(5)
            )
            rows = result.scalars().all()
            if not rows:
                return None
            parts: list[str] = []
            for r in rows:
                peer = f"с {r.last_peer_name}" if r.last_peer_name else ""
                parts.append(
                    f"[{r.created_at.strftime('%d.%m %H:%M')} {peer}]\n{r.summary_text[:500]}"
                )
            return "\n\n".join(parts)
    except Exception:
        import logging

        logging.getLogger(__name__).debug(
            "Failed to load conversation summaries", exc_info=True
        )
        return None


async def set_transcription_meta(user_id: int, meta: dict) -> None:
    """Сохраняет метаданные транскрипции голосового сообщения для инжекта в промпт."""
    ctx = await _get(user_id)
    async with _ctx_lock:
        ctx.transcription_meta = meta


async def get_and_clear_transcription_meta(user_id: int) -> dict | None:
    """Читает и очищает метаданные транскрипции (одноразовое использование)."""
    ctx = await _get(user_id)
    async with _ctx_lock:
        meta = ctx.transcription_meta
        ctx.transcription_meta = None
        return meta


async def cleanup_old_summaries() -> None:
    """Delete conversation summaries older than 7 days."""
    try:
        from datetime import datetime, timedelta, timezone

        from sqlalchemy import delete

        from src.db.session import get_session
        from src.db.models._messaging import ConversationSummary

        async with get_session() as session:
            cutoff = datetime.now(timezone.utc) - timedelta(days=7)
            await session.execute(
                delete(ConversationSummary).where(
                    ConversationSummary.created_at < cutoff
                )
            )
            await session.commit()
    except Exception:
        import logging

        logging.getLogger(__name__).debug(
            "Failed to cleanup old summaries", exc_info=True
        )


async def render_history_block(user_id: int) -> str:
    parts: list[str] = []
    last = await get_last_peer(user_id)
    if last is not None:
        peer_id, name = last
        label = name or str(peer_id)
        parts.append(
            f"Последний упомянутый контакт: {label} (peer_id={peer_id}). "
            f"Если фраза вроде «ему», «ей», «в том же чате», «там» — "
            f"подставляй именно его."
        )

    ctx = await _get(user_id)

    # Сжатая сводка старых ходов
    history_lines: list[str] = []
    if ctx.compressed:
        history_lines.append(ctx.compressed)

    # Последние 10 ходов (детально)
    recent = list(ctx.turns)[-10:]
    if recent:
        history_lines.append(
            "Недавний диалог с владельцем (для понимания «то/там/ему»):"
        )
        for item in recent:
            if len(item) == 3:
                _, u, a = item
            else:
                u, a = item
            if u:
                history_lines.append(f"  Владелец: {u}")
            if a:
                history_lines.append(f"  Я ответил: {a}")

    if history_lines:
        parts.append("\n".join(history_lines))

    return "\n\n".join(parts)
