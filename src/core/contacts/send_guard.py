"""Send Guard — единый предохранитель перед отправкой сообщения с undo-буфером."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from src.core.memory.temporal_layers import utc_naive, utcnow_naive
from src.db.session import get_session
from src.db.repo import (
    get_or_create_user,
    get_contact,
    list_memories,
    get_contact_profile,
)

logger = logging.getLogger(__name__)

_undo_buffer: dict[int, list] = {}


@dataclass
class SendGuardResult:
    risk_level: str = "low"
    warnings: list[str] = field(default_factory=list)
    memory_hints: list[str] = field(default_factory=list)
    profile_hints: list[str] = field(default_factory=list)

    @property
    def formatted_html(self) -> str:
        parts = []
        if self.warnings:
            parts.append("\n".join(f"⚠️ {w}" for w in self.warnings))
        if self.memory_hints:
            parts.append("\n".join(f"🧠 {h}" for h in self.memory_hints))
        if self.profile_hints:
            parts.append("\n".join(f"👤 {h}" for h in self.profile_hints))
        return "\n".join(parts)


async def build_send_guard(
    telegram_id: int, peer_id: int, draft_text: str = ""
) -> SendGuardResult:
    result = SendGuardResult(risk_level="low")
    now = utcnow_naive()

    async with get_session() as session:
        owner = await get_or_create_user(session, telegram_id)
        contact = await get_contact(session, owner, peer_id)
        name = contact.display_name if contact else str(peer_id)

        mems = await list_memories(session, owner)
        neg = [
            m
            for m in mems
            if m.contact_id == peer_id
            and m.sentiment == "negative"
            and m.is_active
            and m.created_at
            and (now - utc_naive(m.created_at)).days < 14
        ]
        if neg:
            result.risk_level = "high"
            neg_texts = "; ".join(m.fact[:50] for m in neg[:3])
            result.warnings.append(
                f"За последние 2 недели негативные факты о {name}: {neg_texts}"
            )

        try:
            prof = await get_contact_profile(session, owner, peer_id)
            if prof:
                if prof.communication_style:
                    result.profile_hints.append(
                        f"Стиль: {prof.communication_style[:60]}"
                    )
                if prof.communication_dos:
                    import json as _j

                    dos = (
                        _j.loads(prof.communication_dos)
                        if prof.communication_dos.startswith("[")
                        else [prof.communication_dos]
                    )
                    if dos:
                        result.profile_hints.append(f"✅ {', '.join(dos[:3])}")
                if prof.communication_donts:
                    import json as _j

                    donts = (
                        _j.loads(prof.communication_donts)
                        if prof.communication_donts.startswith("[")
                        else [prof.communication_donts]
                    )
                    if donts:
                        result.profile_hints.append(f"❌ {', '.join(donts[:3])}")
                if prof.sensitivity and prof.sensitivity > 0.7:
                    result.risk_level = "high"
                    result.warnings.append(
                        "Высокая чувствительность контакта — будь аккуратнее."
                    )
        except Exception:
            logger.debug("send_guard: profile check skipped", exc_info=True)
            pass

        if contact and contact.archetype == "toxic":
            if result.risk_level != "high":
                result.risk_level = "medium"
            result.warnings.append("Конфликтный контакт — перепроверь сообщение.")

    return result


def store_undo(telegram_id: int, peer_id: int, message_id: int, text: str) -> None:
    if telegram_id not in _undo_buffer:
        _undo_buffer[telegram_id] = []
    _undo_buffer[telegram_id].append(
        (peer_id, message_id, text, datetime.now(timezone.utc))
    )
    _undo_buffer[telegram_id] = [
        (p, m, t, ts)
        for p, m, t, ts in _undo_buffer[telegram_id]
        if (datetime.now(timezone.utc) - ts).total_seconds() < 300
    ]


def get_undo(telegram_id: int) -> tuple | None:
    if telegram_id not in _undo_buffer or not _undo_buffer[telegram_id]:
        return None
    last = _undo_buffer[telegram_id][-1]
    age = (datetime.now(timezone.utc) - last[3]).total_seconds()
    if age > 60:
        return None
    return last
