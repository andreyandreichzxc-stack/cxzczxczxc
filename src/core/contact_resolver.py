from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from rapidfuzz import fuzz, process
from telethon import TelegramClient

from src.db.models import Contact, User
from src.db.repo import list_contacts
from src.db.session import get_session
from src.userbot.dialogs import sync_dialogs

logger = logging.getLogger(__name__)


@dataclass
class ContactCandidate:
    peer_id: int
    display_name: str
    username: str | None
    peer_kind: str
    score: int

    def label(self) -> str:
        if self.username:
            return f"{self.display_name} (@{self.username})"
        return self.display_name


def _searchable(c: Contact) -> str:
    parts = [c.display_name]
    if c.username:
        parts.append("@" + c.username)
    if c.phone:
        parts.append(c.phone)
    return " | ".join(parts)


async def resolve(
    client: TelegramClient,
    user: User,
    query: str,
    *,
    limit: int = 5,
    min_score: int = 55,
    kinds: tuple[str, ...] = ("user",),
    include_bots: bool = False,
) -> list[ContactCandidate]:
    # По умолчанию ищем только людей. Для каналов/групп kinds расширять явно.
    async with get_session() as session:
        contacts = await list_contacts(
            session, user, kinds=kinds, include_bots=include_bots
        )

    if not contacts:
        await sync_dialogs(client, user)
        async with get_session() as session:
            contacts = await list_contacts(
                session, user, kinds=kinds, include_bots=include_bots
            )

    if not contacts:
        return []

    choices = {c.peer_id: _searchable(c) for c in contacts}
    raw = process.extract(
        query,
        choices,
        scorer=fuzz.WRatio,
        limit=limit,
    )

    by_id = {c.peer_id: c for c in contacts}
    results: list[ContactCandidate] = []
    for _, score, peer_id in raw:
        if score < min_score:
            continue
        c = by_id[peer_id]
        results.append(
            ContactCandidate(
                peer_id=c.peer_id,
                display_name=c.display_name,
                username=c.username,
                peer_kind=c.peer_kind,
                score=int(score),
            )
        )
    return results


CONTACT_DISAMBIGUATION_PROMPT = """Ты — система разрешения контактов. Пользователь написал имя/описание человека, которому хочет отправить сообщение. Тебе дан список кандидатов из его телефонной книги. Выбери НАИБОЛЕЕ подходящего.

## Правила
- Учитывай падежи: «Насте» → «Настя», «Оле» → «Оля», «маме» → «мама»
- Учитывай уменьшительные формы: «Настюша» → «Настя», «Сашуля» → «Саша»
- Учитывай неформальные имена: «батя» → отец, «мама» → мать, «брат» → «Брат», «сеструха» → сестра
- Игнорируй username и телефон при выборе — смотри ТОЛЬКО на имя
- Если несколько кандидатов с одинаковым именем — выбери того, у кого имя ТОЧНО совпадает
- Если запрос «батя» а есть контакт «Батя» или «Отец» или «Папа» — выбери его
- Если запрос «Настя» а есть контакт «Настя Иванова» и контакт «батя» — выбери «Настя Иванова»

## Формат ответа
Верни ТОЛЬКО JSON:
{"selected_index": 0, "confidence": "high|medium|low", "reason": "почему выбран именно этот"}

Если НИ ОДИН кандидат не подходит — верни:
{"selected_index": -1, "confidence": "none", "reason": "почему"}"""


async def resolve_with_llm(
    client: TelegramClient,
    user: User,
    query: str,
    provider,  # LLMProvider
    *,
    limit: int = 10,
    min_score: int = 50,
    min_confident_score: int = 90,
    kinds: tuple[str, ...] = ("user",),
    include_bots: bool = False,
) -> list[ContactCandidate]:
    """Разрешает контакт: fuzzy сначала, LLM для дисамбигуации если нужно.

    В отличие от resolve(), эта функция не полагается только на fuzzy matching.
    Если найдено несколько кандидатов — используется LLM для семантического выбора.
    """
    # Шаг 1: fuzzy matching с щедрым лимитом
    candidates = await resolve(
        client,
        user,
        query,
        limit=limit,
        min_score=min_score,
        kinds=kinds,
        include_bots=include_bots,
    )

    if not candidates:
        return []

    # Шаг 2: высокая уверенность — возвращаем без LLM
    if len(candidates) == 1 and candidates[0].score >= min_confident_score:
        return [candidates[0]]

    # Шаг 3: если всего один кандидат с низким score — всё равно возвращаем
    if len(candidates) == 1:
        return candidates

    # Шаг 4: LLM дисамбигуация
    if provider is None:
        # Нет LLM — возвращаем всех кандидатов, пусть пользователь выберет
        return candidates

    # Строим описание кандидатов для LLM
    candidate_lines = []
    for i, c in enumerate(candidates):
        extra = f" (@{c.username})" if c.username else ""
        candidate_lines.append(f"[{i}] {c.display_name}{extra} (fuzzy_score={c.score})")

    try:
        from src.llm.base import ChatMessage

        user_prompt = (
            f"Запрос пользователя: «{query}»\n\n"
            f"Кандидаты из телефонной книги:\n" + "\n".join(candidate_lines)
        )

        raw = await provider.chat(
            [
                ChatMessage(role="system", content=CONTACT_DISAMBIGUATION_PROMPT),
                ChatMessage(role="user", content=user_prompt),
            ],
            heavy=False,
        )
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            parsed = json.loads(m.group(0))
            idx = parsed.get("selected_index", -1)
            if isinstance(idx, int) and 0 <= idx < len(candidates):
                confidence = parsed.get("confidence", "medium")
                logger.info(
                    "LLM selected %r for query %r (confidence=%s)",
                    candidates[idx].display_name,
                    query,
                    confidence,
                )
                # Поднимаем score выбранного кандидата — чтобы гарантированно прошёл авто-приём
                if confidence == "high":
                    candidates[idx].score = max(candidates[idx].score, 95)
                elif confidence == "medium":
                    candidates[idx].score = max(candidates[idx].score, 85)
                # Перемещаем выбранного на первое место
                selected = candidates.pop(idx)
                candidates.insert(0, selected)
    except Exception:
        logger.exception("LLM disambiguation failed for query %r", query)

    return candidates
