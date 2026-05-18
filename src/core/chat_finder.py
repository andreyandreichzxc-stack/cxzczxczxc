"""Поиск релевантного чата по теме: keyword expansion (LLM) + локальный FTS5 +
LLM-классификация по именам контактов. Telegram global search — fallback на пустой БД.

Зачем имена: иногда в тексте чата нужного слова вообще нет (контакт называется
«Кул Хаус Магазин» — а тема «мебель»). LLM видит это по названию."""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

from telethon import TelegramClient
from telethon.tl.types import User as TgUser
from telethon.utils import get_peer_id

from src.db.models import Contact, User
from src.db.repo import fts_search, list_contacts
from src.db.session import get_session
from src.llm.base import ChatMessage, LLMProvider


logger = logging.getLogger(__name__)


@dataclass
class FoundChat:
    peer_id: int
    name: str
    sample: str
    text_hits: int
    name_score: int
    kind: str
    is_bot: bool
    username: str | None

    @property
    def total_score(self) -> int:
        return self.text_hits * 2 + self.name_score


_EXPAND_SYS = (
    "Ты помогаешь искать переписки в Telegram. По теме запроса верни JSON-объект "
    '{"keywords": [строка, ...]} — это список из 5–12 коротких ключевых слов или '
    "коротких фраз (1–2 слова), включая синонимы, родовые/видовые термины и переводы "
    "на ту же тему на украинском И английском. Не используй мусор-стоп-слова."
)


_CLASSIFY_SYS = (
    "Тебе дан список имён контактов и тема. Верни JSON-объект "
    '{"matches": [{"peer_id": int, "score": int}, ...]} — где score 1..5 — насколько '
    "название контакта семантически связано с темой (5 — точно, 1 — слабая ассоциация). "
    "Возвращай только peer_id из переданного списка. Контакты без явной связи можно "
    "не упоминать. До 8 кандидатов."
)


def _strip_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    return text


def _safe_json(text: str) -> Any:
    try:
        return json.loads(_strip_fence(text))
    except Exception:
        return None


async def _expand_keywords(provider: LLMProvider, query: str) -> list[str]:
    raw = await provider.chat(
        [
            ChatMessage(role="system", content=_EXPAND_SYS),
            ChatMessage(role="user", content=query),
        ],
        heavy=False,
    )
    parsed = _safe_json(raw) or {}
    kws = parsed.get("keywords") if isinstance(parsed, dict) else None
    if not isinstance(kws, list):
        return [query]
    cleaned: list[str] = []
    seen = set()
    for k in kws:
        if not isinstance(k, str):
            continue
        k = k.strip().lower()
        if not k or k in seen:
            continue
        seen.add(k)
        cleaned.append(k)
    if query.lower() not in seen:
        cleaned.insert(0, query)
    return cleaned[:12]


async def _classify_contacts(provider: LLMProvider, query: str, contacts: list[Contact]) -> dict[int, int]:
    if not contacts:
        return {}
    # Готовим компактный список для LLM (peer_id, name, kind)
    items = []
    for c in contacts[:300]:
        kind = "bot" if c.is_bot else c.peer_kind
        items.append({"peer_id": c.peer_id, "name": c.display_name, "kind": kind})
    payload = json.dumps({"topic": query, "contacts": items}, ensure_ascii=False)

    raw = await provider.chat(
        [
            ChatMessage(role="system", content=_CLASSIFY_SYS),
            ChatMessage(role="user", content=payload),
        ],
        heavy=False,
    )
    parsed = _safe_json(raw) or {}
    matches = parsed.get("matches") if isinstance(parsed, dict) else None
    out: dict[int, int] = {}
    if isinstance(matches, list):
        for m in matches:
            if not isinstance(m, dict):
                continue
            try:
                pid = int(m.get("peer_id"))
                score = int(m.get("score") or 0)
            except (TypeError, ValueError):
                continue
            if 1 <= score <= 5:
                out[pid] = max(out.get(pid, 0), score)
    return out


async def _local_keyword_search(
    owner: User,
    keywords: list[str],
    contact_by_pid: dict[int, Contact],
    *,
    per_kw_limit: int = 50,
) -> dict[int, dict]:
    found: dict[int, dict] = {}
    async with get_session() as session:
        for kw in keywords:
            hits = await fts_search(session, owner.id, kw, limit=per_kw_limit)
            for h in hits:
                entry = found.get(h.peer_id)
                if entry is None:
                    c = contact_by_pid.get(h.peer_id)
                    found[h.peer_id] = {
                        "count": 1,
                        "sample": h.snippet or "",
                        "name": (c.display_name if c else str(h.peer_id)),
                        "kind": (c.peer_kind if c else "user"),
                        "is_bot": (c.is_bot if c else False),
                        "username": (c.username if c else None),
                    }
                else:
                    entry["count"] += 1
    return found


async def _telegram_keyword_search(
    client: TelegramClient,
    keyword: str,
    *,
    per_kw_limit: int = 20,
) -> dict[int, dict]:
    # fallback на пустой БД — Telegram global search
    found: dict[int, dict] = {}
    try:
        async for msg in client.iter_messages(None, search=keyword, limit=per_kw_limit):
            try:
                pid = get_peer_id(msg.peer_id) if msg.peer_id else (msg.chat_id or 0)
            except Exception:
                continue
            if not pid:
                continue
            entry = found.get(pid)
            sample = (msg.text or msg.message or "")[:160]
            if entry is None:
                try:
                    entity = await msg.get_chat()
                except Exception:
                    entity = None
                if isinstance(entity, TgUser):
                    parts = [getattr(entity, "first_name", None), getattr(entity, "last_name", None)]
                    name = " ".join(p for p in parts if p).strip() or (entity.username or str(pid))
                    is_bot = bool(getattr(entity, "bot", False))
                    kind = "user"
                    username = getattr(entity, "username", None)
                else:
                    name = getattr(entity, "title", None) or str(pid)
                    is_bot = False
                    kind = "channel" if entity and getattr(entity, "broadcast", False) else "chat"
                    username = getattr(entity, "username", None)
                found[pid] = {
                    "count": 1, "sample": sample, "name": name,
                    "kind": kind, "is_bot": is_bot, "username": username,
                }
            else:
                entry["count"] += 1
    except Exception:
        logger.exception("telegram search keyword failed: %s", keyword)
    return found


async def smart_find(
    client: TelegramClient,
    owner: User,
    provider: LLMProvider,
    query: str,
    *,
    top_n: int = 5,
    per_kw_limit: int = 20,
) -> list[FoundChat]:
    keywords = await _expand_keywords(provider, query)

    async with get_session() as session:
        contacts = await list_contacts(session, owner)
    contact_by_pid = {c.peer_id: c for c in contacts}

    local_task = _local_keyword_search(owner, keywords, contact_by_pid, per_kw_limit=per_kw_limit)
    name_task = _classify_contacts(provider, query, contacts)
    local_hits, name_scores = await asyncio.gather(local_task, name_task)

    if not local_hits:
        tg_results = await asyncio.gather(*[
            _telegram_keyword_search(client, k, per_kw_limit=per_kw_limit) for k in keywords
        ])
        merged: dict[int, dict] = {}
        for batch in tg_results:
            for pid, info in batch.items():
                m = merged.get(pid)
                if m is None:
                    merged[pid] = info.copy()
                else:
                    m["count"] += info["count"]
        local_hits = merged

    aggregated: dict[int, dict] = {}
    for pid, info in local_hits.items():
        aggregated[pid] = {
            "text_hits": info["count"],
            "sample": info["sample"],
            "name": info["name"],
            "kind": info["kind"],
            "is_bot": info["is_bot"],
            "username": info["username"],
        }

    # имена-кандидаты, которые не попали в текстовые хиты — добавляем отдельно
    for pid, score in name_scores.items():
        agg = aggregated.get(pid)
        if agg is None:
            c = contact_by_pid.get(pid)
            if c is None:
                continue
            aggregated[pid] = {
                "text_hits": 0,
                "sample": "",
                "name": c.display_name,
                "kind": c.peer_kind,
                "is_bot": c.is_bot,
                "username": c.username,
                "name_score": score,
            }
        else:
            agg["name_score"] = score

    out: list[FoundChat] = []
    for pid, agg in aggregated.items():
        if agg.get("is_bot"):
            continue
        out.append(FoundChat(
            peer_id=pid,
            name=agg["name"],
            sample=agg.get("sample") or "",
            text_hits=agg.get("text_hits", 0),
            name_score=agg.get("name_score", 0),
            kind=agg["kind"],
            is_bot=agg.get("is_bot", False),
            username=agg.get("username"),
        ))
    out.sort(key=lambda f: f.total_score, reverse=True)
    return out[:top_n]
