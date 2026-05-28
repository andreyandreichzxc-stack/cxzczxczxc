"""Unified service layer for chat actions.

Вытесняет дублирование между:
- chat_cmd.py (callback handlers: cb_summary, cb_tasks, cb_draft, cb_catchup)
- free_text_exec.py (classic intent handlers: exec_classic_*)

Каждая action-функция:
1. Загружает контекст (клиент, контакт, сообщения, LLM-провайдер)
2. Вызывает LLM
3. Возвращает структурированный результат — отображение остаётся за вызывающим хендлером.
"""

from __future__ import annotations

import json as _json
import logging
from dataclasses import dataclass, field

from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)

from src.core.humanizer import (
    analyze_ai_score as _analyze_ai_score,
    humanize_deep as _humanize_deep,
    humanize_response as _humanize_response,
)
from src.core.infra.text_sanitizer import sanitize_html as _sanitize_html
from src.core.contacts.chat_service import load_chat as _load_chat
from src.core.contacts.contact_memory_digest import (
    get_contact_digest as _get_contact_digest,
)
from src.core.actions.commitment_extractor import (
    extract_and_save_commitments as _extract_commitments,
)
from src.core.intelligence.summarizer import (
    ask_chat as _ask_chat,
    catchup as _catchup,
    draft_reply as _draft_reply,
    summarize_chat as _summarize_chat,
)
from src.core.intelligence.style_matcher import (
    get_or_update_style_profile as _get_style_profile,
)
from src.db.repo import (
    create_pending_action,
    get_contact as _get_contact,
    get_or_create_user as _get_or_create_user,
)
from src.db.session import get_session
from src.llm.base import TaskType
from src.llm.router import build_provider
from src.userbot.manager import UserbotManager


logger = logging.getLogger(__name__)


# ── Result types ────────────────────────────────────────────────────────


@dataclass
class ChatActionResult:
    """Базовый результат чат-действия."""

    html: str  # HTML-текст результата
    display_name: str  # имя контакта для заголовка
    markup: InlineKeyboardMarkup | None = None  # инлайн-клавиатура
    raw_items: list[dict] = field(default_factory=list)  # извлечённые данные


# ── Shared context loading ──────────────────────────────────────────────


async def _load_chat_context(
    telegram_id: int,
    peer_id: int,
    userbot_manager: UserbotManager,
    limit: int = 50,
) -> dict | None:
    """Загружает полный контекст для чат-действия.

    Returns dict с ключами: client, owner, contact, messages, provider
    или None если не удалось (ошибку уже отправил вызывающий).
    """
    client = userbot_manager.get_client(telegram_id)
    if client is None:
        return None  # вызывающий должен сообщить "/login"

    messages = await _load_chat(
        client, telegram_id, peer_id, limit=limit, transcribe=True
    )

    async with get_session() as session:
        owner = await _get_or_create_user(session, telegram_id)
        contact = await _get_contact(session, owner, peer_id)
        provider = await build_provider(session, owner, task_type=TaskType.DEFAULT)

    if contact is None or provider is None:
        return None

    return {
        "client": client,
        "owner": owner,
        "contact": contact,
        "messages": messages,
        "provider": provider,
    }


# ── Message count helper ─────────────────────────────────────────────────


async def get_chat_message_count(telegram_id: int, peer_id: int) -> int:
    """Возвращает общее количество сообщений в чате с peer_id."""
    from src.db.repo import count_messages as _count_messages
    from src.db.session import get_session as _get_session
    from src.db.repo import get_or_create_user as _get_user

    async with _get_session() as session:
        owner = await _get_user(session, telegram_id)
        return await _count_messages(session, owner, peer_id)


# ── Action implementations ──────────────────────────────────────────────


async def summarize_chat_action(
    telegram_id: int,
    peer_id: int,
    userbot_manager: UserbotManager,
    limit: int = 50,
) -> ChatActionResult | None:
    """Саммари последних сообщений с контактом."""
    ctx = await _load_chat_context(telegram_id, peer_id, userbot_manager, limit=limit)
    if ctx is None:
        return None

    text = await _summarize_chat(
        ctx["provider"],
        ctx["contact"],
        ctx["messages"],
        global_style=ctx["owner"].global_style_profile,
        owner_id=ctx["owner"].id,
    )
    html = f"📝 <b>Саммари — {ctx['contact'].display_name}</b>\n\n{text}"
    return ChatActionResult(
        html=html,
        display_name=ctx["contact"].display_name,
        markup=_actions_keyboard(peer_id),
    )


async def extract_tasks_action(
    telegram_id: int,
    peer_id: int,
    userbot_manager: UserbotManager,
    limit: int = 50,
) -> ChatActionResult | None:
    """Извлечение задач/обязательств из чата."""
    ctx = await _load_chat_context(telegram_id, peer_id, userbot_manager, limit=limit)
    if ctx is None:
        return None

    items = await _extract_commitments(
        ctx["provider"],
        telegram_id=ctx["owner"].telegram_id,
        contact_name=ctx["contact"].display_name,
        contact_peer_id=ctx["contact"].peer_id,
        messages=ctx["messages"],
    )

    if not items:
        body = "🤷 Явных обязательств не нашёл."
    else:
        lines = []
        for it in items:
            who = "Я" if it.get("direction") == "mine" else "Они"
            deadline = it.get("deadline")
            tail = f" · до {deadline}" if deadline else ""
            lines.append(f"• <b>{who}</b>: {it.get('text', '')}{tail}")
        body = "\n".join(lines)

    html = f"✅ <b>Обязательства — {ctx['contact'].display_name}</b>\n\n{body}"
    return ChatActionResult(
        html=html,
        display_name=ctx["contact"].display_name,
        markup=_actions_keyboard(peer_id),
        raw_items=items,
    )


async def draft_reply_action(
    telegram_id: int,
    peer_id: int,
    userbot_manager: UserbotManager,
    instruction: str = "",
    limit: int = 50,
) -> ChatActionResult | None:
    """Черновик ответа контакту. Создаёт pending action для подтверждения отправки."""
    ctx = await _load_chat_context(telegram_id, peer_id, userbot_manager, limit=limit)
    if ctx is None:
        return None

    draft_text = await _draft_reply(
        ctx["provider"],
        ctx["contact"],
        ctx["messages"],
        instruction=instruction or None,
        global_style=ctx["owner"].global_style_profile,
        owner_id=ctx["owner"].id,
    )

    payload = _json.dumps({"peer_id": peer_id, "text": draft_text}, ensure_ascii=False)

    async with get_session() as session:
        owner = await _get_or_create_user(session, telegram_id)
        action = await create_pending_action(
            session, user_id=owner.id, kind="send_message", payload=payload
        )

    kb = InlineKeyboardBuilder()
    kb.row(
        InlineKeyboardButton(
            text="✅ Отправить", callback_data=f"send:confirm:{action.id}"
        ),
        InlineKeyboardButton(
            text="❌ Отмена", callback_data=f"send:cancel:{action.id}"
        ),
    )

    html = (
        f"💬 <b>Черновик ответа — {ctx['contact'].display_name}</b>\n\n"
        f"{draft_text}\n\nОтправить?"
    )
    return ChatActionResult(
        html=html,
        display_name=ctx["contact"].display_name,
        markup=kb.as_markup(),
    )


async def catchup_action(
    telegram_id: int,
    peer_id: int,
    userbot_manager: UserbotManager,
    limit: int = 50,
) -> ChatActionResult | None:
    """«Где мы остановились» с контактом."""
    ctx = await _load_chat_context(telegram_id, peer_id, userbot_manager, limit=limit)
    if ctx is None:
        return None

    text = await _catchup(
        ctx["provider"],
        ctx["contact"],
        ctx["messages"],
        global_style=ctx["owner"].global_style_profile,
        owner_id=ctx["owner"].id,
    )

    html = f"⏪ <b>Где мы остановились — {ctx['contact'].display_name}</b>\n\n{text}"
    return ChatActionResult(
        html=html,
        display_name=ctx["contact"].display_name,
        markup=_actions_keyboard(peer_id),
    )


async def _load_memory_context(
    telegram_id: int,
    peer_id: int,
    contact_name: str,
    *,
    max_facts: int = 5,
) -> str:
    """Загружает память о контакте: факты, обещания, стиль.

    Пытается сначала через быстрый contact_digest (кеш),
    при нехватке фактов — fallback на recall().
    Возвращает пустую строку если памяти нет.
    """
    try:
        digest = await _get_contact_digest(telegram_id, peer_id)
        facts = digest.get("facts") or []
        if facts:
            lines = [f"<recall_context>\n📌 Факты о {_sanitize_html(contact_name)}:"]
            for f in facts[:max_facts]:
                lines.append(f"• {_sanitize_html(f.get('fact', ''))}")
            promises = digest.get("promises") or []
            if promises:
                lines.append("\n📋 Обещания:")
                for p in promises[:3]:
                    lines.append(f"• {_sanitize_html(p.get('text', ''))}")
            lines.append("</recall_context>")
            return "\n".join(lines)

        # Fallback: полный recall
        from src.core.memory.memory_recall import (
            recall as _recall,
            format_recall_for_prompt as _fmt_recall,
        )

        result = await _recall(
            telegram_id,
            contact_id=peer_id,
            query="",
            limit=max_facts,
            include_self=False,
            mode="light",
        )
        if result and result.facts:
            return _fmt_recall(result, max_facts=max_facts)
    except Exception:
        logger.warning(
            "Failed to load memory context for %s (peer_id=%s)",
            contact_name,
            peer_id,
            exc_info=True,
        )
    return ""


async def ask_chat_action(
    telegram_id: int,
    peer_id: int,
    userbot_manager: UserbotManager,
    user_query: str = "",
    limit: int = 50,
) -> ChatActionResult | None:
    """Проанализировать чат: задать вопрос LLM про переписку.

    Args:
        telegram_id: ID владельца
        peer_id: ID чата/контакта
        userbot_manager: менеджер userbot'ов
        user_query: вопрос пользователя
        limit: сколько сообщений загрузить
    """
    ctx = await _load_chat_context(telegram_id, peer_id, userbot_manager, limit=limit)
    if ctx is None:
        return None

    # Загружаем память о контакте
    contact_name = ctx["contact"].display_name
    memory_context = await _load_memory_context(telegram_id, peer_id, contact_name)

    text = await _ask_chat(
        ctx["provider"],
        ctx["contact"],
        ctx["messages"],
        user_query=user_query,
        global_style=ctx["owner"].global_style_profile,
        owner_id=ctx["owner"].id,
        memory_context=memory_context,
    )

    # Humanizer: очеловечиваем ответ
    owner_telegram_id = telegram_id
    style_profile = ""
    try:
        style_profile = (await _get_style_profile(owner_telegram_id)) or ""
    except Exception:
        pass

    humanized = _humanize_response(
        text,
        context_hint="analysis",
        style_profile=style_profile,
    )
    score, _ = _analyze_ai_score(humanized)
    if score > 0.3 and len(humanized) > 100:
        try:
            try:
                async with get_session() as session:
                    owner = await _get_or_create_user(session, telegram_id)
                    heavy_provider = await build_provider(
                        session, owner, task_type=TaskType.HUMANIZE
                    )
                    if heavy_provider is None:
                        heavy_provider = ctx["provider"]
            except Exception:
                heavy_provider = ctx["provider"]
            humanized = await _humanize_deep(
                humanized, heavy_provider, user_style=style_profile
            )
        except Exception:
            logger.debug("deep humanize non-critical fail", exc_info=True)

    if user_query:
        truncated_query = user_query[:60] + "…" if len(user_query) > 60 else user_query
        html = f"🤖 <b>Анализ чата «{contact_name}»</b>\n<i>Запрос: {_sanitize_html(truncated_query)}</i>\n\n{humanized}"
    else:
        html = f"🤖 <b>Анализ чата «{contact_name}»</b>\n\n{humanized}"

    return ChatActionResult(
        html=html,
        display_name=contact_name,
        markup=_actions_keyboard(peer_id),
    )


# ── Shared keyboard ─────────────────────────────────────────────────────


def _actions_keyboard(peer_id: int) -> InlineKeyboardMarkup:
    """Кнопки дальнейших действий с чатом."""
    kb = InlineKeyboardBuilder()
    kb.row(
        InlineKeyboardButton(
            text="📝 Саммари", callback_data=f"chat:summary:{peer_id}"
        ),
        InlineKeyboardButton(text="✅ Задачи", callback_data=f"chat:tasks:{peer_id}"),
    )
    kb.row(
        InlineKeyboardButton(text="💬 Черновик", callback_data=f"chat:draft:{peer_id}"),
        InlineKeyboardButton(
            text="⏪ Catchup", callback_data=f"chat:catchup:{peer_id}"
        ),
    )
    kb.row(
        InlineKeyboardButton(text="👁 Следить", callback_data=f"chat:watch:{peer_id}"),
        InlineKeyboardButton(
            text="👁 Не следить", callback_data=f"chat:unwatch:{peer_id}"
        ),
    )
    return kb.as_markup()
