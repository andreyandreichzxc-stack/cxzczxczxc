"""Команда /explain — показать почему бот так думает о контакте.

Строит дерево фактов, дистилляционные выводы и историю конфликтов.
"""

import logging

from aiogram import Router
from aiogram.filters import Command, CommandObject
from aiogram.types import Message

from src.bot.filters import OwnerOnly
from src.core.contacts.contact_resolver import resolve
from src.core.actions.conflict_resolver import find_conflicts
from src.db.repo import get_or_create_user, list_memories, get_memory_graph
from src.db.session import get_session
from src.userbot.manager import UserbotManager

logger = logging.getLogger(__name__)
router = Router(name="explain_cmd")
router.message.filter(OwnerOnly())


def _build_explain_output(
    contact_label: str,
    active: list,
    distill_facts: list,
    conflicts: list,
    graph_nodes: list,
    recall_result=None,
) -> str:
    """Форматирует объяснение: дистилляция → факты → конфликты → граф."""
    lines: list[str] = []

    # Если есть distillation — показываем его первым (главный вывод)
    for df in distill_facts:
        fact_text = df.fact
        if fact_text.startswith("💡 "):
            fact_text = fact_text[2:]
        distill_date = ""
        if df.created_at:
            distill_date = df.created_at.strftime("%Y-%m-%d")
        # Считаем на основе скольких фактов сделан вывод
        base_count = len(active) - len(distill_facts)
        lines.append("💡 <b>Долгосрочный вывод:</b>")
        lines.append(f"«{fact_text}»")
        lines.append(
            f"↳ На основе: {max(base_count, 0)} фактов (дистилляция от {distill_date})"
        )
        lines.append("")

    # Заголовок
    if contact_label:
        lines.insert(0, f"🧠 Почему я так думаю о {contact_label}:\n")
    else:
        lines.insert(0, "🧠 Почему я так думаю:\n")

    # Факты — сгруппированные с эмодзи
    fact_lines = _format_facts(active, distill_facts)
    if fact_lines:
        lines.append("📊 <b>Факты:</b>")
        lines.extend(fact_lines)
        lines.append("")

    # Конфликты (разрешённые)
    if conflicts:
        lines.append("⚠️ <b>Конфликты (разрешены):</b>")
        for c in conflicts:
            lines.append(f"«{c['fact_negative'][:60]}» → «{c['fact_positive'][:60]}»")
        lines.append("")

    # Explainability — recall
    if recall_result and recall_result.facts:
        lines.append("")
        lines.append("<b>🔍 Логика отбора фактов:</b>")
        for rf in recall_result.facts[:5]:
            lines.append(f"• {rf.reason} → «{rf.fact[:80]}»")
        if recall_result.meta:
            lines.append(
                f"<i>Всего активно: {recall_result.meta.get('total_active', '?')} → показано: {len(recall_result.facts)}</i>"
            )

    return "\n".join(lines)


def _format_facts(active: list, distill_facts: list) -> list[str]:
    """Форматирует факты с эмодзи sentiment и датами."""
    # Собираем ID дистилляций, чтобы исключить их из основного списка
    distill_ids = {m.id for m in distill_facts}

    sent_emoji = {
        "positive": "🟢 позитив",
        "negative": "🔴 негатив",
        "neutral": "⚪ нейтрально",
        "contradictory": "🟡 противоречие",
    }
    rel_icon = {
        "cause": "🎯 причина",
        "effect": "⚡ следствие",
        "contradicts": "⚠️ противоречит",
        "supports": "✅ подтверждает",
        "continues": "➡️ продолжение",
        "example_of": "📌 пример",
        "resolves": "🔄 разрешает",
    }

    result: list[str] = []
    for m in active:
        if m.id in distill_ids:
            continue
        date_str = m.created_at.strftime("%d.%m") if m.created_at else "?"
        s_emoji = sent_emoji.get(m.sentiment or "", "⚪")
        rel = rel_icon.get(m.relation_type or "", "")
        if rel:
            line = f"{s_emoji}: «{m.fact}» [{date_str}] {rel}"
        else:
            line = f"{s_emoji}: «{m.fact}» [{date_str}]"
        result.append(line)
    return result


async def _resolve_contact(owner_id: int, contact_name: str, userbot_manager=None):
    """Resolve contact by name, returns (contact_id, label)."""
    async with get_session() as session:
        owner = await get_or_create_user(session, owner_id)

    client = userbot_manager.get_client(owner_id) if userbot_manager else None
    if client is not None:
        candidates = await resolve(client, owner, contact_name)
        if candidates:
            return candidates[0].peer_id, candidates[0].label()
    return None, ""


async def build_explain_text(
    owner_id: int,
    contact_id: int | None = None,
    contact_label: str = "",
) -> str:
    """Строит объяснение: загружает факты, дистилляцию, конфликты и граф."""
    async with get_session() as session:
        owner = await get_or_create_user(session, owner_id)
        memories = await list_memories(session, owner, contact_id=contact_id)

    active = [m for m in memories if m.is_active]
    if not active:
        return "📭 Память пуста. Пока нечего объяснять."

    # 1. Дистилляционные факты
    distill_facts = [m for m in active if m.source == "distillation"]

    # 2. Конфликты для данного контакта
    all_conflicts = await find_conflicts(owner_id)
    if contact_id:
        contact_conflicts = [
            c for c in all_conflicts if c.get("contact_id") == contact_id
        ]
    else:
        contact_conflicts = all_conflicts

    # 3. Граф фактов (первые 5 фактов для построения связей)
    graph_nodes: list = []
    for m in active[:5]:
        if m.id:
            async with get_session() as s:
                owner = await get_or_create_user(s, owner_id)
                nodes = await get_memory_graph(s, owner, m.id, max_depth=2, max_nodes=8)
                graph_nodes.extend(nodes)

    # 4. Recall причины
    from src.core.memory.memory_recall import recall

    try:
        explain_result = await recall(
            owner_id,
            contact_id=contact_id,
            limit=8,
            include_self=True,
            include_pinned=True,
            include_tasks=True,
            mode="normal",
        )
    except Exception:
        logger.exception("explain recall failed")
        explain_result = None

    return _build_explain_output(
        contact_label=contact_label,
        active=active,
        distill_facts=distill_facts,
        conflicts=contact_conflicts,
        graph_nodes=graph_nodes,
        recall_result=explain_result,
    )


@router.message(Command("explain"))
async def cmd_explain(
    message: Message,
    command: CommandObject | None = None,
    userbot_manager: UserbotManager | None = None,
) -> None:
    """Команда /explain — показать почему бот так думает."""
    args = (command.args or "").strip() if command else ""
    args = args or (message.text or "").replace("/explain", "").strip()

    contact_id = None
    contact_label = ""
    if args:
        contact_id, contact_label = await _resolve_contact(
            message.from_user.id, args, userbot_manager
        )
        if contact_id is None:
            await message.answer(
                f"🙅 Не нашёл контакт «{args}». Проверь имя или /sync."
            )
            return

    text = await build_explain_text(
        message.from_user.id,
        contact_id=contact_id,
        contact_label=contact_label,
    )
    await message.answer(text)
