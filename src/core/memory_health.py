"""Memory Health Score — единый балл здоровья памяти 0-100."""

import logging
from datetime import datetime, timedelta, timezone
from src.db.session import get_session
from src.db.repo import get_or_create_user, list_memories, list_contacts

logger = logging.getLogger(__name__)


async def calculate_health_score(owner_id: int) -> dict:
    """
    Вычисляет балл здоровья памяти и компоненты (кэшируется на 5 минут).
    Возвращает {score, confidence_score, coverage_score, freshness_score, distillation_score, diagnostics}
    """
    from src.core.stats_cache import get_cached, set_cache

    cache_key = f"health:{owner_id}"
    cached = await get_cached(cache_key)
    if cached is not None:
        return cached

    async with get_session() as session:
        owner = await get_or_create_user(session, owner_id)
        memories = await list_memories(session, owner)
        contacts = await list_contacts(
            session, owner, kinds=("user",), include_bots=False
        )

        active = [m for m in memories if m.is_active]
        all_contacts = len(contacts)
        now = datetime.now(timezone.utc)

        diagnostics = []

        # 1. Confidence Score (средний confidence активных фактов) × 100
        conf_values = [m.confidence or 0.5 for m in active]
        avg_conf = sum(conf_values) / len(conf_values) if conf_values else 0
        confidence_score = avg_conf * 100
        if confidence_score < 40:
            diagnostics.append(f"🔴 Средний confidence: {avg_conf:.2f} — низкий")
        elif confidence_score < 70:
            diagnostics.append(f"🟡 Средний confidence: {avg_conf:.2f} — средний")
        else:
            diagnostics.append(f"🟢 Средний confidence: {avg_conf:.2f} — высокий")

        # 2. Coverage Score (доля контактов с фактами) × 100
        contacts_with_facts = set()
        for m in active:
            if m.contact_id:
                contacts_with_facts.add(m.contact_id)
        coverage = len(contacts_with_facts) / max(all_contacts, 1)
        coverage_score = coverage * 100
        if coverage_score < 20:
            diagnostics.append(
                f"🔴 Покрытие: {len(contacts_with_facts)}/{all_contacts} контактов ({coverage * 100:.0f}%) — мало"
            )
        elif coverage_score < 50:
            diagnostics.append(
                f"🟡 Покрытие: {len(contacts_with_facts)}/{all_contacts} контактов ({coverage * 100:.0f}%) — средне"
            )
        else:
            diagnostics.append(
                f"🟢 Покрытие: {len(contacts_with_facts)}/{all_contacts} контактов ({coverage * 100:.0f}%) — хорошо"
            )

        # 3. Freshness Score (доля фактов младше 30 дней) × 100
        fresh_cutoff = now - timedelta(days=30)
        fresh_facts = [
            m for m in active if m.created_at and m.created_at >= fresh_cutoff
        ]
        freshness = len(fresh_facts) / max(len(active), 1)
        freshness_score = freshness * 100
        if freshness_score < 30:
            diagnostics.append(
                f"🔴 Свежесть: {len(fresh_facts)}/{len(active)} фактов младше 30 дней ({freshness * 100:.0f}%) — память застаивается"
            )
        elif freshness_score < 60:
            diagnostics.append(
                f"🟡 Свежесть: {len(fresh_facts)}/{len(active)} фактов ({freshness * 100:.0f}%) — средне"
            )
        else:
            diagnostics.append(
                f"🟢 Свежесть: {len(fresh_facts)}/{len(active)} фактов ({freshness * 100:.0f}%) — отлично"
            )

        # 4. Distillation/Structure Score (доля distillation + tier 3 фактов) × 100
        structured = [
            m for m in active if m.source == "distillation" or m.memory_tier == 3
        ]
        structure_ratio = len(structured) / max(len(active), 1)
        structure_score = min(
            structure_ratio * 200, 100
        )  # ×2 потому что distillation мало, но каждая ценна
        if structure_score < 10:
            diagnostics.append(
                f"🟡 Структурированность: {len(structured)} distillation-фактов — можно улучшить"
            )

        # 5. Tag Coverage (доля тегированных фактов)
        tagged = [m for m in active if m.tags]
        tag_ratio = len(tagged) / max(len(active), 1)
        tag_score = min(tag_ratio * 120, 100)
        if tag_score < 40:
            diagnostics.append(
                f"🟡 Теги: {len(tagged)}/{len(active)} фактов протегировано ({tag_ratio * 100:.0f}%)"
            )

        # Композитный score: среднее взвешенное
        weights = {
            "confidence": 0.35,
            "coverage": 0.25,
            "freshness": 0.25,
            "structure": 0.10,
            "tags": 0.05,
        }
        composite = (
            weights["confidence"] * confidence_score
            + weights["coverage"] * coverage_score
            + weights["freshness"] * freshness_score
            + weights["structure"] * structure_score
            + weights["tags"] * tag_score
        )

        # Определяем уровень
        if composite >= 70:
            level = "green"
            emoji = "🟢🧠"
            label = "Отлично"
        elif composite >= 40:
            level = "yellow"
            emoji = "🟡"
            label = "Средне"
        else:
            level = "red"
            emoji = "🔴"
            label = "Плохо"

        result = {
            "score": round(composite, 1),
            "level": level,
            "emoji": emoji,
            "label": label,
            "confidence_score": round(confidence_score, 1),
            "coverage_score": round(coverage_score, 1),
            "freshness_score": round(freshness_score, 1),
            "structure_score": round(structure_score, 1),
            "tag_score": round(tag_score, 1),
            "total_facts": len(active),
            "total_contacts": all_contacts,
            "contacts_with_facts": len(contacts_with_facts),
            "diagnostics": diagnostics,
        }
        await set_cache(cache_key, result)
        return result


def format_health(health: dict) -> str:
    """Форматирует здоровье памяти в HTML."""
    score = health["score"]
    emoji = health["emoji"]
    label = health["label"]

    # Цветной progress bar
    bar_len = 10
    filled = int(score / 100 * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)

    lines = [
        f"<b>{emoji} Здоровье памяти: {score}/100 — {label} {emoji}</b>",
        f"[{bar}]",
        "",
        f"📊 <b>Компоненты:</b>",
        f"  🎯 Confidence: {health['confidence_score']}/100",
        f"  🌐 Покрытие: {health['coverage_score']}/100 ({health['contacts_with_facts']}/{health['total_contacts']} контактов)",
        f"  ⏳ Свежесть: {health['freshness_score']}/100",
        f"  💡 Структура: {health['structure_score']}/100",
        f"  🏷 Теги: {health['tag_score']}/100",
        f"  📝 Всего фактов: {health['total_facts']}",
    ]

    if health["diagnostics"]:
        lines.append("")
        lines.append("<b>🔍 Диагностика:</b>")
        for d in health["diagnostics"]:
            lines.append(f"  {d}")

    return "\n".join(lines)


def format_health_compact(health: dict) -> str:
    """Компактный индикатор для вставки в шапку брифинга."""
    score = health["score"]
    bar = "█" * int(score / 10) + "░" * (10 - int(score / 10))
    return f"🧠 Здоровье памяти: {score}/100 [{bar}]"


async def compute_emotional_trend(owner_id: int) -> str | None:
    """
    Сравнивает sentiment за последние 7 дней vs предыдущие 7 дней.
    Возвращает строку с эмоциональным трендом или None, если данных недостаточно.
    """
    async with get_session() as session:
        owner = await get_or_create_user(session, owner_id)
        memories = await list_memories(session, owner)
    now = datetime.now(timezone.utc)
    recent_cutoff = now - timedelta(days=7)
    prior_cutoff = now - timedelta(days=14)

    recent = [
        m
        for m in memories
        if m.is_active
        and m.sentiment in ("positive", "negative")
        and m.created_at
        and recent_cutoff <= m.created_at <= now
    ]
    prior = [
        m
        for m in memories
        if m.is_active
        and m.sentiment in ("positive", "negative")
        and m.created_at
        and prior_cutoff <= m.created_at < recent_cutoff
    ]

    if not recent or not prior:
        return None

    def positivity_ratio(ms: list) -> float:
        pos = sum(1 for m in ms if m.sentiment == "positive")
        return pos / len(ms) if ms else 0.0

    recent_ratio = positivity_ratio(recent)
    prior_ratio = positivity_ratio(prior)
    diff = recent_ratio - prior_ratio

    if diff > 0.1:
        return "📈 Эмоциональный тренд: отношения улучшаются ✨"
    elif diff < -0.1:
        return "📉 Эмоциональный тренд: растёт напряжение ⚠️"
    else:
        return "➖ Эмоциональный тренд: стабильно"
