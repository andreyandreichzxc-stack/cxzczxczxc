"""Фоновый мониторинг Авито: периодическая проверка сохранённых поисков."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from sqlalchemy import select

from src.config import settings
from src.core.avito.alerts import check_price_alert
from src.core.avito.service import ScanResult, scan_avito, SearchParams
from src.core.scheduling.notification_queue import notification_queue
from src.db.models import Notification
from src.db.models._avito import AvitoListing, AvitoWatch
from src.db.repo import get_or_create_user
from src.db.session import get_session

logger = logging.getLogger(__name__)

# Интервал проверки берётся из конфига
# AVITO_TICK_SECONDS = settings.avito_check_sec  # используется в task loop


async def _check_watches(owner_telegram_id: int) -> None:
    """Один тик: проверить все активные watch'и."""
    # 1. Собрать все активные watch'и
    async with get_session() as session:
        owner = await get_or_create_user(session, owner_telegram_id)
        result = await session.execute(
            select(AvitoWatch)
            .where(AvitoWatch.user_id == owner.id, AvitoWatch.is_active.is_(True))
            .join(AvitoListing)
        )
        watches = list(result.scalars().all())

    if not watches:
        return

    # 2. Группируем по search_query (один запрос — несколько watch'ей)
    queries: dict[str, list[AvitoWatch]] = {}
    for w in watches:
        # Получаем listing через отдельную сессию
        async with get_session() as session:
            listing = await session.get(AvitoListing, w.listing_id)
            if listing:
                q = listing.search_query
                queries.setdefault(q, []).append(w)

    # 3. Для каждого уникального запроса — сканируем
    for query, watch_group in queries.items():
        try:
            await _process_query(owner_telegram_id, query, watch_group)
        except Exception:
            logger.exception("avito checker: ошибка для запроса %r", query)

        # Пауза между запросами чтобы не нагружать Авито
        await asyncio.sleep(5)


async def _process_query(
    owner_telegram_id: int,
    query: str,
    watches: list[AvitoWatch],
) -> None:
    """Обработать один поисковый запрос."""
    # Собираем существующие объявления из БД
    async with get_session() as session:
        owner = await get_or_create_user(session, owner_telegram_id)
        result = await session.execute(
            select(AvitoListing).where(
                AvitoListing.user_id == owner.id,
                AvitoListing.search_query == query,
            )
        )
        existing_listings = {
            item.avito_id: _listing_to_dict(item) for item in result.scalars().all()
        }

    # Сканируем
    params = SearchParams(city=settings.avito_default_city, category="", query=query)
    scan_result: ScanResult = await scan_avito(params, existing=existing_listings)

    if scan_result.error:
        logger.warning("avito checker: scan error for %r: %s", query, scan_result.error)
        return

    if not scan_result.new_listings and not scan_result.price_changes:
        return

    # Сохраняем новые/обновлённые в БД
    await _save_listings(owner_telegram_id, query, scan_result)

    # Проверяем алерты
    alerts = []
    for watch in watches:
        async with get_session() as session:
            listing = await session.get(AvitoListing, watch.listing_id)
            if listing:
                # Ищем предыдущую цену из existing_listings
                prev_price = None
                if existing_listings and listing.avito_id in existing_listings:
                    prev_price = existing_listings[listing.avito_id].get("price")

                # Проверяем алерт по порогу
                if watch.price_threshold is not None:
                    watch_alert = check_price_alert(
                        _listing_to_dict(listing),
                        price_threshold=watch.price_threshold,
                        previous_price=prev_price,
                    )
                    if watch_alert:
                        alerts.append(watch_alert)
                # Проверяем алерт по изменению цены (даже без порога)
                elif prev_price is not None and listing.price is not None:
                    watch_alert = check_price_alert(
                        _listing_to_dict(listing),
                        previous_price=prev_price,
                    )
                    if watch_alert and watch_alert.get("alert_type") == "price_drop":
                        alerts.append(watch_alert)

    # Формируем уведомление
    parts = []

    if scan_result.new_listings:
        top_new = sorted(
            scan_result.new_listings,
            key=lambda x: (
                x.get("deal_score", {}).get("score", 0)
                if isinstance(x.get("deal_score"), dict)
                else x.get("deal_score", 0)
            ),
            reverse=True,
        )[:3]
        lines = []
        for item in top_new:
            score = item.get("deal_score", {})
            grade = score.get("grade", "?") if isinstance(score, dict) else "?"
            price = item.get("price", "?")
            title = item.get("title", "?")[:40]
            lines.append(f"  {grade} {title} — {price}₽")
        parts.append(
            f"🆕 <b>Новых: {len(scan_result.new_listings)}</b>\n" + "\n".join(lines)
        )

    if scan_result.price_changes:
        top_changes = sorted(
            scan_result.price_changes,
            key=lambda x: abs(
                (x.get("price", 0) or 0) - (x.get("previous_price", 0) or 0)
            ),
            reverse=True,
        )[:3]
        lines = []
        for item in top_changes:
            old = item.get("previous_price", 0)
            new = item.get("price", 0)
            delta = new - old
            sign = "📉" if delta < 0 else "📈"
            title = item.get("title", "?")[:40]
            lines.append(f"  {sign} {title}: {old}₽ → {new}₽")
        parts.append(
            f"💰 <b>Изменений цен: {len(scan_result.price_changes)}</b>\n"
            + "\n".join(lines)
        )

    if alerts:
        alert_lines = []
        for a in alerts:
            alert_lines.append(f"  🔔 {a['message']}")
        parts.append("\n".join(alert_lines))

    if parts:
        text = f"🏠 <b>Avito: {query}</b>\n\n" + "\n\n".join(parts)
        await notification_queue.enqueue(
            topic="avito",
            text=text,
            priority=Notification.PRIORITY_MEDIUM,
            category=query,
        )


async def _save_listings(
    owner_telegram_id: int,
    query: str,
    scan_result: ScanResult,
) -> None:
    """Сохранить новые/обновлённые объявления в БД."""
    async with get_session() as session:
        owner = await get_or_create_user(session, owner_telegram_id)

        for listing_data in scan_result.new_listings:
            avito_id = listing_data.get("avito_id", "")
            if not avito_id:
                continue

            # Проверяем — может уже есть
            existing = await session.execute(
                select(AvitoListing).where(
                    AvitoListing.user_id == owner.id,
                    AvitoListing.avito_id == avito_id,
                )
            )
            existing_listing = existing.scalar_one_or_none()

            if existing_listing:
                # Обновляем last_seen_at
                existing_listing.last_seen_at = datetime.now(timezone.utc)
                continue

            # Создаём новый
            deal_score_data = listing_data.get("deal_score", {})
            scam_data = listing_data.get("scam_check", {})

            new_listing = AvitoListing(
                user_id=owner.id,
                avito_id=avito_id,
                search_query=query,
                title=listing_data.get("title", "")[:512],
                price=listing_data.get("price"),
                url=listing_data.get("url", ""),
                image_url=listing_data.get("image_url"),
                city=listing_data.get("city"),
                condition=listing_data.get("condition"),
                delivery=bool(listing_data.get("delivery")),
                seller_name=listing_data.get("seller_name"),
                seller_rating=listing_data.get("seller_rating"),
                seller_reviews=listing_data.get("seller_reviews"),
                description=listing_data.get("description"),
                deal_score=deal_score_data.get("score")
                if isinstance(deal_score_data, dict)
                else deal_score_data,
                is_suspicious=scam_data.get("is_suspicious", False)
                if isinstance(scam_data, dict)
                else False,
                scam_reasons=", ".join(scam_data.get("reasons", []))
                if isinstance(scam_data, dict)
                else None,
            )
            session.add(new_listing)

        # Обновляем цены для изменившихся
        for listing_data in scan_result.price_changes:
            avito_id = listing_data.get("avito_id", "")
            if not avito_id:
                continue

            result = await session.execute(
                select(AvitoListing).where(
                    AvitoListing.user_id == owner.id,
                    AvitoListing.avito_id == avito_id,
                )
            )
            existing = result.scalar_one_or_none()
            if existing:
                old_price = existing.price
                new_price = listing_data.get("price")
                if old_price != new_price and new_price is not None:
                    existing.price = new_price
                    existing.price_changed_at = datetime.now(timezone.utc)
                    existing.last_seen_at = datetime.now(timezone.utc)

                    # Записываем историю цен
                    from src.db.models._avito import AvitoPriceHistory

                    history = AvitoPriceHistory(
                        listing_id=existing.id,
                        price=new_price,
                    )
                    session.add(history)

        await session.commit()


def _listing_to_dict(listing: AvitoListing) -> dict:
    """Конвертация ORM → dict для service."""
    # Формируем структурированный deal_score
    score = listing.deal_score
    if score is not None:
        if score >= 85:
            grade = "A"
        elif score >= 70:
            grade = "B"
        elif score >= 55:
            grade = "C"
        elif score >= 40:
            grade = "D"
        else:
            grade = "F"
        deal_score_data = {"score": score, "grade": grade}
    else:
        deal_score_data = {}

    # Формируем структурированный scam_check
    scam_reasons = []
    if listing.scam_reasons:
        scam_reasons = [r.strip() for r in listing.scam_reasons.split(",") if r.strip()]
    scam_data = {
        "is_suspicious": listing.is_suspicious,
        "risk": "high" if listing.is_suspicious else "low",
        "reasons": scam_reasons,
    }

    return {
        "avito_id": listing.avito_id,
        "title": listing.title,
        "price": listing.price,
        "url": listing.url,
        "image_url": listing.image_url,
        "city": listing.city,
        "condition": listing.condition,
        "delivery": listing.delivery,
        "seller_name": listing.seller_name,
        "seller_rating": listing.seller_rating,
        "seller_reviews": listing.seller_reviews,
        "description": listing.description,
        "deal_score": deal_score_data,
        "scam_check": scam_data,
    }


from src.core.infra.task_manager import task_manager


@task_manager.task("avito-checker")
async def avito_checker_loop() -> None:
    """Фоновый цикл проверки Авито."""
    while True:
        try:
            await _check_watches(settings.owner_telegram_id)
        except Exception:
            logger.exception("avito checker tick failed")
        await asyncio.sleep(settings.avito_check_sec)
