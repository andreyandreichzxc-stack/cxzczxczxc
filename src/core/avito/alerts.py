"""Система ценовых оповещений для Авито.

Проверяет, упала ли цена ниже порога пользователя,
и генерирует текст уведомления.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Форматирование
# ═══════════════════════════════════════════════════════════════════════════


def _fmt_price(price: int | None) -> str:
    """Форматирует цену с разделителем тысяч."""
    if price is None:
        return "не указана"
    return f"{price:,.0f}".replace(",", " ") + " ₽"


def _fmt_delta(old: int, new: int) -> str:
    """Форматирует изменение цены: '1 000 ₽ → 800 ₽ (−20%)'."""
    delta = new - old
    pct = round(delta / old * 100) if old else 0
    arrow = "↓" if delta < 0 else "↑" if delta > 0 else "="
    return f"{_fmt_price(old)} → {_fmt_price(new)} ({arrow}{abs(pct)}%)"


# ═══════════════════════════════════════════════════════════════════════════
#  Публичный API
# ═══════════════════════════════════════════════════════════════════════════


def check_price_alert(
    listing: dict[str, Any],
    *,
    price_threshold: int | None = None,
    previous_price: int | None = None,
) -> dict[str, Any] | None:
    """Проверяет, нужно ли отправить оповещение.

    Args:
        listing: Данные объявления.
        price_threshold: Максимальная цена, при которой срабатывает алерт.
        previous_price: Предыдущая цена (для отслеживания динамики).

    Returns:
        dict с полями alert_type, message, title, url, price, delta_text
        или None, если алерт не сработал.
    """
    try:
        price = listing.get("price")
        if price is None:
            return None

        title = listing.get("title", "Без названия")
        url = listing.get("url", "")
        score_info = listing.get("deal_score", {})
        scam_info = listing.get("scam_check", {})

        # ── Новая цена ниже порога ─────────────────────────────────────
        if price_threshold is not None and price <= price_threshold:
            msg_parts = [
                "🔔 <b>Цена ниже порога!</b>",
                "",
                f"📌 {title}",
                f"💰 {_fmt_price(price)} (порог: {_fmt_price(price_threshold)})",
            ]

            if previous_price is not None and previous_price != price:
                msg_parts.append(f"📊 {_fmt_delta(previous_price, price)}")

            if url:
                msg_parts.append(f"🔗 {url}")

            if score_info:
                msg_parts.append(
                    f"⭐ Оценка: {score_info.get('score', '?')}/100 ({score_info.get('grade', '?')})"
                )

            if scam_info and scam_info.get("is_suspicious"):
                msg_parts.append(f"⚠️ Риск: {scam_info.get('risk', '?')}")

            return {
                "alert_type": "price_below_threshold",
                "message": "\n".join(msg_parts),
                "title": title,
                "url": url,
                "price": price,
                "threshold": price_threshold,
                "previous_price": previous_price,
                "delta_text": _fmt_delta(previous_price, price)
                if previous_price
                else None,
            }

        # ── Цена упала по сравнению с прошлой ──────────────────────────
        if previous_price is not None and price < previous_price:
            drop_pct = round((previous_price - price) / previous_price * 100)
            # Оповещаем при падении >= 10%
            if drop_pct >= 10:
                msg_parts = [
                    "📉 <b>Снижение цены!</b>",
                    "",
                    f"📌 {title}",
                    f"📊 {_fmt_delta(previous_price, price)}",
                ]
                if url:
                    msg_parts.append(f"🔗 {url}")

                return {
                    "alert_type": "price_drop",
                    "message": "\n".join(msg_parts),
                    "title": title,
                    "url": url,
                    "price": price,
                    "previous_price": previous_price,
                    "delta_text": _fmt_delta(previous_price, price),
                }

        return None

    except Exception:
        logger.exception("check_price_alert: ошибка")
        return None


def format_listing_alert(listing: dict[str, Any]) -> str:
    """Форматирует короткое текстовое описание объявления для уведомления.

    Используется для новых объявлений, не связанных с ценовыми порогами.
    """
    try:
        title = listing.get("title", "Без названия")
        price = _fmt_price(listing.get("price"))
        url = listing.get("url", "")
        city = listing.get("city", "")
        score = listing.get("deal_score", {}).get("score", "?")
        grade = listing.get("deal_score", {}).get("grade", "?")
        scam = listing.get("scam_check", {})

        parts = [f"📌 {title}", f"💰 {price}"]

        if city:
            parts.append(f"📍 {city}")
        parts.append(f"⭐ {score}/100 ({grade})")

        if scam.get("is_suspicious"):
            reasons = "; ".join(scam.get("reasons", [])[:2])
            parts.append(f"⚠️ {scam.get('risk', '?')}: {reasons}")

        if url:
            parts.append(f"🔗 {url}")

        return "\n".join(parts)

    except Exception:
        logger.exception("format_listing_alert: ошибка")
        return "⚠️ Ошибка форматирования объявления"
