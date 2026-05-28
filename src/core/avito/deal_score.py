"""Алгоритм оценки выгодности объявления Авито (0-100).

Критерии:
- Цена (35%) — чем ниже, тем лучше
- Состояние (25%) — новое лучше б/у
- Продавец (20%) — рейтинг + количество отзывов
- Доставка (10%) — наличие доставки
- Описание (10%) — наличие + ключевые слова
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  Константы весов
# ═══════════════════════════════════════════════════════════════════════════

_WEIGHT_PRICE = 0.35
_WEIGHT_CONDITION = 0.25
_WEIGHT_SELLER = 0.20
_WEIGHT_DELIVERY = 0.10
_WEIGHT_DESCRIPTION = 0.10

# Оценка состояния (normalized 0-100)
_CONDITION_SCORES: dict[str, int] = {
    "новый": 100,
    "новое": 100,
    "отличное": 85,
    "отлично": 85,
    "хорошее": 70,
    "хорошо": 70,
    "удовлетворительное": 50,
    "удовлетворительно": 50,
    "б/у": 50,
}
_DEFAULT_CONDITION = 60

# Бонусные слова в описании (каждое +5, макс +15)
_BONUS_KEYWORDS = ("чек", "гарантия", "комплект", "оригинал", "запечатан")


# ═══════════════════════════════════════════════════════════════════════════
#  Вспомогательные функции
# ═══════════════════════════════════════════════════════════════════════════


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Ограничивает значение диапазоном [lo, hi]."""
    return max(lo, min(hi, value))


def _score_price(
    price: int | None, avg_price: float | None, min_price: float | None
) -> float:
    """Оценка цены: чем ниже, тем лучше. 100 = минимальная цена, 0 = средняя и выше."""
    if price is None or avg_price is None or min_price is None:
        return 60.0  # нейтральная оценка при отсутствии данных
    if avg_price <= min_price:
        return 100.0  # все цены одинаковы
    # Линейная интерполяция: min_price → 100, avg_price → 0
    score = 100.0 - ((price - min_price) / (avg_price - min_price)) * 100.0
    return _clamp(score)


def _score_condition(condition: str | None) -> float:
    """Оценка состояния по справочнику."""
    if not condition:
        return float(_DEFAULT_CONDITION)
    normalized = condition.strip().lower()
    return float(_CONDITION_SCORES.get(normalized, _DEFAULT_CONDITION))


def _score_seller(rating: float | None, reviews: int | None) -> float:
    """Оценка продавца: рейтинг (70%) + отзывы (30%)."""
    # Рейтинговая составляющая
    if rating is not None:
        rating_part = (rating / 5.0) * 70.0
    else:
        rating_part = 35.0  # нейтрально

    # Отзывная составляющая
    if reviews is not None:
        review_part = min(reviews / 100.0, 1.0) * 30.0
    else:
        review_part = 15.0  # нейтрально

    return _clamp(rating_part + review_part)


def _score_delivery(has_delivery: bool | None) -> float:
    """Оценка доставки."""
    if has_delivery:
        return 100.0
    return 50.0


def _score_description(description: str | None) -> float:
    """Оценка описания: наличие + бонусные ключевые слова."""
    if not description or len(description.strip()) < 5:
        return 40.0

    base = 80.0
    desc_lower = description.lower()

    # Бонус за ключевые слова (+5 за каждое, макс +15)
    bonus = 0.0
    for kw in _BONUS_KEYWORDS:
        if kw in desc_lower:
            bonus += 5.0
            if bonus >= 15.0:
                break

    return _clamp(base + bonus)


# ═══════════════════════════════════════════════════════════════════════════
#  Публичный API
# ═══════════════════════════════════════════════════════════════════════════


def calculate_deal_score(
    listing: dict[str, Any],
    *,
    avg_price: float | None = None,
    min_price: float | None = None,
) -> dict[str, Any]:
    """Рассчитывает оценку выгодности объявления (0-100).

    Args:
        listing: Данные объявления (из parser.parse_listings).
        avg_price: Средняя цена по рынку.
        min_price: Минимальная цена по рынку.

    Returns:
        dict: {
            "score": int (0-100),
            "breakdown": {
                "price": float,
                "condition": float,
                "seller": float,
                "delivery": float,
                "description": float,
            },
            "grade": str ("A" / "B" / "C" / "D" / "F"),
        }
    """
    try:
        p_price = _score_price(listing.get("price"), avg_price, min_price)
        p_condition = _score_condition(listing.get("condition"))
        p_seller = _score_seller(
            listing.get("seller_rating"),
            listing.get("seller_reviews"),
        )
        p_delivery = _score_delivery(listing.get("delivery"))
        p_description = _score_description(listing.get("description"))

        raw_score = (
            p_price * _WEIGHT_PRICE
            + p_condition * _WEIGHT_CONDITION
            + p_seller * _WEIGHT_SELLER
            + p_delivery * _WEIGHT_DELIVERY
            + p_description * _WEIGHT_DESCRIPTION
        )
        score = int(round(_clamp(raw_score)))

        # Буквенная оценка
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

        return {
            "score": score,
            "breakdown": {
                "price": round(p_price, 1),
                "condition": round(p_condition, 1),
                "seller": round(p_seller, 1),
                "delivery": round(p_delivery, 1),
                "description": round(p_description, 1),
            },
            "grade": grade,
        }

    except Exception:
        logger.exception("calculate_deal_score: ошибка расчёта")
        return {
            "score": 0,
            "breakdown": {
                "price": 0.0,
                "condition": 0.0,
                "seller": 0.0,
                "delivery": 0.0,
                "description": 0.0,
            },
            "grade": "F",
        }
