"""Детектор подозрительных объявлений на Авито.

Эвристики для выявления мошеннических / ненадёжных объявлений.
Возвращает уровень риска и список причин.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  Константы
# ═══════════════════════════════════════════════════════════════════════════

# Пороги для ценовых проверок
_PRICE_VERY_LOW_RATIO = 0.60  # < 60% от средней → HIGH
_PRICE_LOW_RATIO = 0.80  # < 80% от средней → MEDIUM (с доп. флагами)

# Подозрительные ключевые слова в описании
_SCAM_KEYWORDS = (
    "алиэкспресс",
    "али экспресс",
    "алиэкcпресс",  # опечатка
    "али",
    "1688",
    "таobao",
    "taobao",
    "алибаба",
    "alibaba",
    "banggood",
    "wish.com",
)

# Порог минимальной длины описания
_SHORT_DESCRIPTION_LEN = 20


# ═══════════════════════════════════════════════════════════════════════════
#  Результат проверки
# ═══════════════════════════════════════════════════════════════════════════


class ScamCheckResult:
    """Результат проверки на мошенничество."""

    __slots__ = ("risk", "reasons", "is_suspicious")

    def __init__(self) -> None:
        self.risk: str = "low"  # low / medium / high
        self.reasons: list[str] = []
        self.is_suspicious: bool = False

    def escalate(self, level: str, reason: str) -> None:
        """Повышает уровень риска (idempotent escalation)."""
        self.reasons.append(f"[{level.upper()}] {reason}")
        priority = {"high": 3, "medium": 2, "low": 1}
        if priority.get(level, 0) > priority.get(self.risk, 0):
            self.risk = level
        if level in ("medium", "high"):
            self.is_suspicious = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_suspicious": self.is_suspicious,
            "risk": self.risk,
            "reasons": self.reasons,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Проверки
# ═══════════════════════════════════════════════════════════════════════════


def _check_price(
    result: ScamCheckResult,
    listing: dict[str, Any],
    avg_price: float | None,
) -> None:
    """Проверка на подозрительно низкую цену."""
    price = listing.get("price")
    if price is None or avg_price is None or avg_price <= 0:
        return

    ratio = price / avg_price

    if ratio < _PRICE_VERY_LOW_RATIO:
        pct = round((1 - ratio) * 100)
        result.escalate(
            "high",
            f"Цена на {pct}% ниже средней ({price} ₽ vs ~{round(avg_price)} ₽)",
        )
    elif ratio < _PRICE_LOW_RATIO:
        pct = round((1 - ratio) * 100)
        result.escalate(
            "medium",
            f"Цена на {pct}% ниже средней ({price} ₽ vs ~{round(avg_price)} ₽)",
        )


def _check_seller(result: ScamCheckResult, listing: dict[str, Any]) -> None:
    """Проверка продавца."""
    reviews = listing.get("seller_reviews")
    if reviews is not None and reviews == 0:
        # Само по себе — LOW, но повышается при других флагах
        result.escalate("low", "Продавец без отзывов (0 отзывов)")


def _check_image(result: ScamCheckResult, listing: dict[str, Any]) -> None:
    """Проверка наличия фото."""
    image_url = listing.get("image_url")
    if not image_url:
        result.escalate("medium", "Нет фотографий")


def _check_description(result: ScamCheckResult, listing: dict[str, Any]) -> None:
    """Проверка описания на мошеннические паттерны."""
    description = (listing.get("description") or "").lower()

    # Подозрительные ключевые слова
    for kw in _SCAM_KEYWORDS:
        if kw in description:
            result.escalate("high", f"Подозрительное ключевое слово: «{kw}»")
            break  # одного достаточно

    # Очень короткое описание
    if 0 < len(listing.get("description") or "") < _SHORT_DESCRIPTION_LEN:
        result.escalate("low", "Очень короткое описание (< 20 символов)")


def _check_combo_flags(result: ScamCheckResult, listing: dict[str, Any]) -> None:
    """Повышение риска при комбинации флагов."""
    reviews = listing.get("seller_reviews")
    price = listing.get("price")
    # 0 отзывов + цена < 80% от средней → MEDIUM
    if reviews is not None and reviews == 0 and price is not None:
        # Этот флаг уже мог быть повышен в _check_price,
        # но если нет — повышаем до medium
        if result.risk == "low":
            result.escalate("medium", "Комбинация: новый продавец + низкая цена")


# ═══════════════════════════════════════════════════════════════════════════
#  Публичный API
# ═══════════════════════════════════════════════════════════════════════════


def check_scam(
    listing: dict[str, Any],
    *,
    avg_price: float | None = None,
) -> dict[str, Any]:
    """Проверяет объявление на мошенничество.

    Args:
        listing: Данные объявления (из parser.parse_listings).
        avg_price: Средняя цена по рынку (для ценовых проверок).

    Returns:
        dict: {
            "is_suspicious": bool,
            "risk": "low" | "medium" | "high",
            "reasons": list[str],
        }
    """
    try:
        result = ScamCheckResult()

        _check_price(result, listing, avg_price)
        _check_seller(result, listing)
        _check_image(result, listing)
        _check_description(result, listing)
        _check_combo_flags(result, listing)

        return result.to_dict()

    except Exception:
        logger.exception("check_scam: ошибка проверки")
        return {
            "is_suspicious": False,
            "risk": "low",
            "reasons": ["[ERROR] Не удалось выполнить проверку"],
        }
