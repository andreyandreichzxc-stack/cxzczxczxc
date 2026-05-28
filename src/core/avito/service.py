"""Сервис мониторинга объявлений Авито.

Оркестрирует:
- Построение URL поиска
- Загрузка страницы (stealth-сессия с антидетектом)
- Парсинг объявлений
- Оценка выгодности (deal_score)
- Проверка на мошенничество (anti_scam)
- Сравнение с БД (инкрементальный анализ)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time as _time_module
from typing import Any
from urllib.parse import quote_plus

from src.core.avito.anti_scam import check_scam
from src.core.avito.deal_score import calculate_deal_score
from src.core.avito.parser import parse_listings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Типы
# ═══════════════════════════════════════════════════════════════════════════


class SearchParams:
    """Параметры поиска на Авито."""

    def __init__(
        self,
        city: str,
        category: str,
        query: str,
        *,
        price_min: int | None = None,
        price_max: int | None = None,
    ) -> None:
        self.city = city
        self.category = category
        self.query = query
        self.price_min = price_min
        self.price_max = price_max


class ScanResult:
    """Результат сканирования."""

    def __init__(self) -> None:
        self.listings: list[dict[str, Any]] = []
        self.new_listings: list[dict[str, Any]] = []
        self.price_changes: list[dict[str, Any]] = []
        self.unchanged: list[dict[str, Any]] = []
        self.error: str | None = None
        self.url: str = ""
        self.total_parsed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "listings": self.listings,
            "new_listings": self.new_listings,
            "price_changes": self.price_changes,
            "unchanged": self.unchanged,
            "error": self.error,
            "url": self.url,
            "total_parsed": self.total_parsed,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  HTTP-загрузка (stealth-сессия)
# ═══════════════════════════════════════════════════════════════════════════

_stealth_session: object | None = None
_stealth_lock = asyncio.Lock()


async def _get_stealth_session():
    """Lazy-init the stealth session (warmup once, reuse)."""
    global _stealth_session
    if _stealth_session is None:
        async with _stealth_lock:
            if _stealth_session is None:
                from src.core.avito.stealth.session import AvitoSession

                _stealth_session = AvitoSession()
                await _stealth_session.warmup()  # type: ignore[attr-defined]
    return _stealth_session


async def _fetch_page(url: str) -> str:
    """Загружает HTML-страницу через stealth-сессию (httpx + browser fallback)."""
    session = await _get_stealth_session()
    resp = await session.fetch(url)  # type: ignore[attr-defined]
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: страница не загружена")
    return resp.text


# ═══════════════════════════════════════════════════════════════════════════
#  Построение URL
# ═══════════════════════════════════════════════════════════════════════════


def build_avito_url(params: SearchParams) -> str:
    """Формирует URL поиска Авито.

    Формат: https://www.avito.ru/{city}/{category}?q={query}&pmin={min}&pmax={max}
    """
    city = params.city.strip().lower().replace(" ", "_")
    category = params.category.strip().lower().replace(" ", "_")
    query_encoded = quote_plus(params.query)

    url = f"https://www.avito.ru/{city}/{category}?q={query_encoded}"

    if params.price_min is not None:
        url += f"&pmin={params.price_min}"
    if params.price_max is not None:
        url += f"&pmax={params.price_max}"

    return url


# ═══════════════════════════════════════════════════════════════════════════
#  Рыночная статистика
# ═══════════════════════════════════════════════════════════════════════════


def _calc_market_stats(listings: list[dict[str, Any]]) -> dict[str, float | None]:
    """Рассчитывает рыночную статистику (средняя, минимальная цена)."""
    prices = [item["price"] for item in listings if item.get("price") is not None]
    if not prices:
        return {"avg_price": None, "min_price": None}
    return {
        "avg_price": sum(prices) / len(prices),
        "min_price": float(min(prices)),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Инкрементальный анализ
# ═══════════════════════════════════════════════════════════════════════════


def _compare_with_db(
    parsed: list[dict[str, Any]],
    existing: dict[str, dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Сравнивает спарсенные объявления с существующими в БД.

    Args:
        parsed: Список спарсенных объявлений.
        existing: Словарь {avito_id: listing_data} из БД (или None).

    Returns:
        (new_listings, price_changes, unchanged)
    """
    if existing is None:
        return parsed, [], []

    new_listings: list[dict[str, Any]] = []
    price_changes: list[dict[str, Any]] = []
    unchanged: list[dict[str, Any]] = []

    for listing in parsed:
        avito_id = listing.get("avito_id")
        if not avito_id:
            new_listings.append(listing)
            continue

        old = existing.get(avito_id)
        if old is None:
            new_listings.append(listing)
            continue

        old_price = old.get("price")
        new_price = listing.get("price")

        if old_price is not None and new_price is not None and old_price != new_price:
            listing["previous_price"] = old_price
            price_changes.append(listing)
        else:
            unchanged.append(listing)

    return new_listings, price_changes, unchanged


# ═══════════════════════════════════════════════════════════════════════════
#  Кэш результатов сканирования
# ═══════════════════════════════════════════════════════════════════════════

_SCAN_CACHE: dict[str, tuple[float, ScanResult]] = {}
_SCAN_CACHE_TTL = 300  # 5 минут


def _cache_hash(params: SearchParams) -> str:
    return hashlib.md5(
        f"{params.city}:{params.query}:{params.price_min}:{params.price_max}".encode()
    ).hexdigest()


async def scan_avito_cached(params: SearchParams) -> ScanResult:
    """scan_avito с кэшированием результата на 5 минут."""
    key = _cache_hash(params)
    now = _time_module.time()
    if key in _SCAN_CACHE:
        ts, result = _SCAN_CACHE[key]
        if now - ts < _SCAN_CACHE_TTL:
            return result
    result = await scan_avito(params)
    _SCAN_CACHE[key] = (now, result)
    # Очистка старых записей
    if len(_SCAN_CACHE) > 100:
        for k in list(_SCAN_CACHE.keys()):
            if now - _SCAN_CACHE[k][0] > _SCAN_CACHE_TTL * 2:
                del _SCAN_CACHE[k]
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Публичный API
# ═══════════════════════════════════════════════════════════════════════════


async def scan_avito(
    params: SearchParams,
    *,
    existing: dict[str, dict[str, Any]] | None = None,
) -> ScanResult:
    """Полный цикл сканирования Авито.

    1. Строит URL
    2. Загружает страницу
    3. Парсит объявления
    4. Считает deal_score для каждого
    5. Проверяет на мошенничество
    6. Сравнивает с existing (инкрементальный анализ)

    Args:
        params: Параметры поиска.
        existing: Словарь {avito_id: listing_data} из БД для инкрементального анализа.

    Returns:
        ScanResult с полными данными.
    """
    result = ScanResult()

    # 1. URL
    url = build_avito_url(params)
    result.url = url
    logger.info("scan_avito: загрузка %s", url)

    # 2. Загрузка
    try:
        html = await _fetch_page(url)
    except RuntimeError as exc:
        result.error = str(exc)
        logger.error("scan_avito: ошибка загрузки — %s", exc)
        return result
    except asyncio.TimeoutError:
        result.error = "Таймаут загрузки страницы"
        logger.error("scan_avito: timeout")
        return result
    except OSError:
        result.error = "Не удалось подключиться к avito.ru"
        logger.error("scan_avito: connection error", exc_info=True)
        return result
    except Exception:
        result.error = "Неизвестная ошибка загрузки"
        logger.exception("scan_avito: неизвестная ошибка")
        return result

    # 3. Парсинг
    try:
        parsed = parse_listings(html)
    except Exception:
        result.error = "Ошибка парсинга HTML"
        logger.exception("scan_avito: ошибка парсинга")
        return result

    result.total_parsed = len(parsed)

    if not parsed:
        result.error = "Объявления не найдены"
        logger.info("scan_avito: 0 объявлений на странице")
        return result

    # 4. Рыночная статистика
    stats = _calc_market_stats(parsed)

    # 5. Оценка и проверка каждого объявления
    for listing in parsed:
        try:
            deal = calculate_deal_score(
                listing,
                avg_price=stats["avg_price"],
                min_price=stats["min_price"],
            )
            listing["deal_score"] = deal
        except Exception:
            logger.exception(
                "scan_avito: ошибка deal_score для %s", listing.get("avito_id")
            )
            listing["deal_score"] = {"score": 0, "breakdown": {}, "grade": "F"}

        try:
            scam = check_scam(listing, avg_price=stats["avg_price"])
            listing["scam_check"] = scam
        except Exception:
            logger.exception(
                "scan_avito: ошибка anti_scam для %s", listing.get("avito_id")
            )
            listing["scam_check"] = {
                "is_suspicious": False,
                "risk": "low",
                "reasons": [],
            }

    result.listings = parsed

    # 6. Инкрементальный анализ
    new_listings, price_changes, unchanged = _compare_with_db(parsed, existing)
    result.new_listings = new_listings
    result.price_changes = price_changes
    result.unchanged = unchanged

    logger.info(
        "scan_avito: всего=%d, новых=%d, цен=%d, без изменений=%d",
        len(parsed),
        len(new_listings),
        len(price_changes),
        len(unchanged),
    )

    return result


async def quick_scan(url: str) -> ScanResult:
    """Быстрое сканирование по прямой ссылке (без построения URL).

    Полезно для ручной проверки конкретной страницы.
    """
    result = ScanResult()
    result.url = url

    try:
        html = await _fetch_page(url)
    except Exception:
        result.error = "Ошибка загрузки страницы"
        logger.exception("quick_scan: ошибка загрузки %s", url)
        return result

    try:
        parsed = parse_listings(html)
    except Exception:
        result.error = "Ошибка парсинга HTML"
        logger.exception("quick_scan: ошибка парсинга")
        return result

    stats = _calc_market_stats(parsed)

    for listing in parsed:
        listing["deal_score"] = calculate_deal_score(
            listing, avg_price=stats["avg_price"], min_price=stats["min_price"]
        )
        listing["scam_check"] = check_scam(listing, avg_price=stats["avg_price"])

    result.listings = parsed
    result.total_parsed = len(parsed)
    result.new_listings = parsed  # без existing — всё новое

    return result
