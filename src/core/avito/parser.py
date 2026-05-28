"""HTML-парсер результатов поиска Авито.

Извлекает объявления из HTML-страницы поиска Avito.
Поддерживает несколько стратегий селекторов для устойчивости к изменениям вёрстки.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  Типы
# ═══════════════════════════════════════════════════════════════════════════

ListingDict = dict[str, Any]


# ═══════════════════════════════════════════════════════════════════════════
#  Утилиты
# ═══════════════════════════════════════════════════════════════════════════


def _clean(text: str | None) -> str:
    """Убирает лишние пробелы и неразрывные символы."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def _attr(el: Tag | None, name: str) -> str:
    """Безопасно извлекает строковый атрибут из BS4-элемента."""
    if el is None:
        return ""
    val = el.get(name)
    if val is None:
        return ""
    return str(val)


def _parse_price(raw: str | None) -> int | None:
    """Извлекает числовое значение цены из строки ('12 000 ₽' → 12000)."""
    if not raw:
        return None
    digits = re.sub(r"[^\d]", "", str(raw))
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def _extract_avito_id(url: str | None) -> str | None:
    """Извлекает ID объявления из URL (последний числовой сегмент)."""
    if not url:
        return None
    m = re.search(r"_(\d+)$", url.rstrip("/"))
    if m:
        return m.group(1)
    m = re.search(r"/(\d+)$", url.rstrip("/"))
    return m.group(1) if m else None


# ═══════════════════════════════════════════════════════════════════════════
#  Стратегии извлечения полей (fallback-цепочки)
# ═══════════════════════════════════════════════════════════════════════════


def _get_title(item: Tag) -> str:
    """Заголовок объявления — несколько стратегий."""
    # 1) data-marker="item-title"
    el = item.select_one('[data-marker="item-title"]')
    if el:
        return _clean(el.get_text())
    # 2) <a> с itemprop="name"
    el = item.select_one('a[itemprop="name"]')
    if el:
        return _clean(el.get_text())
    # 3) Первый <a> с текстом внутри item-description
    el = item.select_one('a[itemprop="url"]')
    if el:
        title = _attr(el, "title") or el.get_text()
        return _clean(title)
    # 4) h3
    el = item.select_one("h3")
    if el:
        return _clean(el.get_text())
    return ""


def _get_price(item: Tag) -> int | None:
    """Цена — несколько стратегий."""
    # 1) <meta itemprop="price" content="...">
    meta = item.select_one('meta[itemprop="price"]')
    if meta and meta.get("content"):
        val = _parse_price(_attr(meta, "content"))
        if val is not None:
            return val
    # 2) data-marker="item-price"
    el = item.select_one('[data-marker="item-price"]')
    if el:
        val = _parse_price(el.get_text())
        if val is not None:
            return val
    # 3) item-price в itemprop
    el = item.select_one('[itemprop="price"]')
    if el:
        raw = _attr(el, "content") or el.get_text()
        val = _parse_price(raw)
        if val is not None:
            return val
    # 4) Класс с "price" в названии (последний fallback)
    for span in item.select("span"):
        cls = " ".join(str(c) for c in (span.get("class") or []))
        if "price" in cls.lower():
            val = _parse_price(span.get_text())
            if val is not None:
                return val
    return None


def _get_url(item: Tag) -> str:
    """URL объявления."""
    el = item.select_one('[data-marker="item-title"]')
    if el and el.get("href"):
        href = _attr(el, "href")
        if href.startswith("/"):
            return f"https://www.avito.ru{href}"
        return href
    el = item.select_one('a[itemprop="url"]')
    if el and el.get("href"):
        href = _attr(el, "href")
        if href.startswith("/"):
            return f"https://www.avito.ru{href}"
        return href
    # Первый <a> с href на /...
    for a in item.select("a[href]"):
        href = _attr(a, "href")
        if "/_next/" not in href and href.startswith("/"):
            return f"https://www.avito.ru{href}"
    return ""


def _get_image_url(item: Tag) -> str:
    """URL изображения."""
    # 1) <img> внутри data-marker="item-photo"
    photo_el = item.select_one('[data-marker="item-photo"]')
    if photo_el:
        img = photo_el.select_one("img")
        if img:
            src = _attr(img, "src") or _attr(img, "data-src")
            if src:
                return src
    # 2) Первый <img> с itemprop="image"
    img = item.select_one('img[itemprop="image"]')
    if img:
        src = _attr(img, "src") or _attr(img, "data-src")
        if src:
            return src
    # 3) Первый <img> с src
    for img in item.select("img"):
        src = _attr(img, "src") or _attr(img, "data-src")
        if src and "avito" in src:
            return src
    return ""


def _get_city(item: Tag) -> str:
    """Город."""
    el = item.select_one('[data-marker="item-address"]')
    if el:
        return _clean(el.get_text())
    # itemprop="address"
    el = item.select_one('[itemprop="address"]')
    if el:
        return _clean(el.get_text())
    # geo-параметр в data-атрибутах
    el = item.select_one("[data-item-address]")
    if el:
        return _clean(_attr(el, "data-item-address"))
    return ""


def _get_condition(item: Tag) -> str:
    """Состояние товара (новый/б/у и т.д.)."""
    # Ищем текст вида "Новый", "Б/У", "Отличное" и т.п.
    for el in item.select("span"):
        text = _clean(el.get_text()).lower()
        if text in ("новый", "отличное", "хорошее", "удовлетворительное"):
            return _clean(el.get_text())
        if "б/у" in text:
            return "Б/У"
    # data-marker
    el = item.select_one('[data-marker="item-condition"]')
    if el:
        return _clean(el.get_text())
    return ""


def _get_delivery(item: Tag) -> bool:
    """Есть ли доставка."""
    el = item.select_one('[data-marker="item-delivery"]')
    if el:
        return True
    text = item.get_text().lower()
    return "доставк" in text


def _get_seller_name(item: Tag) -> str:
    """Имя продавца."""
    el = item.select_one('[data-marker="seller-name"]')
    if el:
        return _clean(el.get_text())
    el = item.select_one('[itemprop="name"]')
    if (
        el
        and el.parent
        and "seller" in " ".join(str(c) for c in (el.parent.get("class") or [])).lower()
    ):
        return _clean(el.get_text())
    return ""


def _get_seller_rating(item: Tag) -> float | None:
    """Рейтинг продавца (0-5)."""
    el = item.select_one('[data-marker="seller-rating"]')
    if el:
        text = el.get_text().strip()
        try:
            return float(text.replace(",", "."))
        except ValueError:
            pass
    # aria-label вида "Рейтинг 4.8"
    for el in item.select("[aria-label]"):
        label = _attr(el, "aria-label")
        m = re.search(r"(\d+[.,]\d+)", label)
        if m and "рейтинг" in label.lower():
            return float(m.group(1).replace(",", "."))
    return None


def _get_seller_reviews(item: Tag) -> int | None:
    """Количество отзывов продавца."""
    el = item.select_one('[data-marker="seller-reviews"]')
    if el:
        digits = re.sub(r"[^\d]", "", el.get_text())
        if digits:
            return int(digits)
    # Ищем "N отзывов" рядом с seller-name
    for el in item.select("span"):
        text = _clean(el.get_text())
        m = re.match(r"(\d+)\s*отзыв", text)
        if m:
            return int(m.group(1))
    return None


def _get_description(item: Tag) -> str:
    """Описание (краткое)."""
    el = item.select_one('[data-marker="item-description"]')
    if el:
        return _clean(el.get_text())
    # itemprop="description"
    el = item.select_one('[itemprop="description"]')
    if el:
        return _clean(el.get_text())
    return ""


# ═══════════════════════════════════════════════════════════════════════════
#  Основной парсер
# ═══════════════════════════════════════════════════════════════════════════


def parse_listings(html: str) -> list[ListingDict]:
    """Парсит HTML страницы поиска Авито и возвращает список объявлений.

    Каждый элемент — dict с ключами:
        avito_id, title, price, url, image_url, city, condition,
        delivery, seller_name, seller_rating, seller_reviews, description
    """
    if not html or not html.strip():
        logger.warning("parse_listings: пустой HTML")
        return []

    soup = BeautifulSoup(html, "html.parser")

    # ── Поиск контейнеров объявлений ──────────────────────────────────
    items: list[Tag] = []

    # 1) data-marker="item"
    items = soup.select('[data-marker="item"]')

    # 2) itemprop="itemListElement" (schema.org)
    if not items:
        items = soup.select('[itemprop="itemListElement"]')

    # 3) Класс catalog-item (legacy)
    if not items:
        items = soup.select(".catalog-item")

    # 4) Секции с data-marker="catalog-serp"
    if not items:
        serp = soup.select_one('[data-marker="catalog-serp"]')
        if serp:
            items = serp.select('[data-marker="item"]')

    if not items:
        logger.info("parse_listings: объявления не найдены (items=0)")
        return []

    listings: list[ListingDict] = []

    for item in items:
        url = _get_url(item)
        avito_id = _extract_avito_id(url)

        listing: ListingDict = {
            "avito_id": avito_id,
            "title": _get_title(item),
            "price": _get_price(item),
            "url": url,
            "image_url": _get_image_url(item),
            "city": _get_city(item),
            "condition": _get_condition(item),
            "delivery": _get_delivery(item),
            "seller_name": _get_seller_name(item),
            "seller_rating": _get_seller_rating(item),
            "seller_reviews": _get_seller_reviews(item),
            "description": _get_description(item),
        }

        # Пропускаем объявления без ID и заголовка
        if not listing["avito_id"] and not listing["title"]:
            continue

        listings.append(listing)

    logger.info("parse_listings: найдено %d объявлений", len(listings))
    return listings
