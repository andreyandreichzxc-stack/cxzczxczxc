"""Hallucination Guard — проверяет утверждения модели против памяти."""

import re
import logging

logger = logging.getLogger(__name__)

# Паттерны утверждений о людях/фактах
_CLAIM_PATTERNS = [
    r"(?:сказал[а]?|говорил[а]?|писал[а]?|отправил[а]?|сделал[а]?|купил[а]?|был[а]?|ходил[а]?|ездил[а]?)\s+(?:что\s+)?(.+?)(?:[.!?]|$)",
    r"(?:он[а]?|этот человек|пользователь)\s+(.+?)(?:[.!?]|$)",
    r"(?:по словам|со слов)\s+(\w+)[,. ](.+?)(?:[.!?]|$)",
    r"(?:встреча|созвон|митинг)\s+(?:с\s+)?(\w+)\s+(?:в|на|завтра|сегодня|через)(.+?)(?:[.!?]|$)",
]


async def verify_claims(
    final_response: str,
    memory_facts: list[str],
    contact_names: list[str],
) -> dict:
    """
    Проверяет утверждения в final_response против известных фактов.

    Returns:
        {"ok": True} — всё в порядке
        {"ok": False, "unverified": ["утверждение 1", ...]} — есть неподтверждённые утверждения
    """
    if not final_response or not memory_facts:
        return {"ok": True}  # нечего проверять

    # Извлекаем утверждения о людях
    claims = []
    for pattern in _CLAIM_PATTERNS:
        matches = re.findall(pattern, final_response, re.IGNORECASE)
        for m in matches:
            claim = m if isinstance(m, str) else " ".join(filter(None, m))
            if len(claim) > 10:  # фильтруем короткие/шумные
                claims.append(claim.strip())

    if not claims:
        return {"ok": True}  # нет утверждений для проверки

    # Проверяем каждое утверждение против памяти
    unverified = []
    memory_text = " ".join(memory_facts).lower()
    all_names = "|".join(re.escape(n.lower()) for n in contact_names if n)

    for claim in claims:
        claim_lower = claim.lower()

        # Проверка 1: содержит ли память ключевые слова из утверждения
        keywords = [w for w in claim_lower.split() if len(w) > 3]
        matched_keywords = sum(1 for kw in keywords if kw in memory_text)
        keyword_ratio = matched_keywords / len(keywords) if keywords else 0

        # Проверка 2: есть ли упоминание контакта + действия
        if all_names:
            has_contact = bool(re.search(all_names, claim_lower))
        else:
            has_contact = False

        # Если утверждение содержит контакт, но память не подтверждает — подозрительно
        if has_contact and keyword_ratio < 0.3:
            unverified.append(claim)
        elif not has_contact and keyword_ratio < 0.1:
            unverified.append(claim)

    if unverified:
        logger.debug("Unverified claims: %s", unverified[:3])
        return {"ok": False, "unverified": unverified[:5]}

    return {"ok": True}


def apply_guard(
    final_response: str,
    verify_result: dict,
    confidence: float = 0.8,
) -> tuple[str, bool]:
    """
    Применяет результат верификации к ответу.

    Returns: (modified_response, was_modified)
    """
    if verify_result.get("ok", True):
        return final_response, False

    unverified = verify_result.get("unverified", [])

    # Если высокая уверенность модели, но есть неподтверждённые утверждения — добавляем дисклеймер
    if confidence > 0.5 and unverified:
        disclaimer = "\n\n⚠️ Часть информации выше не подтверждена моей памятью."
        return final_response + disclaimer, True

    # Если низкая уверенность — заменяем ответ
    return "Хм, я не уверен в точности этой информации. Давай я перепроверю?", True
