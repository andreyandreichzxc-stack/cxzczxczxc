"""Contradiction detector — finds contradictions between new statements and stored memory."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ── Pending contradictions storage ──────────────────────────────
# Format: {telegram_id: dict with {contradicted_fact, confidence, memory_id, suggestion, stored_at}}
_pending_contradictions: dict[int, dict[str, Any]] = {}
_pending_lock: asyncio.Lock = asyncio.Lock()
_PENDING_TTL: float = 3600.0  # 1 hour — auto-expire unhandled contradictions

# ── Contradiction patterns ──────────────────────────────────────

# Russian negation words + patterns
_NEGATION_WORDS = frozenset(
    {
        "не",
        "нет",
        "никогда",
        "перестал",
        "перестала",
        "бросил",
        "бросила",
        "уже не",
        "больше не",
        "не буду",
        "не хочу",
        "не стану",
    }
)

# Words suggesting a positive statement (opposite of negation)
_AFFIRMATION_PATTERNS = re.compile(
    r"\b(?:да|буду|хочу|стану|люблю|нравится|закажу|куплю|возьму|пойду|сделаю)\b",
    re.IGNORECASE,
)

# Opposite concept pairs for semantic contradiction detection
_OPPOSITE_PAIRS: list[tuple[set[str], set[str]]] = [
    # Food/drink category opposites
    (
        {
            "кофе",
            "латте",
            "капучино",
            "эспрессо",
            "американо",
            "раф",
            "мокачино",
            "кофеин",
            "зерновой",
            "молотый",
        },
        set(),
    ),  # no direct opposite — any coffee word contradicts "не пью кофе"
    ({"чай", "чаёк", "чаю", "заварка"}, {"кофе", "латте", "капучино", "эспрессо"}),
    # Alcohol
    (
        {
            "алкоголь",
            "выпить",
            "пью",
            "водка",
            "вино",
            "пиво",
            "коньяк",
            "виски",
            "шампанское",
            "коктейль",
        },
        set(),
    ),
    # Work
    (
        {"работаю", "работа", "офис", "компания", "фирма", "зарплата", "трудоустроен"},
        {"безработный", "уволился", "уволен", "не работаю", "декрет"},
    ),
    # Diet
    ({"веган", "вегетарианец", "мясоед", "сыроед"}, set()),
    ({"мясо", "стейк", "курица", "говядина", "свинина", "шашлык", "бургер"}, set()),
    # Smoking
    ({"курю", "сигареты", "вейп", "калья"}, set()),
]

# Words that are too common to be meaningful keywords
_STOP_WORDS = frozenset(
    {
        "это",
        "что",
        "как",
        "так",
        "вот",
        "ещё",
        "уже",
        "есть",
        "было",
        "быть",
        "тебя",
        "тебе",
        "мне",
        "меня",
        "себя",
        "себе",
        "очень",
        "просто",
        "сейчас",
        "тогда",
        "потом",
        "можно",
        "надо",
        "нужно",
        "будет",
        "когда",
        "если",
        "чтобы",
        "всех",
        "всем",
        "всего",
        "все",
        "там",
        "тут",
        "здесь",
        "мой",
        "твой",
        "свой",
    }
)

# Minimum word length for keyword extraction
_MIN_WORD_LEN = 3


def _extract_keywords(text: str) -> set[str]:
    """Extract meaningful lowercase keywords from text, filtering stop words."""
    words = re.findall(r"[а-яёa-z]{%d,}" % _MIN_WORD_LEN, text.lower())
    return {w for w in words if w not in _STOP_WORDS}


def _has_negation(text: str) -> bool:
    """Check if text contains negation words."""
    text_lower = text.lower()
    return any(neg in text_lower for neg in _NEGATION_WORDS)


def _check_opposite_categories(user_text: str, fact_text: str) -> bool:
    """Check if user_text and fact_text belong to opposite concept categories."""
    user_lower = user_text.lower()
    fact_lower = fact_text.lower()

    for category_a, category_b in _OPPOSITE_PAIRS:
        # Check if fact has words from a category
        fact_in_a = any(w in fact_lower for w in category_a)
        fact_in_b = any(w in fact_lower for w in category_b)

        # Check if user text has words from a category
        user_in_a = any(w in user_lower for w in category_a)
        user_in_b = any(w in user_lower for w in category_b)

        # Contradiction: one side says A, other side says B (for explicit opposites)
        if fact_in_a and user_in_b and category_b:
            return True
        if fact_in_b and user_in_a and category_b:
            return True

        # Contradiction: fact says "не [category]", user uses words from [category]
        if _has_negation(fact_text) and (user_in_a or user_in_b):
            return True
        if _has_negation(user_text) and (fact_in_a or fact_in_b):
            return True

    return False


async def detect_contradiction(
    telegram_id: int,
    user_text: str,
) -> dict[str, Any] | None:
    """Check if *user_text* contradicts any stored memory fact.

    Uses lightweight recall (SQLite-only, mode="light") for fast checking.

    Returns:
        None if no contradiction found.
        dict with {contradicted_fact, confidence, memory_id, suggestion} if found.
    """
    from src.core.memory.memory_recall import recall

    # Fast light-mode recall — SQLite only, no Qdrant, no deep memory
    try:
        result = await recall(
            telegram_id=telegram_id,
            query=user_text[:200],
            limit=5,
            mode="light",
            include_self=True,
            include_deep=False,
        )
    except Exception:
        logger.debug("recall failed during contradiction check", exc_info=True)
        return None

    if not result.facts:
        return None

    user_keywords = _extract_keywords(user_text)
    if len(user_keywords) < 2:
        return None  # too short for meaningful contradiction

    user_has_negation = _has_negation(user_text)

    best_contradiction: dict[str, Any] | None = None
    best_confidence: float = 0.0

    for fact in result.facts:
        fact_keywords = _extract_keywords(fact.fact)
        overlap = user_keywords & fact_keywords

        fact_has_negation = _has_negation(fact.fact)

        contradiction_found = False
        confidence_boost = 0.0

        # ── Heuristic 1: Negation mismatch on overlapping keywords ──
        # If one text has negation and the other doesn't, and they share
        # significant keywords → possible contradiction
        if len(overlap) >= 2 and user_has_negation != fact_has_negation:
            contradiction_found = True
            confidence_boost = 0.15

        # ── Heuristic 2: Semantic opposite categories ──
        if not contradiction_found and _check_opposite_categories(user_text, fact.fact):
            contradiction_found = True
            confidence_boost = 0.20

        # ── Heuristic 3: Sentiment flip (positive ↔ negative on same topic) ──
        if not contradiction_found:
            user_sentiment_words = _AFFIRMATION_PATTERNS.findall(user_text.lower())
            if user_sentiment_words and fact_has_negation and len(overlap) >= 1:
                contradiction_found = True
                confidence_boost = 0.10

        if not contradiction_found:
            continue

        # Calculate confidence
        overlap_ratio = len(overlap) / max(len(user_keywords), 1)
        base_confidence = min(overlap_ratio + confidence_boost + 0.3, 0.95)
        confidence = max(base_confidence, fact.confidence or 0.5)

        # Prefer higher-confidence contradictions
        if confidence > best_confidence:
            best_confidence = confidence
            best_contradiction = {
                "contradicted_fact": fact.fact,
                "confidence": round(confidence, 2),
                "memory_id": fact.memory_id,
                "suggestion": (f"Ты говорил что {fact.fact}. Передумал или уточнить?"),
            }

    return best_contradiction


async def store_pending_contradiction(
    telegram_id: int,
    contradiction: dict[str, Any],
) -> None:
    """Store a pending contradiction so we can check the user's next response."""
    import time as _time

    contradiction["stored_at"] = _time.monotonic()
    async with _pending_lock:
        # Evict expired entries before adding
        now = _time.monotonic()
        expired = [
            uid
            for uid, d in _pending_contradictions.items()
            if now - d.get("stored_at", 0) > _PENDING_TTL
        ]
        for uid in expired:
            del _pending_contradictions[uid]
        _pending_contradictions[telegram_id] = contradiction


async def pop_pending_contradiction(
    telegram_id: int,
) -> dict[str, Any] | None:
    """Retrieve and remove a pending contradiction for the user.

    Returns None if no pending contradiction or if it expired.
    """
    import time as _time

    async with _pending_lock:
        pending = _pending_contradictions.pop(telegram_id, None)
        if pending is None:
            return None
        # Check TTL
        if _time.monotonic() - pending.get("stored_at", 0) > _PENDING_TTL:
            return None  # expired
        return pending


async def check_contradiction_response(
    telegram_id: int,
    user_text: str,
) -> str | None:
    """Check if user_text is a response to a pending contradiction question.

    Returns:
        A response message string if the user confirmed/denied, or None if
        this message is unrelated to the pending contradiction.
    """
    pending = await pop_pending_contradiction(telegram_id)
    if pending is None:
        return None

    user_lower = user_text.lower().strip()

    # Short response detection
    confirm_words = {
        "да",
        "ага",
        "угу",
        "верно",
        "правильно",
        "изменилось",
        "передумал",
        "передумала",
        "точно",
        "давай",
        "ок",
        "хорошо",
        "забудь",
        "сотри",
        "удали",
    }
    deny_words = {
        "нет",
        "не",
        "неправда",
        "ошибка",
        "не так",
        "неверно",
        "не менял",
        "не менялось",
    }

    # Check if it looks like a direct short answer to our contradiction question
    # (strip punctuation for matching)
    cleaned = user_lower.strip(".,!?;: ")
    first_word = cleaned.split()[0] if cleaned else ""

    if cleaned in confirm_words or first_word in confirm_words:
        # User confirms the change — mark old fact as inactive + contradictory
        old_memory_id = pending.get("memory_id")
        if old_memory_id:
            try:
                from src.db.models import Memory
                from src.db.repo import get_or_create_user
                from src.db.session import get_session

                async with get_session() as link_session:
                    link_owner = await get_or_create_user(link_session, telegram_id)
                    old_mem = await link_session.get(Memory, old_memory_id)
                    if old_mem is not None and old_mem.user_id == link_owner.id:
                        old_mem.sentiment = "contradictory"
                        old_mem.is_active = False
                        await link_session.commit()
            except Exception:
                logger.debug("Failed to mark contradicted fact", exc_info=True)

        return (
            f"🧠 Понял! Запомню, что «{pending['contradicted_fact']}» "
            f"больше не актуально. Спасибо за уточнение!"
        )

    if cleaned in deny_words or first_word in deny_words:
        # User denies — fact stays as is
        return (
            "👍 Ок, оставляю как было. Если что — просто скажи «забудь про ...» "
            "или «я больше не ...»."
        )

    # Message is too long/complex to be a direct answer to our question.
    # Re-store the pending contradiction so next response can resolve it.
    # Refresh stored_at to reset TTL.
    import time as _time

    pending["stored_at"] = _time.monotonic()
    async with _pending_lock:
        _pending_contradictions[telegram_id] = pending
    return None


# _BATCH_OPPOSITE_PAIRS for batch scanning (simple string pairs, not set-based)
_BATCH_OPPOSITE_PAIRS = [
    ("люблю", "не люблю"),
    ("пью", "не пью"),
    ("работаю", "не работаю"),
    ("учусь", "не учусь"),
    ("занимаюсь", "не занимаюсь"),
    ("курю", "не курю"),
    ("ем", "не ем"),
    ("хожу", "не хожу"),
    ("смотрю", "не смотрю"),
    ("играю", "не играю"),
    ("читаю", "не читаю"),
]


async def _scan_contradictions_batch(
    memories, owner_id: int, *, session=None, owner=None
) -> int:
    """Batch scan all memories for contradictions. No LLM — pure heuristic.
    Returns count of found contradiction pairs.

    If *session* and *owner* are provided, creates MemoryLink edges
    (relation_type='contradicts') for each detected contradiction pair.
    """
    found = 0
    _MAX_PAIRS = min(len(memories) * 50, 5000)
    pairs_checked = 0
    for i, m1 in enumerate(memories):
        if not m1.fact or not m1.is_active:
            continue
        f1 = m1.fact.lower()
        for j in range(i + 1, min(i + 50, len(memories))):
            if pairs_checked >= _MAX_PAIRS:
                return found
            m2 = memories[j]
            if not m2.fact or not m2.is_active or m1.contact_id != m2.contact_id:
                continue
            f2 = m2.fact.lower()
            pairs_checked += 1
            # Check 1: Opposite pairs
            for pos, neg in _BATCH_OPPOSITE_PAIRS:
                if (pos in f1 and neg in f2) or (neg in f1 and pos in f2):
                    found += 1
                    # Create contradiction MemoryLink edge
                    if session is not None and owner is not None:
                        try:
                            from src.db.repo import link_memories

                            await link_memories(
                                session,
                                owner,
                                source_id=m1.id,
                                target_id=m2.id,
                                relation_type="contradicts",
                                weight=0.8,
                            )
                        except Exception:
                            pass
                    break
            # Check 2: Negation mismatch
            if found < 100:
                words1 = {w for w in f1.split() if len(w) >= 4}
                words2 = {w for w in f2.split() if len(w) >= 4}
                common = words1 & words2
                if common and len(common) >= 2:
                    has_neg1 = any(
                        neg in f1 for neg in ("не ", "нет ", "никогда ", "больше не ")
                    )
                    has_neg2 = any(
                        neg in f2 for neg in ("не ", "нет ", "никогда ", "больше не ")
                    )
                    if has_neg1 != has_neg2:
                        found += 1
                        # Create contradiction MemoryLink edge
                        if session is not None and owner is not None:
                            try:
                                from src.db.repo import link_memories

                                await link_memories(
                                    session,
                                    owner,
                                    source_id=m1.id,
                                    target_id=m2.id,
                                    relation_type="contradicts",
                                    weight=0.7,
                                )
                            except Exception:
                                pass
            if found >= 100:
                return found
    return found
