"""
LLM-driven instruction optimization and consolidation.
Replaces pure regex detection with hybrid: regex + LLM post-review.

Адаптирован под реальную схему БД:
- InstructionCandidate: rule, category, is_safe, created_at (+ llm_reviewed)
- build_provider(session, user, purpose)
- notification_queue.enqueue(topic, text, priority, ...)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import InstructionProfile, InstructionCandidate, InstructionEvent
from src.db.repo import get_or_create_user
from src.db.session import SessionLocal
from src.core.scheduling.notification_queue import notification_queue
from src.llm.router import build_provider

logger = logging.getLogger(__name__)

UTC_NOW = lambda: datetime.now(timezone.utc)


class InstructionOptimizer:
    """
    Оптимизатор инструкций.

    Задачи:
    1. consolidate_rules() — находит противоречия/дубликаты в активных правилах
    2. review_candidates() — LLM анализирует диалоги и предлагает новые правила
    3. optimize_tone() — анализирует adaptive_persona на конфликты
    """

    async def consolidate_rules(
        self,
        session: AsyncSession,
        user_id: int,
        user_obj=None,  # User object для build_provider
    ) -> dict:
        """
        Собирает все активные InstructionProfile.rules_json,
        отправляет LLM для поиска противоречий и дубликатов.

        Возвращает словарь с предложениями (не применяет автоматически!).
        """
        # 1. Получить все активные правила
        q = select(InstructionProfile).where(
            InstructionProfile.user_id == user_id,
        )
        result = await session.execute(q)
        profiles = result.scalars().all()

        if not profiles:
            return {
                "contradictions": [],
                "duplicates": [],
                "outdated": [],
                "consolidated": [],
            }

        all_rules: list[str] = []
        for p in profiles:
            if p.rules_json:
                try:
                    rules = json.loads(p.rules_json)
                    all_rules.extend(
                        rules if isinstance(rules, list) else [p.rules_json]
                    )
                except (json.JSONDecodeError, TypeError):
                    all_rules.append(p.rules_json)

        if len(all_rules) < 2:
            return {
                "contradictions": [],
                "duplicates": [],
                "outdated": [],
                "consolidated": [],
            }

        # 2. LLM ищет противоречия
        llm_result = {
            "contradictions": [],
            "duplicates": [],
            "outdated": [],
            "consolidated": [],
        }

        if user_obj:
            try:
                provider = await build_provider(
                    session, user_obj, purpose="instructions"
                )
                if provider:
                    prompt = (
                        "Ты — оптимизатор правил AI-ассистента. Проанализируй список правил и найди:\n"
                        "1. Противоречия (правило А конфликтует с правилом Б)\n"
                        "2. Дубликаты (два правила говорят об одном и том же)\n"
                        "3. Устаревшие правила\n"
                        "4. Предложи консолидированный набор\n\n"
                        "Правила:\n"
                        + "\n".join(f"{i + 1}. {r}" for i, r in enumerate(all_rules))
                        + "\n\n"
                        "Верни ТОЛЬКО JSON (без markdown-разметки):\n"
                        '{"contradictions": [{"rule_a": idx, "rule_b": idx, "reason": str}],'
                        ' "duplicates": [{"keep": idx, "remove": idx, "reason": str}],'
                        ' "outdated": [{"idx": int, "reason": str}],'
                        ' "consolidated": [str]}'
                    )
                    response = await provider.chat(
                        [{"role": "user", "content": prompt}]
                    )
                    # Извлекаем JSON из ответа (может быть обёрнут в ```json)
                    json_str = response.strip()
                    if json_str.startswith("```"):
                        json_str = json_str.split("\n", 1)[1]
                        if json_str.endswith("```"):
                            json_str = json_str[:-3]
                    llm_result = json.loads(json_str)
            except Exception:
                logger.debug("LLM consolidation failed", exc_info=True)

        # 3. Сохранить предложения как InstructionCandidate (не применяем!)
        # Проверяем safety_gate для каждого предложения
        try:
            from src.core.intelligence.soul_snapshot import soul_snapshot as _ss
        except Exception:
            _ss = None

        for dup in llm_result.get("duplicates", []):
            rule_text = f"[ДУБЛИКАТ] Правило {dup.get('keep')} и {dup.get('remove')}: {dup.get('reason', '')}"
            if _ss:
                try:
                    allowed, _ = _ss.safety_gate("context", rule_text)
                    if not allowed:
                        continue
                except Exception:
                    continue
            cand = InstructionCandidate(
                user_id=user_id,
                rule=rule_text,
                category="consolidation",
                is_safe=False,
                llm_reviewed=True,
            )
            session.add(cand)

        for contr in llm_result.get("contradictions", []):
            rule_text = f"[ПРОТИВОРЕЧИЕ] {contr.get('rule_a')} vs {contr.get('rule_b')}: {contr.get('reason', '')}"
            if _ss:
                try:
                    allowed, _ = _ss.safety_gate("context", rule_text)
                    if not allowed:
                        continue
                except Exception:
                    continue
            cand = InstructionCandidate(
                user_id=user_id,
                rule=rule_text,
                category="conflict",
                is_safe=False,
                llm_reviewed=True,
            )
            session.add(cand)

        await session.commit()

        # 4. Уведомить владельца
        total_issues = len(llm_result.get("contradictions", [])) + len(
            llm_result.get("duplicates", [])
        )
        if total_issues > 0:
            await notification_queue.enqueue(
                topic="instructions",
                text=(
                    f"🤖 Найдено {len(llm_result.get('contradictions', []))} противоречий "
                    f"и {len(llm_result.get('duplicates', []))} дубликатов в правилах. "
                    f"Проверьте /settings → instructions"
                ),
                priority=2,  # MEDIUM
                category="instructions",
            )

        return llm_result

    async def post_turn_review(
        self,
        session: AsyncSession,
        user_id: int,
        user_obj=None,
        user_message: str = "",
        assistant_response: str = "",
    ) -> list[InstructionCandidate]:
        """
        После каждого ответа ассистента — LLM анализирует:
        - Не противоречит ли ответ памяти?
        - Какие новые правила стоит предложить?

        Создаёт InstructionCandidate со статусом is_safe=False (pending).
        """
        candidates: list[InstructionCandidate] = []

        if not user_obj or not user_message or not assistant_response:
            return candidates

        # LLM анализ диалога
        try:
            provider = await build_provider(session, user_obj, purpose="instructions")
            if provider:
                prompt = (
                    "Проанализируй диалог и предложи улучшения правил ассистента:\n\n"
                    f"Пользователь: {user_message}\n"
                    f"Ассистент: {assistant_response}\n\n"
                    "1. Есть ли в ответе что-то, что противоречит предыдущим правилам?\n"
                    "2. Какое новое правило стоило бы добавить на основе этого диалога?\n"
                    "3. Нужно ли изменить тон/формат ответа?\n\n"
                    "Верни ТОЛЬКО JSON (без markdown-разметки):\n"
                    '{"conflicts": [str], "suggestions": [str], "tone_changes": [str]}'
                )
                response = await provider.chat([{"role": "user", "content": prompt}])
                json_str = response.strip()
                if json_str.startswith("```"):
                    json_str = json_str.split("\n", 1)[1]
                    if json_str.endswith("```"):
                        json_str = json_str[:-3]
                llm_result = json.loads(json_str)
            else:
                llm_result = {"conflicts": [], "suggestions": [], "tone_changes": []}
        except Exception:
            logger.debug("LLM post-turn review failed", exc_info=True)
            llm_result = {"conflicts": [], "suggestions": [], "tone_changes": []}

        for suggestion in llm_result.get("suggestions", []):
            # Проверяем safety_gate перед сохранением
            try:
                from src.core.intelligence.soul_snapshot import soul_snapshot

                allowed, _ = soul_snapshot.safety_gate("context", suggestion)
                if not allowed:
                    continue
            except Exception:
                continue

            cand = InstructionCandidate(
                user_id=user_id,
                rule=suggestion[:500],  # обрезаем длинные
                category="llm_suggestion",
                is_safe=False,
                llm_reviewed=True,
            )
            session.add(cand)
            candidates.append(cand)

        await session.commit()
        return candidates

    async def instruction_optimizer_loop(self, user_id: int):
        """
        Ежедневный цикл оптимизации инструкций.
        Запускать раз в день (например, в 02:00 по крону).
        """
        async with SessionLocal() as session:
            # user_id здесь — это telegram_id (по конвенции репозитория)
            owner = await get_or_create_user(session, user_id)
            await self.consolidate_rules(
                session=session,
                user_id=owner.id,
                user_obj=owner,
            )


instruction_optimizer = InstructionOptimizer()
