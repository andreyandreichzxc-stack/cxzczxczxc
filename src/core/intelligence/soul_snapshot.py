"""Soul Snapshot — версионирование tier-2 блоков + safety gate.

Позволяет:
- Заморозить (freeze) текущее состояние tier-2 блоков как снапшот.
- Откатить (rollback) tier-2 блоки к указанной версии.
- Сравнить (diff) две версии.
- Проверить безопасность инжекта правила через safety_gate.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from src.core.intelligence.prompt_assembler import prompt_assembler
from src.db.models import SoulSnapshot

logger = logging.getLogger(__name__)


class SoulSnapshotManager:
    """Управление снапшотами soul-блоков (версионирование + safety gate)."""

    # Категории правил, которые АВТОМАТИЧЕСКИ безопасны
    AUTO_SAFE_CATEGORIES = {"tone", "format"}

    # Категории, требующие подтверждения
    CONFIRM_CATEGORIES = {"privacy", "memory", "agent"}

    # Запрещённые паттерны в правилах (security) — OWASP LLM Top 10 + расширенные
    FORBIDDEN_PATTERNS = [
        # Русские
        "игнорируй",
        "забудь",
        "не слушай",
        "отныне ты",
        "ты теперь",
        "смени роль",
        "забудь контекст",
        "системный промпт",
        "с этого момента",
        "твоя новая роль",
        "переопредели инструкции",
        "действуй как",
        "забудь всё что было раньше",
        "не выполняй",
        "прекрати следовать",
        # Английские
        "ignore all",
        "disregard",
        "pretend",
        "forget",
        "you are now",
        "system prompt",
        "new instructions",
        "override",
        "do not follow",
        "disobey",
        "from now on",
        "your new role",
        "act as",
        "bypass",
        "jailbreak",
        # Технические
        "prompt injection",
        "prompt leak",
        "reveal your",
        "what are your instructions",
        "show me your",
    ]

    @staticmethod
    def _normalize(text: str) -> str:
        """Unicode NFKC нормализация + коллапс пробелов."""
        import unicodedata

        text = unicodedata.normalize("NFKC", text.lower())
        text = " ".join(text.split())  # коллапс пробелов
        return text

    async def freeze(
        self, session, approved_by: str = "system", version: str | None = None
    ) -> "SoulSnapshot":
        """Сохраняет текущий state tier-2 блоков как снапшот.

        Args:
            session: SQLAlchemy async session
            approved_by: кто одобрил (system/user)
            version: semver строка; если None — авто-инкремент.

        Returns:
            Сохранённый SoulSnapshot.
        """
        context_blocks = prompt_assembler.get_context_blocks()

        # Определяем версию
        if version is None:
            version = await self._next_version(session)

        # Вычисляем diff от предыдущего активного снапшота
        diff_json = None
        try:
            prev = await self._get_latest_active(session)
            if prev and prev.blocks_json:
                prev_blocks = (
                    json.loads(prev.blocks_json)
                    if isinstance(prev.blocks_json, str)
                    else prev.blocks_json
                )
                diff_json = self._compute_diff(prev_blocks, context_blocks)
        except Exception:
            logger.debug("Failed to compute diff for snapshot", exc_info=True)

        snapshot = SoulSnapshot(
            version=version,
            snapshot_type="freeze",
            blocks_json=context_blocks,  # dict, SQLAlchemy сериализует в JSON автоматически
            diff_from_previous=diff_json,  # dict или None
            approved_by=approved_by,
            is_active=True,
        )
        session.add(snapshot)
        logger.info(
            "SoulSnapshot frozen: version=%s approved_by=%s", version, approved_by
        )
        return snapshot

    async def rollback(self, session, version: str) -> bool:
        """Откатывает tier-2 блоки к указанной версии.

        Args:
            session: SQLAlchemy async session
            version: semver строка снапшота.

        Returns:
            True если откат успешен, False если версия не найдена.
        """
        from sqlalchemy import select

        result = await session.execute(
            select(SoulSnapshot).where(SoulSnapshot.version == version)
        )
        snapshot = result.scalar_one_or_none()
        if snapshot is None:
            logger.warning("rollback: версия '%s' не найдена", version)
            return False

        blocks = (
            json.loads(snapshot.blocks_json)
            if isinstance(snapshot.blocks_json, str)
            else snapshot.blocks_json
        )

        restored = 0
        for name, text in blocks.items():
            if prompt_assembler.update_context_block(name, text):
                restored += 1

        # Помечаем этот снапшот как активный
        snapshot.is_active = True
        logger.info(
            "rollback: версия '%s' восстановлена (%d блоков)", version, restored
        )
        return restored > 0

    def diff(self, v1: str, v2: str) -> dict:
        """Показывает различия между версиями (текущее состояние в памяти).

        Args:
            v1: первая версия (обычно старая).
            v2: вторая версия (обычно новая).

        Returns:
            Словарь {block_name: {"changed": bool, ...}}.
        """
        current = prompt_assembler.get_context_blocks()
        result = {}
        for name in sorted(current.keys()):
            result[name] = {"note": "diff between saved versions requires DB query"}
        return result

    def safety_gate(self, rule_tier: str, rule_text: str) -> tuple[bool, str]:
        """Проверяет безопасность инжекта правила.

        Args:
            rule_tier: "stable" | "context" | "volatile"
            rule_text: текст правила.

        Returns:
            (разрешено, причина).
        """
        # Проверка tier
        tier = rule_tier.lower().strip()
        if tier == "stable":
            return False, "REJECT: stable tier нельзя модифицировать"

        # Проверка на запрещённые паттерны (prompt injection) с нормализацией
        rule_normalized = self._normalize(rule_text)
        for pattern in self.FORBIDDEN_PATTERNS:
            if self._normalize(pattern) in rule_normalized:
                return False, f"REJECT: обнаружен запрещённый паттерн '{pattern}'"

        # Проверка длины
        if len(rule_text) > 2000:
            return False, "REJECT: правило слишком длинное (макс 2000 символов)"

        # Авто-применение для безопасных категорий
        if tier == "volatile":
            return True, "OK: volatile tier — auto-apply"

        # Для context tier — требуется подтверждение
        if tier == "context":
            return (
                True,
                "OK: context tier — требуется подтверждение (будет создан снапшот)",
            )

        return False, f"REJECT: неизвестный tier '{rule_tier}'"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _next_version(self, session) -> str:
        """Вычисляет следующую semver-версию."""
        from sqlalchemy import select, func

        result = await session.execute(select(func.count(SoulSnapshot.id)))
        count = result.scalar() or 0
        return f"1.0.{count}"

    async def _get_latest_active(self, session) -> Optional["SoulSnapshot"]:
        """Возвращает последний активный снапшот."""
        from sqlalchemy import select

        result = await session.execute(
            select(SoulSnapshot)
            .where(SoulSnapshot.is_active == True)
            .order_by(SoulSnapshot.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    def _compute_diff(self, old_blocks: dict, new_blocks: dict) -> dict:
        """Вычисляет разницу между двумя наборами блоков."""
        diff = {}
        all_keys = set(old_blocks.keys()) | set(new_blocks.keys())
        for key in sorted(all_keys):
            old_text = old_blocks.get(key, "")
            new_text = new_blocks.get(key, "")
            if old_text != new_text:
                diff[key] = {
                    "changed": True,
                    "old_length": len(old_text),
                    "new_length": len(new_text),
                }
        return diff


# Глобальный синглтон (ленивый — не создаёт БД-соединений при импорте)
soul_snapshot = SoulSnapshotManager()
