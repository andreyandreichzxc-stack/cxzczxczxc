"""Learning/adaptation models: AgentCache, SelfProfile, AdaptivePersona, SoulSnapshot,
Trajectory, Skill, SkillUsage, InstructionProfile, InstructionCandidate, InstructionEvent."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from ._base import Base


class AgentCache(Base):
    """Кэш результатов сабагентов."""

    __tablename__ = "agent_cache"

    cache_key: Mapped[str] = mapped_column(String(128), primary_key=True)
    result_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    ttl_seconds: Mapped[int] = mapped_column(Integer, default=0)


class SelfProfile(Base):
    """Память о владельце — предпочтения, цели, проекты, стиль."""

    __tablename__ = "self_profile"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), unique=True, index=True
    )

    preferences: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # JSON ["чай", "утренние созвоны", ...]
    goals: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # JSON ["закончить проект X", ...]
    current_projects: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    decision_style: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # "быстрый"/"аналитический"/"советуется"
    communication_preferences: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # JSON
    sleep_pattern: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # "сова"/"жаворонок"/"00:00-08:00"
    work_hours: Mapped[str | None] = mapped_column(
        String(32), nullable=True
    )  # "09:00-18:00"

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class InstructionProfile(Base):
    """Профиль инструкций — активные правила поведения бота."""

    __tablename__ = "instruction_profiles"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    rules_json: Mapped[str] = mapped_column(Text, default="[]")  # JSON список правил
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class InstructionCandidate(Base):
    """Кандидат в инструкции — предложенное правило, ждёт подтверждения."""

    __tablename__ = "instruction_candidates"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    rule: Mapped[str] = mapped_column(Text)  # текст правила
    category: Mapped[str] = mapped_column(
        String(32), default="tone"
    )  # tone/format/privacy/memory/agent/llm_suggestion/consolidation/conflict
    is_safe: Mapped[bool] = mapped_column(
        Boolean, default=False
    )  # безопасное → авто-применить
    llm_reviewed: Mapped[bool] = mapped_column(
        Boolean, default=False
    )  # обработано LLM-оптимизатором
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class InstructionEvent(Base):
    """Событие — когда пользователь дал обратную связь."""

    __tablename__ = "instruction_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    raw_text: Mapped[str] = mapped_column(Text)  # что сказал пользователь
    detected_rule: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # какое правило извлекли
    action: Mapped[str] = mapped_column(
        String(16), default="detected"
    )  # detected/applied/asked/ignored
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class AdaptivePersona(Base):
    """Адаптивный профиль личности бота — стиль общения подстраивается под пользователя."""

    __tablename__ = "adaptive_personas"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), unique=True, index=True
    )
    # Стиль
    brevity: Mapped[str] = mapped_column(
        String(16), default="normal"
    )  # short/normal/detailed
    formality: Mapped[str] = mapped_column(
        String(16), default="friendly"
    )  # formal/friendly/casual
    emoji_usage: Mapped[str] = mapped_column(
        String(16), default="normal"
    )  # none/minimal/normal/rich
    initiative: Mapped[str] = mapped_column(
        String(16), default="reactive"
    )  # reactive/proactive/balanced
    # Формат
    preferred_format: Mapped[str] = mapped_column(
        String(16), default="text"
    )  # text/bullets/numbered
    use_html: Mapped[bool] = mapped_column(Boolean, default=True)
    max_response_len: Mapped[int] = mapped_column(
        Integer, default=500
    )  # макс символов в ответе
    # Запреты
    forbidden_patterns: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # JSON-список запретов
    # Режимы
    quiet_hours_active: Mapped[bool] = mapped_column(Boolean, default=False)
    work_mode: Mapped[str] = mapped_column(
        String(16), default="normal"
    )  # normal/focus/relax
    # ── ChatGPT-style поля настройки личности ──
    base_tone: Mapped[str] = mapped_column(
        String(32), default="default", nullable=False
    )  # default/professional/friendly/frank/whimsical/efficient/cynical
    warmth: Mapped[str] = mapped_column(
        String(16), default="normal", nullable=False
    )  # low/normal/high
    enthusiasm: Mapped[str] = mapped_column(
        String(16), default="normal", nullable=False
    )  # low/normal/high
    headings_lists: Mapped[str] = mapped_column(
        String(16), default="normal", nullable=False
    )  # low/normal/high
    emoji_level: Mapped[str] = mapped_column(
        String(16), default="normal", nullable=False
    )  # low/normal/high
    custom_instructions: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # свободные инструкции
    alias: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )  # как обращаться к пользователю
    adaptive_mode_enabled: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )  # авто-адаптация на основе обратной связи
    base_snapshot_json: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # JSON-снапшот базовых настроек для сброса
    # ── Style‑by‑example ──
    style_profile: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # JSON-профиль стиля пользователя
    style_profile_updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Метрики
    total_interactions: Mapped[int] = mapped_column(Integer, default=0)
    total_corrections: Mapped[int] = mapped_column(Integer, default=0)
    last_correction_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class SoulSnapshot(Base):
    """Снапшот tier-2 soul-блоков для версионирования промптов."""

    __tablename__ = "soul_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    version: Mapped[str] = mapped_column(String(32), nullable=False)  # semver "1.0.0"
    snapshot_type: Mapped[str] = mapped_column(
        String(16), nullable=False, default="auto"
    )  # manual / auto / freeze
    blocks_json: Mapped[dict] = mapped_column(JSON, nullable=False)  # все tier-2 блоки
    diff_from_previous: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    approved_by: Mapped[str] = mapped_column(
        String(64), nullable=False, default="system"
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Trajectory(Base):
    """Recorded assistant turn for learning, debugging, and skill extraction."""

    __tablename__ = "trajectories"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    request_text: Mapped[str] = mapped_column(Text, nullable=False)
    route_mode: Mapped[str | None] = mapped_column(
        String(32), nullable=True, index=True
    )
    intent_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    actions_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    used_skills_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    memory_ids_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    response_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    success: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), index=True
    )


class Skill(Base):
    """Prompt-level procedural memory. V1 skills are hints, not executable code.

    V2 additions (SkillOpt-inspired):
    - version: semver versioning for skill evolution tracking
    - edit_history_json: bounded edit history (append/insert/replace/delete)
    - rejected_edits_json: rejected-edit buffer for negative feedback
    - validation_score: last validation gate score (0.0-1.0)
    - best_body: best-performing body snapshot (like best_skill.md)
    """

    __tablename__ = "skills"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    trigger_patterns_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    review_status: Mapped[str] = mapped_column(
        String(16), default="approved", index=True
    )  # approved | pending | rejected
    success_count: Mapped[int] = mapped_column(Integer, default=0)
    failure_count: Mapped[int] = mapped_column(Integer, default=0)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    # ── V2: SkillOpt-inspired fields ──
    version: Mapped[str] = mapped_column(
        String(32), default="1.0.0", nullable=False
    )  # semver version for skill evolution
    edit_history_json: Mapped[list | None] = mapped_column(
        JSON, nullable=True
    )  # [{op, target, content, timestamp, score_before, score_after}]
    rejected_edits_json: Mapped[list | None] = mapped_column(
        JSON, nullable=True
    )  # [{op, target, content, reason, timestamp}] — negative feedback buffer
    validation_score: Mapped[float | None] = mapped_column(
        nullable=True
    )  # last validation gate score (0.0-1.0), None = never validated
    best_body: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # best-performing body snapshot (auto-saved on validation success)
    last_compressed_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, default=None
    )  # timestamp of last skill body compression


class SkillUsage(Base):
    """Skill application telemetry linked to a trajectory."""

    __tablename__ = "skill_usages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    skill_id: Mapped[int] = mapped_column(
        ForeignKey("skills.id", ondelete="CASCADE"), index=True
    )
    trajectory_id: Mapped[int | None] = mapped_column(
        ForeignKey("trajectories.id", ondelete="SET NULL"), nullable=True, index=True
    )
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), index=True
    )
