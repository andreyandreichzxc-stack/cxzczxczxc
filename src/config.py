from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_telethon_proxy(proxy_url: str) -> tuple | None:
    if not proxy_url:
        return None
    parsed = urlparse(proxy_url)
    scheme = parsed.scheme or "socks5"
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (1080 if scheme == "socks5" else 8080)
    if parsed.username and parsed.password:
        return (scheme, host, port, True, parsed.username, parsed.password)
    return (scheme, host, port)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    bot_token: str = Field(..., description="Токен control-бота из @BotFather")
    owner_telegram_id: int = Field(
        ..., description="Telegram user_id единственного владельца"
    )
    encryption_key: str = Field(..., description="Fernet-ключ (base64)")
    database_url: str = Field("sqlite+aiosqlite:///data/app.db")
    proxy_url: str = Field(
        "",
        description="Прокси для aiogram и Telethon (socks5://ip:port или http://ip:port)",
    )
    disable_local_transcription: bool = Field(
        False, description="Полностью отключить faster-whisper (для VDS с малым RAM)"
    )

    # --- Интервалы фоновых циклов (секунды) ---
    global_style_interval_sec: int = Field(
        12 * 3600, description="Интервал обновления глобального стиля"
    )
    instruction_optimizer_interval_sec: int = Field(
        24 * 3600, description="Интервал цикла оптимизатора инструкций"
    )
    skill_optimizer_interval_sec: int = Field(
        24 * 3600, description="Интервал цикла оптимизатора навыков"
    )
    weekly_digest_check_sec: int = Field(
        3600, description="Проверка еженедельного дайджеста"
    )
    weekly_summary_check_sec: int = Field(
        3600, description="Проверка еженедельного саммари"
    )
    conflict_predictor_interval_sec: int = Field(
        3 * 3600, description="Интервал предсказания конфликтов"
    )
    follow_up_interval_sec: int = Field(
        4 * 3600, description="Интервал follow-up напоминаний"
    )
    memory_clusterer_interval_sec: int = Field(
        600, description="Интервал кластеризации памяти"
    )
    temporal_migration_interval_sec: int = Field(
        3600, description="Интервал миграции временных слоёв"
    )
    habit_tracker_interval_sec: int = Field(
        3600, description="Интервал трекера привычек"
    )
    memory_check_interval_sec: int = Field(600, description="Интервал проверки памяти")
    auto_sync_interval_sec: int = Field(
        3600, description="Интервал авто-синхронизации контактов"
    )
    auto_sync_fallback_sec: int = Field(
        300, description="Fallback-интервал при ошибке синхронизации"
    )
    digest_check_sec: int = Field(60, description="Интервал проверки дайджеста")
    news_check_sec: int = Field(60, description="Интервал проверки новостей")
    avito_check_sec: int = Field(1800, description="Интервал проверки Авито (сек)")
    avito_default_city: str = Field(
        "moskva", description="Город по умолчанию для Авито"
    )
    sleep_tracker_check_sec: int = Field(900, description="Интервал трекера сна")
    sleep_tracker_fallback_sec: int = Field(600, description="Fallback трекера сна")
    memory_patterns_interval_sec: int = Field(
        600, description="Интервал поиска паттернов памяти"
    )
    proactive_briefing_check_sec: int = Field(
        300, description="Интервал проактивного брифинга"
    )
    conflict_resolver_interval_sec: int = Field(
        600, description="Интервал разрешения конфликтов"
    )
    knowledge_distiller_interval_sec: int = Field(
        600, description="Интервал дистилляции знаний"
    )

    # --- Cloudflare Workers AI ---
    openai_base_url: str = Field(
        "",
        description="Кастомный base_url для OpenAI-совместимых API (например, https://macky1.icu/v1). Оставь пустым для стандартного OpenAI.",
    )

    cloudflare_account_id: str = Field(
        "", description="Cloudflare Account ID (из URL дашборда)"
    )

    context7_api_key: str = Field(
        "",
        description="Context7 API key for documentation search (https://context7.com)",
    )

    embedding_dim: int = Field(
        1536,
        description="Размерность эмбеддингов (OpenAI text-embedding-3-small: 1536, BGE-M3: 1024, Gemini text-embedding-004: 768)",
    )

    # Capability toggles
    embedding_enabled: bool = Field(True, description="Enable embedding models")
    vision_enabled: bool = Field(False, description="Enable vision/image analysis")
    audio_enabled: bool = Field(True, description="Enable STT/speech-to-text")
    tts_enabled: bool = Field(False, description="Enable TTS/text-to-speech")
    auto_select_model: bool = Field(
        False, description="Auto-select best model per task"
    )

    # ── Фото-кэш ──
    photo_cache_ttl_sec: int = Field(300, description="TTL кэша фотографий (секунды)")

    # ── Streaming ──
    streaming_enabled: bool = Field(True, description="Включить streaming-ответы")
    streaming_edit_interval: float = Field(
        0.3, description="Интервал обновления streaming (сек)"
    )
    streaming_cursor: str = Field(" 🦊", description="Курсор при streaming")

    memory_warmup_idle_timeout_sec: int = Field(
        86400, description="Таймаут простоя для сброса warmup-счётчика (24 часа)"
    )
    memory_warmup_max_contacts: int = Field(
        10,
        description="Макс контактов при штатной экстракции (в warmup — все контакты)",
    )

    # Авто-пересборка профиля каждые N новых личных фактов (0 = только вручную)
    persona_trigger_every_n_facts: int = Field(
        default=15,
        description="Trigger persona rebuild every N new personal facts",
    )

    # --- Telegram API credentials (опционально — нужны только для userbot-режима) ---
    api_id: int | None = Field(
        default=None, description="Telegram API ID from https://my.telegram.org"
    )
    api_hash: str | None = Field(
        default=None,
        description="Telegram API hash from https://my.telegram.org",
    )

    disk_critical_mb: int = Field(
        100, description="Критический порог свободного места (MB)"
    )
    disk_warning_mb: int = Field(
        500, description="Предупредительный порог свободного места (MB)"
    )
    disk_monitor_interval_sec: int = Field(600, description="Интервал проверки диска")

    # Memory
    max_recall_cache_size: int = Field(
        1000, description="Максимальный размер кэша recall"
    )
    memory_consolidation_interval_sec: int = Field(
        21600, description="Интервал консолидации памяти (6 часов)"
    )

    # ── Recall defaults ──
    recall_default_limit: int = Field(8, description="Default recall limit")
    recall_max_limit: int = Field(20, description="Max recall limit")
    recall_semantic_threshold: float = Field(
        0.55, description="Min cosine similarity for semantic search"
    )
    recall_rrf_k: int = Field(60, description="RRF k-parameter")
    recall_mmr_lambda: float = Field(
        0.7, description="MMR lambda (relevance vs diversity)"
    )

    # ── Ebbinghaus retention scoring ──
    ebbinghaus_decay_base: float = Field(
        0.07, description="Base decay rate for Ebbinghaus retention (no recall boost)"
    )
    ebbinghaus_access_weight: float = Field(
        0.5, description="Weight of access count in retention boost"
    )
    auto_forget_threshold: float = Field(
        0.15,
        description="Retention score below which facts are candidates for forgetting",
    )
    auto_forget_enabled: bool = Field(
        True, description="Enable automatic forgetting of low-retention facts"
    )

    # ── Limits & timeouts ──
    max_message_length: int = Field(4096, description="Telegram max message length")
    safe_message_length: int = Field(4000, description="Buffer before Telegram limit")
    max_voice_queue_size: int = Field(20, description="Max voice messages in queue")
    voice_queue_timeout: float = Field(
        10.0, description="Seconds before dropping voice msg"
    )

    # ── Caching ──
    context_cache_max_size: int = Field(2000, description="Max context cache entries")
    contact_digest_cache_max: int = Field(
        500, description="Max contact digest cache entries"
    )
    recall_cache_max_size: int = Field(1000, description="Max recall cache entries")
    recall_cache_result_ttl: float = Field(
        30.0, description="Recall cache TTL with facts (sec)"
    )
    recall_cache_empty_ttl: float = Field(
        60.0, description="Recall cache TTL without facts (sec)"
    )

    # Humanizer
    humanizer_deep_min_length: int = Field(
        100, description="Минимальная длина текста для deep humanizer"
    )
    humanizer_deep_min_score: float = Field(
        0.3, description="Минимальный AI-score для deep humanizer"
    )

    # Tool loop
    max_tool_iterations: int = Field(
        5, description="Макс. итераций tool-calling в Maestro"
    )

    # ── Skill Evolution (SkillOpt-inspired) ──
    skill_edit_budget: int = Field(
        3,
        description="Макс. количество bounded edits за одну итерацию (textual learning rate)",
    )
    skill_optimizer_model: str = Field(
        "",
        description="Модель для оптимизации навыков (пустая = использовать heavy). "
        "Формат: 'provider/model' или 'model_name'",
    )
    skill_target_model: str = Field(
        "",
        description="Целевая модель для исполнения навыков (пустая = использовать light). "
        "Формат: 'provider/model' или 'model_name'",
    )
    skill_validation_enabled: bool = Field(
        True,
        description="Включить validation gate для обновлений навыков",
    )
    skill_auto_edit_enabled: bool = Field(
        True,
        description="Разрешить автоматические bounded edits вместо полной замены навыков",
    )
    skill_edit_cooldown_sec: int = Field(
        60,
        description="Минимальный интервал между edits одного навыка (rate limiting)",
    )
    skill_auto_evolve_interval_sec: int = Field(
        21600,  # 6 hours
        description="Интервал auto-evolution цикла (по умолчанию 6 часов)",
    )
    skill_auto_evolve_min_failures: int = Field(
        3,
        description="Минимальное количество провалов для запуска auto-evolution навыка",
    )

    # Pending
    pending_ttl_sec: int = Field(
        300, description="TTL ожидающих подтверждений (5 минут)"
    )

    # Auto-reply
    auto_reply_global_limit_per_hour: int = Field(
        100, description="Глобальный лимит авто-ответов в час"
    )

    # Context
    context_max_turns: int = Field(50, description="Макс. витков диалога перед сжатием")

    # ── Skill seeding ──
    skill_seed_on_startup: bool = Field(
        True, description="Auto-seed skills from skills/*/SKILL.md on startup"
    )

    # Agent/task-specific model overrides (из .env)
    maestro_model: str = Field("", description="Model override for maestro agent")
    draft_model: str = Field("", description="Model override for draft agent")
    memory_model: str = Field("", description="Model override for memory agent")
    search_model: str = Field("", description="Model override for search agent")
    humanize_model: str = Field("", description="Model override for humanize agent")
    classify_model: str = Field("", description="Model override for classify agent")
    summarize_model: str = Field("", description="Model override for summarize agent")
    skills_model: str = Field("", description="Model override for skills agent")
    background_model: str = Field("", description="Model override for background tasks")
    vision_model: str = Field("", description="Model override for vision tasks")

    @property
    def data_dir(self) -> Path:
        path = PROJECT_ROOT / "data"
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
