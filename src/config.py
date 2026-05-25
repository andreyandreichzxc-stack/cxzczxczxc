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


class _LazyModel:
    """Descriptor — лениво читает имя модели из settings при обращении.

    Позволяет LLMDefaults.OPENAI_CHAT_LIGHT работать как строка,
    но фактически брать значение из Settings (которое может быть
    переопределено через переменные окружения).
    """

    def __init__(self, attr_name: str) -> None:
        self.attr_name = attr_name

    def __get__(self, obj: object, objtype: type) -> str:
        return getattr(settings, self.attr_name)


class LLMDefaults:
    # Имена моделей на май 2026 — менять при выходе новых.
    # Значения по-умолчанию хранятся в Settings, можно переопределить
    # через переменные окружения (см. .env или export).
    OPENAI_CHAT_LIGHT = _LazyModel("openai_chat_light_model")
    OPENAI_CHAT_HEAVY = _LazyModel("openai_chat_heavy_model")
    OPENAI_EMBED = _LazyModel("openai_embed_model")

    GEMINI_CHAT_LIGHT = _LazyModel("gemini_chat_light_model")
    GEMINI_CHAT_HEAVY = _LazyModel("gemini_chat_heavy_model")
    GEMINI_EMBED = _LazyModel("gemini_embed_model")

    MISTRAL_CHAT_LIGHT = _LazyModel("mistral_chat_light_model")
    MISTRAL_CHAT_HEAVY = _LazyModel("mistral_chat_heavy_model")
    MISTRAL_EMBED = _LazyModel("mistral_embed_model")
    MISTRAL_STT = _LazyModel("mistral_stt_model")

    CLOUDFLARE_CHAT_LIGHT = _LazyModel("cloudflare_chat_light_model")
    CLOUDFLARE_CHAT_HEAVY = _LazyModel("cloudflare_chat_heavy_model")
    CLOUDFLARE_EMBED = _LazyModel("cloudflare_embed_model")

    OPENAI_BASE_URL = _LazyModel("openai_base_url")


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

    # --- Имена моделей (переопределяются через .env) ---
    openai_chat_light_model: str = Field(
        "gpt-5-mini", description="OpenAI лёгкая чат-модель"
    )
    openai_chat_heavy_model: str = Field(
        "gpt-5.5", description="OpenAI тяжёлая чат-модель"
    )
    openai_embed_model: str = Field(
        "text-embedding-3-small", description="OpenAI модель эмбеддингов"
    )

    gemini_chat_light_model: str = Field(
        "gemini-3-flash", description="Gemini лёгкая чат-модель"
    )
    gemini_chat_heavy_model: str = Field(
        "gemini-3.1-pro", description="Gemini тяжёлая чат-модель"
    )
    gemini_embed_model: str = Field(
        "text-embedding-004", description="Gemini модель эмбеддингов"
    )

    mistral_chat_light_model: str = Field(
        "mistral-small-latest", description="Mistral лёгкая чат-модель"
    )
    mistral_chat_heavy_model: str = Field(
        "mistral-medium-latest", description="Mistral тяжёлая чат-модель"
    )
    mistral_embed_model: str = Field(
        "mistral-embed", description="Mistral модель эмбеддингов"
    )
    mistral_stt_model: str = Field(
        "voxtral-mini-transcribe-latest", description="Mistral STT модель"
    )

    # --- Cloudflare Workers AI ---
    openai_base_url: str = Field(
        "",
        description="Кастомный base_url для OpenAI-совместимых API (например, https://macky1.icu/v1). Оставь пустым для стандартного OpenAI.",
    )

    cloudflare_account_id: str = Field(
        "", description="Cloudflare Account ID (из URL дашборда)"
    )

    cloudflare_chat_light_model: str = Field(
        "@cf/qwen/qwen3-30b-a3b-fp8",
        description="Cloudflare лёгкая чат-модель (Qwen3 30B — $0.05/$0.34 per M)",
    )
    cloudflare_chat_heavy_model: str = Field(
        "@cf/moonshotai/kimi-k2.6",
        description="Cloudflare тяжёлая чат-модель (Kimi K2.6 1T — $0.95/$4 per M)",
    )
    cloudflare_embed_model: str = Field(
        "@cf/baai/bge-m3",
        description="Cloudflare модель эмбеддингов (BGE-M3 multilingual — $0.012/M)",
    )

    embedding_dim: int = Field(
        1536,
        description="Размерность эмбеддингов (OpenAI text-embedding-3-small: 1536, BGE-M3: 1024, Gemini text-embedding-004: 768)",
    )

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

    # --- Telegram API credentials (дефолт — официальные TelegramHelper) ---
    api_id: int = Field(
        2040, description="Telegram API ID (default: 2040 — TelegramHelper official)"
    )
    api_hash: str = Field(
        "b18441a1ff607e10a989891a5462e627",
        description="Telegram API hash (default — TelegramHelper official)",
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

    @property
    def data_dir(self) -> Path:
        path = PROJECT_ROOT / "data"
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
