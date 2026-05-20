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
        return (scheme, host, port, parsed.username, parsed.password)
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
        "magistral-medium-latest", description="Mistral тяжёлая чат-модель"
    )
    mistral_embed_model: str = Field(
        "mistral-embed", description="Mistral модель эмбеддингов"
    )
    mistral_stt_model: str = Field(
        "voxtral-mini-transcribe-latest", description="Mistral STT модель"
    )

    disk_critical_mb: int = Field(
        100, description="Критический порог свободного места (MB)"
    )
    disk_warning_mb: int = Field(
        500, description="Предупредительный порог свободного места (MB)"
    )
    disk_monitor_interval_sec: int = Field(600, description="Интервал проверки диска")

    @property
    def data_dir(self) -> Path:
        path = PROJECT_ROOT / "data"
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
