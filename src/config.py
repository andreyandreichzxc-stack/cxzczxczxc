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


class LLMDefaults:
    # Имена моделей на май 2026 — менять при выходе новых
    OPENAI_CHAT_LIGHT = "gpt-5-mini"
    OPENAI_CHAT_HEAVY = "gpt-5.5"
    OPENAI_EMBED = "text-embedding-3-small"

    GEMINI_CHAT_LIGHT = "gemini-3-flash"
    GEMINI_CHAT_HEAVY = "gemini-3.1-pro"
    GEMINI_EMBED = "text-embedding-004"

    MISTRAL_CHAT_LIGHT = "mistral-small-latest"
    MISTRAL_CHAT_HEAVY = "magistral-medium-latest"
    MISTRAL_EMBED = "mistral-embed"
    MISTRAL_STT = "voxtral-mini-transcribe-latest"


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

    @property
    def data_dir(self) -> Path:
        path = PROJECT_ROOT / "data"
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
