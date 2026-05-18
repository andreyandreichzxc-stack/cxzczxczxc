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

    MISTRAL_CHAT_LIGHT = "mistral-small-4-latest"
    MISTRAL_CHAT_VISION = "mistral-medium-3-5-latest"
    MISTRAL_CHAT_HEAVY = "magistral-medium-1.2-latest"
    MISTRAL_EMBED = "mistral-embed"
    MISTRAL_STT = "voxtral-mini-transcribe-latest"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    bot_token: str = Field(..., description="Токен control-бота из @BotFather")
    owner_telegram_id: int = Field(..., description="Telegram user_id единственного владельца")
    encryption_key: str = Field(..., description="Fernet-ключ (base64)")
    database_url: str = Field("sqlite+aiosqlite:///data/app.db")
    proxy_url: str = Field("", description="Прокси для aiogram и Telethon (socks5://ip:port или http://ip:port)")

    @property
    def data_dir(self) -> Path:
        path = PROJECT_ROOT / "data"
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
