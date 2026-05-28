"""Provider Catalog — recommended LLM + STT providers with models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderInfo:
    name: str  # "openai", "anthropic", "groq", "deepgram"
    display: str  # "OpenAI", "Anthropic", "Groq", "Deepgram"
    category: str  # "llm" | "stt" | "tts" | "vision"
    tier: str  # "free" | "paid" | "custom" | "local"
    key_prefix: str  # "sk-" | "sk-ant-" | "gsk_" | ""
    default_endpoint: str | None  # None = use SDK default
    models: list[str]  # ["gpt-4o", "gpt-4o-mini"]
    description: str  # one-liner


# ── Free LLM ──────────────────────────────────────────────────────────

GROQ = ProviderInfo(
    name="groq",
    display="Groq",
    category="llm",
    tier="free",
    key_prefix="gsk_",
    default_endpoint=None,
    models=["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"],
    description="Бесплатные токены, быстрый вывод, OpenAI-совместимый API",
)

GEMINI = ProviderInfo(
    name="gemini",
    display="Google Gemini",
    category="llm",
    tier="free",
    key_prefix="AIza",
    default_endpoint=None,
    models=[
        "gemini-3-flash",
        "gemini-3.1-pro",
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ],
    description="Бесплатный тир, мультимодальный, Google SDK",
)

CLOUDFLARE = ProviderInfo(
    name="cloudflare",
    display="Cloudflare Workers AI",
    category="llm",
    tier="free",
    key_prefix="",
    default_endpoint=None,
    models=[
        "@cf/qwen/qwen3-30b-a3b-fp8",
        "@cf/moonshotai/kimi-k2.6",
        "@cf/meta/llama-3.1-8b-instruct",
        "@cf/mistral/mistral-7b-instruct",
    ],
    description="Бесплатные Workers AI, Cloudflare-специфичный API",
)

# ── Paid LLM ──────────────────────────────────────────────────────────

OPENAI = ProviderInfo(
    name="openai",
    display="OpenAI",
    category="llm",
    tier="paid",
    key_prefix="sk-",
    default_endpoint=None,
    models=[
        "gpt-5-mini",
        "gpt-5.5",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "o3-mini",
        "o4-mini",
    ],
    description="Лучшее качество, дорогой, стандартный API",
)

ANTHROPIC = ProviderInfo(
    name="anthropic",
    display="Anthropic",
    category="llm",
    tier="paid",
    key_prefix="sk-ant-",
    default_endpoint=None,
    models=[
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
    ],
    description="Claude, Messages API, лучший для длинных текстов",
)

DEEPSEEK = ProviderInfo(
    name="deepseek",
    display="DeepSeek",
    category="llm",
    tier="paid",
    key_prefix="sk-",
    default_endpoint="https://api.deepseek.com/v1",
    models=["deepseek-chat", "deepseek-reasoner", "deepseek-embedding"],
    description="Дешёвый, качественный, OpenAI-совместимый",
)

MISTRAL = ProviderInfo(
    name="mistral",
    display="Mistral AI",
    category="llm",
    tier="paid",
    key_prefix="",
    default_endpoint=None,
    models=[
        "mistral-small-latest",
        "mistral-medium-latest",
        "mistral-large-latest",
        "codestral-latest",
    ],
    description="Французский LLM, хорошее соотношение цена/качество",
)

# ── Custom / Local ────────────────────────────────────────────────────

CUSTOM_OPENAI = ProviderInfo(
    name="openai-compatible",
    display="OpenAI-совместимый",
    category="llm",
    tier="custom",
    key_prefix="",
    default_endpoint=None,  # user provides
    models=[],  # user types model name
    description="Любой OpenAI-совместимый endpoint. Нужен URL + модель.",
)

LOCAL = ProviderInfo(
    name="local",
    display="Локальный (llama.cpp/vLLM)",
    category="llm",
    tier="local",
    key_prefix="not-needed",
    default_endpoint=None,
    models=[],  # user types
    description="Локальный сервер. Ключ не нужен. Нужен URL + модель.",
)

# ── STT ───────────────────────────────────────────────────────────────

WHISPER_LOCAL = ProviderInfo(
    name="faster-whisper",
    display="faster-whisper (локальный)",
    category="stt",
    tier="local",
    key_prefix="not-needed",
    default_endpoint=None,
    models=["tiny", "small", "medium", "large-v3"],
    description="Локальная транскрипция. Не нужен ключ. Модель small/medium/large.",
)

WHISPER_OPENAI = ProviderInfo(
    name="whisper-openai",
    display="OpenAI Whisper",
    category="stt",
    tier="paid",
    key_prefix="sk-",
    default_endpoint=None,
    models=["whisper-1"],
    description="OpenAI Whisper API. Платно за минуту.",
)

DEEPGRAM = ProviderInfo(
    name="deepgram",
    display="Deepgram",
    category="stt",
    tier="paid",
    key_prefix="",
    default_endpoint=None,
    models=["nova-2", "nova-3", "whisper"],
    description="Лучшее качество STT. Платно за минуту.",
)

ASSEMBLYAI = ProviderInfo(
    name="assemblyai",
    display="AssemblyAI",
    category="stt",
    tier="paid",
    key_prefix="",
    default_endpoint=None,
    models=["best", "nano"],
    description="Качественная транскрипция. Платно.",
)

# ── Catalogs for UI ───────────────────────────────────────────────────

LLM_PROVIDERS = [
    GROQ,
    GEMINI,
    CLOUDFLARE,
    OPENAI,
    ANTHROPIC,
    DEEPSEEK,
    MISTRAL,
    CUSTOM_OPENAI,
    LOCAL,
]
STT_PROVIDERS = [WHISPER_LOCAL, WHISPER_OPENAI, DEEPGRAM, ASSEMBLYAI]


def get_provider(name: str) -> ProviderInfo | None:
    for p in LLM_PROVIDERS + STT_PROVIDERS:
        if p.name == name:
            return p
    return None


def get_providers_by_category(category: str) -> list[ProviderInfo]:
    all_providers = LLM_PROVIDERS + STT_PROVIDERS
    return [p for p in all_providers if p.category == category]


def get_providers_by_tier(tier: str) -> list[ProviderInfo]:
    all_providers = LLM_PROVIDERS + STT_PROVIDERS
    return [p for p in all_providers if p.tier == tier]
