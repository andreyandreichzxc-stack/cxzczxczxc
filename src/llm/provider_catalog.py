"""Provider Catalog — recommended LLM + STT providers with models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProviderInfo:
    name: str  # "openai", "anthropic", "groq", "deepgram"
    display: str  # "OpenAI", "Anthropic", "Groq", "Deepgram"
    category: str  # "llm" | "stt" | "tts" | "vision"
    tier: str  # "free" | "paid" | "custom" | "local"
    key_prefix: str  # "sk-" | "sk-ant-" | "gsk_" | ""
    default_endpoint: str | None  # None = use SDK default
    description: str  # one-liner (required, no default)
    models: list[str] = field(default_factory=list)  # removed hardcoded models
    supports_vision: bool = False  # supports image/vision
    supports_embeddings: bool = False  # supports embeddings
    ai_markers: tuple[
        str, ...
    ] = ()  # русскоязычные маркеры-фразы, характерные для провайдера


# ── Free LLM ──────────────────────────────────────────────────────────

GROQ = ProviderInfo(
    name="groq",
    display="Groq",
    category="llm",
    tier="free",
    key_prefix="gsk_",
    default_endpoint=None,
    description="Бесплатные токены, быстрый вывод, OpenAI-совместимый API",
    ai_markers=("в конечном счёте", "я бы предложил"),
)

GEMINI = ProviderInfo(
    name="gemini",
    display="Google Gemini",
    category="llm",
    tier="free",
    key_prefix="AIza",
    default_endpoint=None,
    description="Бесплатный тир, мультимодальный, Google SDK",
    supports_vision=True,
    supports_embeddings=True,
    ai_markers=(
        "вот что я нашел",
        "давайте разберем",
        "вот основные моменты",
        "рад был помочь",
    ),
)

CLOUDFLARE = ProviderInfo(
    name="cloudflare",
    display="Cloudflare Workers AI",
    category="llm",
    tier="free",
    key_prefix="",
    default_endpoint=None,
    description="Бесплатные Workers AI, Cloudflare-специфичный API",
    supports_vision=True,
    supports_embeddings=True,
)

# ── Paid LLM ──────────────────────────────────────────────────────────

OPENAI = ProviderInfo(
    name="openai",
    display="OpenAI",
    category="llm",
    tier="paid",
    key_prefix="sk-",
    default_endpoint=None,
    description="Лучшее качество, дорогой, стандартный API",
    supports_vision=True,
    ai_markers=(
        "я бы посоветовал",
        "я бы рекомендовал",
        "важно подчеркнуть",
        "хочу обратить внимание",
        "позвольте заметить",
        "не могу не отметить",
        "следует упомянуть",
    ),
)

ANTHROPIC = ProviderInfo(
    name="anthropic",
    display="Anthropic",
    category="llm",
    tier="paid",
    key_prefix="sk-ant-",
    default_endpoint=None,
    description="Claude, Messages API, лучший для длинных текстов",
    supports_vision=True,
    ai_markers=(
        "я стремлюсь",
        "я стараюсь",
        "позвольте уточнить",
        "я бы с радостью",
        "не стесняйтесь обращаться",
        "чем могу быть полезен",
    ),
)

DEEPSEEK = ProviderInfo(
    name="deepseek",
    display="DeepSeek",
    category="llm",
    tier="paid",
    key_prefix="sk-",
    default_endpoint="https://api.deepseek.com/v1",
    description="Дешёвый, качественный, OpenAI-совместимый",
    supports_embeddings=True,
    ai_markers=("наконец", "в конечном счете", "резюмируя", "в итоге"),
)

MISTRAL = ProviderInfo(
    name="mistral",
    display="Mistral AI",
    category="llm",
    tier="paid",
    key_prefix="",
    default_endpoint=None,
    description="Французский LLM, хорошее соотношение цена/качество",
    supports_embeddings=True,
    ai_markers=("я полагаю", "по всей видимости", "в сущности"),
)

GROK = ProviderInfo(
    name="grok",
    display="Grok (xAI)",
    category="llm",
    tier="paid",
    key_prefix="xai-",
    default_endpoint="https://api.x.ai/v1",
    description="xAI Grok, OpenAI-совместимый API",
    supports_vision=True,
    ai_markers=("по-моему", "как по мне", "на мой взгляд", "жду твоего мнения"),
)

MIMO = ProviderInfo(
    name="mimo",
    display="MiMo (Xiaomi)",
    category="llm",
    tier="paid",
    key_prefix="",
    default_endpoint="https://api.xiaomimimo.com/v1",
    description="Xiaomi MiMo, OpenAI-совместимый, мультимодальный. Региональные endpoint'ы: EU, US, Asia.",
    supports_vision=True,
    supports_embeddings=True,
    ai_markers=("позвольте заметить", "стоит отметить", "обратите внимание"),
)

# ── Custom / Local ────────────────────────────────────────────────────

CUSTOM_OPENAI = ProviderInfo(
    name="openai-compatible",
    display="OpenAI-совместимый",
    category="llm",
    tier="custom",
    key_prefix="",
    default_endpoint=None,  # user provides
    description="Любой OpenAI-совместимый endpoint. Нужен URL + модель.",
    supports_vision=True,  # может поддерживать vision если endpoint позволяет
    ai_markers=(),  # пользователь сам добавит
)

LOCAL = ProviderInfo(
    name="local",
    display="Локальный (llama.cpp/vLLM)",
    category="llm",
    tier="local",
    key_prefix="not-needed",
    default_endpoint=None,
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
    description="Локальная транскрипция. Не нужен ключ. Модель small/medium/large.",
)

WHISPER_OPENAI = ProviderInfo(
    name="whisper-openai",
    display="OpenAI Whisper",
    category="stt",
    tier="paid",
    key_prefix="sk-",
    default_endpoint=None,
    description="OpenAI Whisper API. Платно за минуту.",
)

DEEPGRAM = ProviderInfo(
    name="deepgram",
    display="Deepgram",
    category="stt",
    tier="paid",
    key_prefix="",
    default_endpoint=None,
    description="Лучшее качество STT. Платно за минуту.",
)

ASSEMBLYAI = ProviderInfo(
    name="assemblyai",
    display="AssemblyAI",
    category="stt",
    tier="paid",
    key_prefix="",
    default_endpoint=None,
    description="Качественная транскрипция. Платно.",
)

# ── TTS ───────────────────────────────────────────────────────────────

OPENAI_TTS = ProviderInfo(
    name="openai-tts",
    display="OpenAI TTS",
    category="tts",
    tier="paid",
    key_prefix="sk-",
    default_endpoint="https://api.openai.com/v1",
    description="OpenAI синтез речи. 6 голосов.",
)

MIMO_TTS = ProviderInfo(
    name="mimo-tts",
    display="MiMo TTS",
    category="tts",
    tier="paid",
    key_prefix="",
    default_endpoint="https://api.xiaomimimo.com/v1",
    description="Xiaomi MiMo синтез речи с клонированием голоса.",
)

MISTRAL_TTS = ProviderInfo(
    name="mistral-tts",
    display="Mistral TTS",
    category="tts",
    tier="paid",
    key_prefix="",
    default_endpoint="https://api.mistral.ai/v1",
    description="Mistral синтез речи.",
)

# ── Catalogs for UI ───────────────────────────────────────────────────

LLM_PROVIDERS = [
    GROQ,
    GEMINI,
    CLOUDFLARE,
    OPENAI,
    ANTHROPIC,
    DEEPSEEK,
    GROK,
    MIMO,
    MISTRAL,
    CUSTOM_OPENAI,
    LOCAL,
]
STT_PROVIDERS = [WHISPER_LOCAL, WHISPER_OPENAI, DEEPGRAM, ASSEMBLYAI]

TTS_PROVIDERS = [
    OPENAI_TTS,
    MIMO_TTS,
    MISTRAL_TTS,
]


_ALL_PROVIDERS: list[ProviderInfo] = LLM_PROVIDERS + STT_PROVIDERS + TTS_PROVIDERS


def get_provider(name: str) -> ProviderInfo | None:
    for p in _ALL_PROVIDERS:
        if p.name == name:
            return p
    return None


def get_providers_by_category(category: str) -> list[ProviderInfo]:
    return [p for p in _ALL_PROVIDERS if p.category == category]


def get_providers_by_tier(tier: str) -> list[ProviderInfo]:
    return [p for p in _ALL_PROVIDERS if p.tier == tier]
