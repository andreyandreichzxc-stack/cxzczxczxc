"""Pre-registered gating checks for common dependencies."""

import shutil

from src.config import settings
from src.core.infra.gating import gates


def register_default_gates() -> None:
    """Register all default dependency checks."""
    # Check ffmpeg (for voice messages)
    gates.register(
        "ffmpeg",
        check=lambda: shutil.which("ffmpeg") is not None,
        fallback=None,
        description="ffmpeg (voice messages)",
        install_hint="apt install ffmpeg  # Linux\nbrew install ffmpeg  # macOS\nwinget install ffmpeg  # Windows",
    )

    # Check Qdrant (for semantic search)
    gates.register(
        "qdrant",
        check=lambda: _check_qdrant_available(),
        fallback="fts5_only",
        description="Qdrant (semantic search)",
        install_hint="Qdrant embedded — no install needed. Check data/qdrant/ permissions.",
    )

    # Check faster-whisper (for local transcription)
    gates.register(
        "faster_whisper",
        check=lambda: _check_import("faster_whisper"),
        fallback="openai_whisper_api",
        description="faster-whisper (local transcription)",
        install_hint="pip install faster-whisper",
    )

    # Check pyyaml (for skill YAML frontmatter)
    gates.register(
        "pyyaml",
        check=lambda: _check_import("yaml"),
        fallback="json_only",
        description="PyYAML (skill frontmatter)",
        install_hint="pip install pyyaml",
    )

    # Check bs4 (for web fetching)
    gates.register(
        "beautifulsoup",
        check=lambda: _check_import("bs4"),
        fallback=None,
        description="BeautifulSoup4 (web fetching)",
        install_hint="pip install beautifulsoup4",
    )

    # Check psutil (for system status)
    gates.register(
        "psutil",
        check=lambda: _check_import("psutil"),
        fallback=None,
        description="psutil (system monitoring)",
        install_hint="pip install psutil",
    )

    # Check httpx (for Avito stealth / HTTP/2)
    gates.register(
        "httpx",
        check=lambda: _check_import("httpx"),
        fallback="requests_only",
        description="httpx (HTTP/2, Avito stealth)",
        install_hint="pip install httpx",
    )

    # Check playwright (for Avito stealth browser fallback)
    gates.register(
        "playwright",
        check=lambda: _check_import("playwright"),
        fallback=None,
        description="Playwright (Avito browser stealth)",
        install_hint="pip install playwright && playwright install chromium",
    )


def _check_import(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def _check_qdrant_available() -> bool:
    """Check that embedded Qdrant storage directory is usable."""
    try:
        (settings.data_dir / "qdrant").mkdir(parents=True, exist_ok=True)
        return True
    except OSError:
        return False
