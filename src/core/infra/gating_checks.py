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
        fallback=None,  # voice transcription won't work
        description="ffmpeg (voice messages)",
    )

    # Check Qdrant (for semantic search)
    gates.register(
        "qdrant",
        check=lambda: _check_qdrant_available(),
        fallback="fts5_only",  # FTS5 keyword search still works
        description="Qdrant (embedded, semantic search)",
    )

    # Check faster-whisper (for local transcription)
    gates.register(
        "faster_whisper",
        check=lambda: _check_import("faster_whisper"),
        fallback="openai_whisper_api",
        description="faster-whisper (local transcription)",
    )

    # Check pyyaml (for skill YAML frontmatter)
    gates.register(
        "pyyaml",
        check=lambda: _check_import("yaml"),
        fallback="json_only",
        description="PyYAML (skill frontmatter)",
    )

    # Check bs4 (for web fetching)
    gates.register(
        "beautifulsoup",
        check=lambda: _check_import("bs4"),
        fallback=None,  # web tools won't work
        description="BeautifulSoup4 (web fetching)",
    )

    # Check psutil (for system status)
    gates.register(
        "psutil",
        check=lambda: _check_import("psutil"),
        fallback=None,
        description="psutil (system monitoring)",
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
