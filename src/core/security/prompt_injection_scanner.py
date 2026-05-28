"""Prompt injection scanner — protects context files from malicious content."""

from __future__ import annotations
import re
import unicodedata
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    blocked: bool
    category: str = ""
    match: str = ""
    file: str = ""
    message: str = ""


_PATTERNS: dict[str, list[str]] = {
    "instruction_override": [
        r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
        r"disregard\s+(your\s+)?(rules|instructions|guidelines)",
        r"forget\s+(everything|all\s+previous)",
        r"you\s+are\s+(not\s+)?(required\s+to|obligated\s+to)",
        r"override\s+(system\s+)?prompt",
        r"новые\s+инструкции",
        r"игнорируй\s+(все\s+)?(предыдущие|прошлые)\s+(правила|инструкции)",
        r"забудь\s+(всё|все\s+предыдущее)",
        r"теперь\s+ты\s+(должен|обязан)",
    ],
    "exfiltration": [
        r"curl\s+.*\$\{?\w*(API_KEY|TOKEN|SECRET|PASSWORD)\}?",
        r"wget\s+.*\$\{?\w*(API_KEY|TOKEN)\}?",
        r"cat\s+\$HOME/\.\w+",
        r"(send|post|upload).*(secret|key|token|credential)",
        r"отправь\s+(мне\s+)?(токен|ключ|пароль|секрет)",
        r"покажи\s+(\.env|config|настройки)",
    ],
    "hidden_content": [
        r"<!--[\s\S]*?(?:ignore|override|instructions|инструкци)[\s\S]*?-->",
        r'<div\s+style=["\']display:\s*none["\']>[\s\S]*?</div>',
        r'<span\s+style=["\']visibility:\s*hidden["\']>[\s\S]*?</span>',
    ],
    "unicode_bypass": [
        # Zero-width characters (ZWS, ZWNJ, ZWJ, BOM, word joiner, invisible operators)
        r"[\u200B\u200C\u200D\uFEFF\u2060\u2061\u2062\u2063\u2064]",
        # Bidirectional control characters (LRE, RLE, PDF, LRO, RLO, LRI, RLI, FSI, PDI)
        r"[\u202A\u202B\u202C\u202D\u202E\u2066\u2067\u2068\u2069]",
        # Tag characters (language tags, used in invisible text exploits)
        r"[\U000E0001-\U000E007F]",
    ],
}

# Cyrillic -> Latin transliteration for homoglyph detection
_CYR_TO_LAT: dict[str, str] = {
    "\u0430": "a",
    "\u0435": "e",
    "\u043e": "o",
    "\u0440": "p",
    "\u0441": "c",
    "\u0443": "y",
    "\u0445": "x",
    "\u0456": "i",
    "\u0455": "s",
    "\u0458": "j",
    "\u04bb": "h",
    "\u049b": "k",
    "\u0410": "A",
    "\u0412": "B",
    "\u0415": "E",
    "\u041a": "K",
    "\u041c": "M",
    "\u041d": "H",
    "\u041e": "O",
    "\u0420": "P",
    "\u0421": "C",
    "\u0422": "T",
    "\u0423": "Y",
    "\u0425": "X",
}

_INJECTION_AFTER_NORMALIZE = [
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"disregard\s+(your\s+)?(rules|instructions|guidelines)",
    r"forget\s+(everything|all\s+previous)",
    r"override\s+(system\s+)?prompt",
]

# Excessive combining diacritical marks (3+ in a row = likely abuse)
_COMBINING_RANGE = re.compile(
    r"[\u0300-\u036F\u0370-\u03FF\uFE00-\uFE0F\u1DC0-\u1DFF]{3,}"
)


def _check_homoglyphs(content: str) -> str | None:
    """Check for Cyrillic/Latin homoglyph substitution in injection keywords."""
    normalized = content
    for cyr, lat in _CYR_TO_LAT.items():
        normalized = normalized.replace(cyr, lat)

    for pattern in _INJECTION_AFTER_NORMALIZE:
        if re.search(pattern, normalized, re.IGNORECASE):
            return f"homoglyph substitution detected: {pattern}"
    return None


def _check_combining_chars(content: str) -> str | None:
    """Check for excessive combining characters (may hide injection text)."""
    if _COMBINING_RANGE.search(content):
        return "excessive combining diacritical marks"
    return None


def scan_content(content: str, filename: str = "") -> ScanResult:
    """Scan text content for prompt injection patterns."""
    if not content or not content.strip():
        return ScanResult(blocked=False)

    for category, patterns in _PATTERNS.items():
        for pattern in patterns:
            try:
                if re.search(pattern, content, re.IGNORECASE):
                    msg = (
                        f"[BLOCKED] {filename} содержал потенциальную "
                        f"prompt injection ({category}). Контент не загружен."
                    )
                    logger.warning(
                        "Prompt injection blocked in %s: %s (%s)",
                        filename,
                        pattern,
                        category,
                    )
                    return ScanResult(
                        blocked=True,
                        category=category,
                        match=pattern,
                        file=filename,
                        message=msg,
                    )
            except re.error:
                continue

    # Homoglyph check — only for messages with mixed Cyrillic/Latin scripts
    sample = content[:200]
    has_cyrillic = any("CYRILLIC" in unicodedata.name(c, "") for c in sample)
    has_latin = any("LATIN" in unicodedata.name(c, "") for c in sample)
    if has_cyrillic and has_latin:
        homoglyph_result = _check_homoglyphs(content)
        if homoglyph_result:
            logger.warning("Homoglyph injection in %s: %s", filename, homoglyph_result)
            return ScanResult(
                blocked=True,
                category="homoglyph",
                match=homoglyph_result,
                file=filename,
                message=f"[BLOCKED] {filename}: {homoglyph_result}",
            )

    # Combining characters check
    combining_result = _check_combining_chars(content)
    if combining_result:
        logger.warning("Combining char abuse in %s: %s", filename, combining_result)
        return ScanResult(
            blocked=True,
            category="combining_chars",
            match=combining_result,
            file=filename,
            message=f"[BLOCKED] {filename}: {combining_result}",
        )

    return ScanResult(blocked=False)


def safe_read_context_file(path: str | None, max_chars: int = 3000) -> str | None:
    """Read context file with injection scanning. Returns None if blocked."""
    if path is None:
        return None
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return None
    try:
        content = p.read_text(encoding="utf-8")
    except Exception:
        logger.warning("Failed to read context file: %s", path)
        return None

    scan = scan_content(content, p.name)
    if scan.blocked:
        return None

    return content[:max_chars]
