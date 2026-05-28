"""Avito Stealth — multi-layered anti-detection fetcher.

Replaces naive requests.get() with browser-like sessions:
- Level 1: httpx with realistic Chrome headers + cookie persistence
- Level 2: Playwright headless browser fallback (if blocked)

Exports:
    AvitoFetcher  — high-level async fetcher
    AvitoSession  — browser-like session lifecycle
    FingerprintPool — access to fingerprint profiles
    get_fetcher   — factory function
"""

from .fingerprint import Fingerprint, FingerprintPool, random_fingerprint
from .session import AvitoSession

__all__ = [
    "AvitoSession",
    "Fingerprint",
    "FingerprintPool",
    "get_fetcher",
    "random_fingerprint",
]


def get_fetcher(proxy: str | None = None) -> AvitoSession:
    """Factory: create a stealth session ready for fetching."""
    return AvitoSession(proxy=proxy)
