"""Browser fingerprint pool — realistic Chrome profiles."""

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Fingerprint:
    """A single browser fingerprint profile."""

    user_agent: str
    sec_ch_ua: str
    sec_ch_ua_platform: str
    sec_ch_ua_mobile: str
    viewport_width: int
    viewport_height: int
    platform: str
    languages: str


# Pool of 8 realistic Chrome profiles (Win10, Win11, MacOS, different Chrome versions 120-131)
FINGERPRINTS: list[Fingerprint] = [
    Fingerprint(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        sec_ch_ua='"Google Chrome";v="131", "Chromium";v="131", "Not=A?Brand";v="24"',
        sec_ch_ua_platform='"Windows"',
        sec_ch_ua_mobile="?0",
        viewport_width=1920,
        viewport_height=1080,
        platform="Win32",
        languages="ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    ),
    # 2: Chrome 130 on Win10 (1366x768 laptop)
    Fingerprint(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/130.0.0.0 Safari/537.36"
        ),
        sec_ch_ua='"Google Chrome";v="130", "Chromium";v="130", "Not=A?Brand";v="99"',
        sec_ch_ua_platform='"Windows"',
        sec_ch_ua_mobile="?0",
        viewport_width=1366,
        viewport_height=768,
        platform="Win32",
        languages="ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    ),
    # 3: Chrome 131 on Win11 (2560x1440)
    Fingerprint(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        sec_ch_ua='"Google Chrome";v="131", "Chromium";v="131", "Not=A?Brand";v="24"',
        sec_ch_ua_platform='"Windows"',
        sec_ch_ua_mobile="?0",
        viewport_width=2560,
        viewport_height=1440,
        platform="Win32",
        languages="ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    ),
    # 4: Chrome 129 on MacOS (1440x900)
    Fingerprint(
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/129.0.0.0 Safari/537.36"
        ),
        sec_ch_ua='"Google Chrome";v="129", "Chromium";v="129", "Not=A?Brand";v="8"',
        sec_ch_ua_platform='"macOS"',
        sec_ch_ua_mobile="?0",
        viewport_width=1440,
        viewport_height=900,
        platform="MacIntel",
        languages="ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    ),
    # 5: Chrome 130 on MacOS (1680x1050)
    Fingerprint(
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/130.0.0.0 Safari/537.36"
        ),
        sec_ch_ua='"Google Chrome";v="130", "Chromium";v="130", "Not=A?Brand";v="99"',
        sec_ch_ua_platform='"macOS"',
        sec_ch_ua_mobile="?0",
        viewport_width=1680,
        viewport_height=1050,
        platform="MacIntel",
        languages="ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    ),
    # 6: Chrome 131 on Win10 (1536x864)
    Fingerprint(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        sec_ch_ua='"Google Chrome";v="131", "Chromium";v="131", "Not=A?Brand";v="24"',
        sec_ch_ua_platform='"Windows"',
        sec_ch_ua_mobile="?0",
        viewport_width=1536,
        viewport_height=864,
        platform="Win32",
        languages="ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    ),
    # 7: Chrome 128 on Win11 (1920x1080)
    Fingerprint(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/128.0.0.0 Safari/537.36"
        ),
        sec_ch_ua='"Google Chrome";v="128", "Chromium";v="128", "Not=A?Brand";v="8"',
        sec_ch_ua_platform='"Windows"',
        sec_ch_ua_mobile="?0",
        viewport_width=1920,
        viewport_height=1080,
        platform="Win32",
        languages="ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    ),
    # 8: Chrome 130 on Win10 (1280x720)
    Fingerprint(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/130.0.0.0 Safari/537.36"
        ),
        sec_ch_ua='"Google Chrome";v="130", "Chromium";v="130", "Not=A?Brand";v="99"',
        sec_ch_ua_platform='"Windows"',
        sec_ch_ua_mobile="?0",
        viewport_width=1280,
        viewport_height=720,
        platform="Win32",
        languages="ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    ),
]


class FingerprintPool:
    """Access to the fingerprint pool with rotation support."""

    @staticmethod
    def all() -> list[Fingerprint]:
        """Return all available fingerprints."""
        return list(FINGERPRINTS)

    @staticmethod
    def random() -> Fingerprint:
        """Return a random fingerprint from the pool."""
        return random.choice(FINGERPRINTS)

    @staticmethod
    def by_index(idx: int) -> Fingerprint:
        """Return a specific fingerprint by index (0-based)."""
        return FINGERPRINTS[idx % len(FINGERPRINTS)]


def random_fingerprint() -> Fingerprint:
    """Convenience: return a random fingerprint."""
    return FingerprintPool.random()
