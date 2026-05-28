"""Playwright-based stealth browser — Level 2 fallback.

Uses headless Chromium with anti-detection patches:
- Removes ``navigator.webdriver`` flag.
- Fakes ``navigator.plugins`` and ``navigator.languages``.
- Injects ``window.chrome`` object.
- Scrolls and waits like a human.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import TYPE_CHECKING

from .fingerprint import Fingerprint, random_fingerprint

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class StealthBrowser:
    """Headless Chrome with anti-detection patches.

    Usage::

        browser = StealthBrowser()
        html = await browser.fetch("https://www.avito.ru/moskva?q=iphone")
    """

    def __init__(self, proxy: str | None = None) -> None:
        self.proxy: str | None = proxy
        self.fingerprint: Fingerprint = random_fingerprint()

    async def fetch(self, url: str) -> str:
        """Fetch a page using stealth browser.

        Returns:
            HTML content as string. Empty string if Playwright is not installed.
        """
        try:
            from playwright.async_api import async_playwright  # type: ignore[import-untyped]
        except ImportError:
            logger.error(
                "Playwright not installed — install: "
                "pip install playwright && playwright install chromium"
            )
            return ""

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    f"--window-size={self.fingerprint.viewport_width},{self.fingerprint.viewport_height}",
                ],
            )
            context = await browser.new_context(
                user_agent=self.fingerprint.user_agent,
                viewport={
                    "width": self.fingerprint.viewport_width,
                    "height": self.fingerprint.viewport_height,
                },
                locale="ru-RU",
            )

            # Remove automation traces
            await context.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
                Object.defineProperty(navigator, 'languages', {get: () => ['ru-RU', 'ru', 'en-US']});
                window.chrome = {runtime: {}};
            """)

            page = await context.new_page()

            try:
                # Human-like: visit homepage first
                await page.goto("https://www.avito.ru/", wait_until="domcontentloaded")
                await asyncio.sleep(random.uniform(2.0, 5.0))

                # Scroll down slightly
                await page.evaluate("window.scrollBy(0, 300)")
                await asyncio.sleep(random.uniform(0.5, 1.5))

                # Navigate to target
                await page.goto(url, wait_until="networkidle")
                html = await page.content()
            finally:
                await browser.close()

            return html
