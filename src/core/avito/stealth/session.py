"""AvitoSession — manages browser-like session lifecycle."""

from __future__ import annotations

import asyncio
import logging
import random
from typing import TYPE_CHECKING

from .cookies import CookieStore
from .fingerprint import Fingerprint, random_fingerprint
from .headers import build_headers

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)


class AvitoSession:
    """Browser-like session with cookies, fingerprint, warmup.

    Two-level fetching:
      1. httpx with realistic Chrome headers (fast).
      2. Playwright headless browser (fallback if blocked).

    Usage::

        session = AvitoSession()
        await session.warmup()
        resp = await session.fetch("https://www.avito.ru/moskva?q=iphone")
        html = resp.text
    """

    def __init__(
        self,
        proxy: str | None = None,
        use_browser_fallback: bool = True,
    ) -> None:
        self.fingerprint: Fingerprint = random_fingerprint()
        self.cookies: CookieStore = CookieStore()
        self.proxy: str | None = proxy
        self.use_browser_fallback: bool = use_browser_fallback
        self._created_at: float = asyncio.get_event_loop().time()
        self._requests_count: int = 0
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy-create the httpx AsyncClient (imports httpx on first use)."""
        if self._client is None:
            import httpx

            limits = httpx.Limits(max_keepalive_connections=1, max_connections=1)
            transport = httpx.AsyncHTTPTransport(retries=1)
            self._client = httpx.AsyncClient(
                http2=True,
                limits=limits,
                transport=transport,
                timeout=httpx.Timeout(30.0, connect=10.0),
                proxy=self.proxy,
            )
        return self._client

    async def warmup(self) -> bool:
        """Visit homepage to get initial cookies (mimics real user)."""
        await asyncio.sleep(random.uniform(2.0, 5.0))
        try:
            client = await self._get_client()
            resp = await client.get(
                "https://www.avito.ru/",
                headers=build_headers(
                    self.fingerprint, referer="https://www.google.com/"
                ),
                follow_redirects=True,
            )
            if resp.status_code == 200:
                cookie_dict: dict[str, str] = {}
                for cookie in resp.cookies.jar:
                    if cookie.value:
                        cookie_dict[cookie.name] = cookie.value
                if cookie_dict:
                    self.cookies.save("avito.ru", cookie_dict)
                logger.debug("Session warmed up: %d cookies", len(cookie_dict))
                return True
            logger.debug("Warmup got status %d", resp.status_code)
            return False
        except Exception:
            logger.debug("Warmup failed", exc_info=True)
            return False

    async def fetch(self, url: str) -> httpx.Response:
        """Fetch a page with full stealth.

        Level 1: httpx with realistic Chrome headers + persistent cookies.
        Level 2: Playwright headless browser (if Level 1 is blocked).
        """

        self._requests_count += 1
        headers = build_headers(self.fingerprint)

        # Load saved cookies
        cookies = self.cookies.load("avito.ru")

        # ---- Level 1: httpx ----
        client = await self._get_client()
        try:
            resp = await client.get(url, headers=headers, cookies=cookies)
            if _is_blocked(resp):
                logger.warning(
                    "httpx blocked (status %d, len %d), trying browser fallback",
                    resp.status_code,
                    len(resp.text),
                )
                if self.use_browser_fallback:
                    return await self._browser_fallback(url)
            return resp
        except Exception:
            logger.debug("httpx request failed", exc_info=True)
            if self.use_browser_fallback:
                return await self._browser_fallback(url)
            raise

    async def _browser_fallback(self, url: str) -> httpx.Response:
        """Level 2: Playwright browser fallback."""

        from .browser import StealthBrowser

        browser = StealthBrowser(proxy=self.proxy)
        html = await browser.fetch(url)
        return _make_pseudo_response(html, url)

    async def close(self) -> None:
        """Close the underlying httpx client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _is_blocked(resp: httpx.Response) -> bool:
    """Detect if Avito blocked the request."""
    if resp.status_code in (403, 429):
        return True
    text = resp.text.lower()
    if len(resp.text) < 500:
        return True
    if "captcha" in text or "проверка" in text or "доступ ограничен" in text:
        return True
    if "Доступ ограничен" in resp.text:
        return True
    return False


def _make_pseudo_response(html: str, url: str) -> "httpx.Response":
    """Wrap raw HTML in a response-like httpx.Response."""
    import httpx

    return httpx.Response(
        200,
        text=html,
        request=httpx.Request("GET", url),
    )
