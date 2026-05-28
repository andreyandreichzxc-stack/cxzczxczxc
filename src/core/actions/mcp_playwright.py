"""mcp_playwright tool — registered via @tool decorator.

Browser automation via Playwright (headless Chromium).  Provides navigation,
screenshot, click, JavaScript evaluation, and accessibility snapshot actions.

Actions:
- ``action="navigate" url="https://..." wait=3``
    — opens URL, waits, returns page text[:3000] + title.
- ``action="screenshot" url="https://..." full_page=false``
    — takes screenshot, saves to ``data/screenshots/``, returns path.
- ``action="click" url="https://..." selector="button.submit"``
    — navigates, clicks element, returns new page text.
- ``action="evaluate" url="https://..." js="document.title"``
    — runs JavaScript, returns result.
- ``action="snapshot" url="https://..."``
    — returns accessibility tree text.

Security:
- SSRF protection via ``_check_ssrf()`` from ``mcp_http``.
- Only ``http://`` and ``https://`` schemes are allowed.
- Screenshots are saved only to ``data/screenshots/``.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from src.config import PROJECT_ROOT, settings
from src.core.actions.mcp_http import _check_ssrf
from src.core.actions.tool_registry import tool

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────

_SCREENSHOTS_DIR: Path = settings.data_dir / "screenshots"
_NAVIGATE_TIMEOUT = 30_000  # ms (30 s)
_OPERATION_TIMEOUT = 30  # seconds for asyncio.wait_for
_IDLE_TIMEOUT = 300  # seconds (5 min) before closing idle page
_BROWSER_ARGS = ["--no-sandbox", "--disable-setuid-sandbox"]
_MAX_TEXT_CHARS = 3000
_VALID_SCHEMES = frozenset({"http", "https"})
_BLOCKED_SCHEMES = frozenset({"file", "chrome", "data", "javascript"})

_VALID_ACTIONS = frozenset({"navigate", "screenshot", "click", "evaluate", "snapshot"})


# ══════════════════════════════════════════════════════════════════════════
# Browser Manager (singleton)
# ══════════════════════════════════════════════════════════════════════════


class _BrowserManager:
    """Manages a singleton Playwright browser instance with idle timeout.

    Lazily initialises the browser and page on first use.  After
    ``_IDLE_TIMEOUT`` seconds of inactivity the page is closed (the
    browser process stays alive for reuse).

    Thread-safe via ``asyncio.Lock``.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._playwright: Any = None
        self._browser: Any = None
        self._page: Any = None
        self._last_used: float = 0.0
        self._idle_task: asyncio.Task[None] | None = None
        self._closed = False

    # ── Public API ─────────────────────────────────────────────────────

    async def ensure_page(self) -> Any:
        """Return the current page (lazy-init browser if needed).

        Acquires the lock, starts browser on first call, opens a new
        page if none exists, and resets the idle timer.
        """
        async with self._lock:
            if self._closed:
                raise RuntimeError("BrowserManager has been closed")

            if self._playwright is None:
                await self._init_browser()

            if self._page is None:
                self._page = await self._browser.new_page()

            self._touch()
            return self._page

    async def close_page(self) -> None:
        """Close the current page (keep browser alive)."""
        async with self._lock:
            if self._page is not None:
                try:
                    await self._page.close()
                except Exception:
                    logger.debug("Error closing page", exc_info=True)
                self._page = None
            self._cancel_idle_timer()

    async def close(self) -> None:
        """Shut down the browser and clean up."""
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            self._cancel_idle_timer()
            if self._browser is not None:
                try:
                    await self._browser.close()
                except Exception:
                    logger.debug("Error closing browser", exc_info=True)
                self._browser = None
            if self._playwright is not None:
                try:
                    await self._playwright.stop()
                except Exception:
                    logger.debug("Error stopping playwright", exc_info=True)
                self._playwright = None
            self._page = None

    # ── Internal helpers ───────────────────────────────────────────────

    async def _init_browser(self) -> None:
        """Lazy-import playwright and launch Chromium."""
        try:
            from playwright.async_api import async_playwright  # type: ignore[import-untyped]
        except ImportError:
            raise RuntimeError(
                "playwright not installed: "
                "pip install playwright && playwright install chromium"
            )

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=True,
            args=_BROWSER_ARGS,
        )

    def _touch(self) -> None:
        """Record activity and restart idle timer."""
        self._last_used = time.monotonic()
        self._cancel_idle_timer()
        self._idle_task = asyncio.create_task(self._idle_loop())

    def _cancel_idle_timer(self) -> None:
        if self._idle_task is not None and not self._idle_task.done():
            self._idle_task.cancel()
        self._idle_task = None

    async def _idle_loop(self) -> None:
        """Wait for idle timeout, then close the page."""
        try:
            await asyncio.sleep(_IDLE_TIMEOUT)
            async with self._lock:
                if self._closed:
                    return
                if self._page is not None:
                    logger.info(
                        "Closing idle page (no activity for %ds)", _IDLE_TIMEOUT
                    )
                    try:
                        await self._page.close()
                    except Exception:
                        logger.debug("Error closing idle page", exc_info=True)
                    self._page = None
        except asyncio.CancelledError:
            pass


# Module-level singleton
_browser_manager = _BrowserManager()


# ══════════════════════════════════════════════════════════════════════════
# atexit cleanup — best-effort close on interpreter shutdown
# ══════════════════════════════════════════════════════════════════════════


def _atexit_cleanup() -> None:
    """Synchronous cleanup hook — schedules close on any available loop."""
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            loop.create_task(_browser_manager.close())
    except RuntimeError:
        pass  # No running loop — browser will be killed on process exit


atexit.register(_atexit_cleanup)


# ══════════════════════════════════════════════════════════════════════════
# Tool: playwright
# ══════════════════════════════════════════════════════════════════════════


@tool(
    name="playwright",
    description=(
        "Browser automation via Playwright (headless Chromium). "
        "Supports five actions:\n"
        "- 'navigate' — open a URL, wait, return page text + title.\n"
        "- 'screenshot' — screenshot a page, save to data/screenshots/.\n"
        "- 'click' — navigate to a URL, click a CSS selector, return new text.\n"
        "- 'evaluate' — run JavaScript on the page, return the result.\n"
        "- 'snapshot' — return the accessibility tree of the page.\n"
        "SSRF protection is enabled — internal/private IPs are blocked."
    ),
    category="system",
    risk="medium",
    requires_confirmation=True,
    params={
        "action": "str — one of: navigate, screenshot, click, evaluate, snapshot",
        "url": "str — page URL (required for navigate, screenshot, click, evaluate, snapshot)",
        "selector": "str — CSS selector (required for action='click')",
        "js": "str — JavaScript expression (required for action='evaluate')",
        "full_page": "bool — capture full scrollable page (default false, used with screenshot)",
        "wait": "int — seconds to wait after navigation (default 3, used with navigate)",
    },
)
async def mcp_playwright(
    action: str = "",
    url: str = "",
    selector: str = "",
    js: str = "",
    full_page: bool = False,
    wait: int = 3,
    **kwargs: Any,
) -> dict[str, Any]:
    """Browser automation via Playwright.

    Args:
        action: ``"navigate"``, ``"screenshot"``, ``"click"``,
                ``"evaluate"``, or ``"snapshot"``.
        url: Target page URL (required for most actions).
        selector: CSS selector (required for ``action="click"``).
        js: JavaScript expression (required for ``action="evaluate"``).
        full_page: Capture full scrollable page (default ``False``,
                   used with ``action="screenshot"``).
        wait: Seconds to wait after navigation (default 3).

    Returns:
        A dict with action-specific result fields or ``"error"``.
    """
    try:
        # ── Validate action ────────────────────────────────────────────
        if action not in _VALID_ACTIONS:
            return {
                "error": (
                    f"Unknown action {action!r}. "
                    f"Valid: {', '.join(sorted(_VALID_ACTIONS))}"
                )
            }

        # ── Validate URL (for actions that need it) ────────────────────
        needs_url = {"navigate", "screenshot", "click", "evaluate", "snapshot"}
        if action in needs_url:
            if not url or not url.strip():
                return {"error": "url parameter is required"}
            url = url.strip()

            # Check scheme
            try:
                parsed = urlparse(url)
            except Exception:
                return {"error": f"Invalid URL: {url}"}

            if parsed.scheme in _BLOCKED_SCHEMES or parsed.scheme not in _VALID_SCHEMES:
                return {
                    "error": (
                        f"URL scheme {parsed.scheme!r} is not allowed. "
                        f"Only http:// and https:// are supported."
                    )
                }

            # SSRF protection
            ssrf_error = _check_ssrf(url)
            if ssrf_error:
                return ssrf_error

        # ── Route to action handler ────────────────────────────────────
        handlers = {
            "navigate": _handle_navigate,
            "screenshot": _handle_screenshot,
            "click": _handle_click,
            "evaluate": _handle_evaluate,
            "snapshot": _handle_snapshot,
        }

        handler = handlers[action]
        return await asyncio.wait_for(
            handler(url=url, selector=selector, js=js, full_page=full_page, wait=wait),
            timeout=_OPERATION_TIMEOUT,
        )

    except asyncio.TimeoutError:
        logger.warning(
            "playwright action %r timed out after %ds", action, _OPERATION_TIMEOUT
        )
        return {"error": f"Operation timed out after {_OPERATION_TIMEOUT}s"}
    except RuntimeError as exc:
        logger.warning("playwright runtime error: %s", exc)
        return {"error": str(exc)}
    except Exception as exc:
        logger.exception("playwright(%r) failed unexpectedly", action)
        return {"error": f"Unexpected error: {exc}"}


# ══════════════════════════════════════════════════════════════════════════
# Action Handlers
# ══════════════════════════════════════════════════════════════════════════


async def _handle_navigate(
    url: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Navigate to *url*, wait, return page text + title."""
    page = await _browser_manager.ensure_page()
    wait_sec = kwargs.get("wait", 3)

    await page.goto(url, timeout=_NAVIGATE_TIMEOUT, wait_until="load")
    await asyncio.sleep(wait_sec)

    title = await page.title()
    text = await page.evaluate("document.body?.innerText || ''")
    text_preview = text[:_MAX_TEXT_CHARS] if text else ""
    truncated = len(text) > _MAX_TEXT_CHARS if text else False

    return {
        "ok": True,
        "title": title,
        "text": text_preview,
        "truncated": truncated,
        "total_chars": len(text) if text else 0,
    }


async def _handle_screenshot(
    url: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Navigate to *url*, screenshot, return path."""
    page = await _browser_manager.ensure_page()
    full_page = kwargs.get("full_page", False)

    await page.goto(url, timeout=_NAVIGATE_TIMEOUT, wait_until="load")
    await asyncio.sleep(3)

    # Ensure screenshots directory exists
    _SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"playwright_{timestamp}.png"
    output_path = _SCREENSHOTS_DIR / filename

    await page.screenshot(path=str(output_path), full_page=full_page)

    if not output_path.is_file():
        return {"error": "Screenshot file was not created"}

    size_bytes = output_path.stat().st_size
    return {
        "ok": True,
        "path": str(output_path.relative_to(PROJECT_ROOT)),
        "size_bytes": size_bytes,
        "url": url,
    }


async def _handle_click(
    url: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Navigate to *url*, click *selector*, return new page text."""
    selector = kwargs.get("selector", "")
    if not selector.strip():
        return {"error": "selector parameter is required for action='click'"}

    page = await _browser_manager.ensure_page()

    await page.goto(url, timeout=_NAVIGATE_TIMEOUT, wait_until="load")
    await asyncio.sleep(2)

    element = await page.query_selector(selector)
    if element is None:
        return {"error": f"Selector {selector!r} not found on the page"}

    await element.click()

    # Wait a bit for dynamic content after click
    await asyncio.sleep(2)

    title = await page.title()
    text = await page.evaluate("document.body?.innerText || ''")
    text_preview = text[:_MAX_TEXT_CHARS] if text else ""
    truncated = len(text) > _MAX_TEXT_CHARS if text else False

    return {
        "ok": True,
        "title": title,
        "text": text_preview,
        "truncated": truncated,
        "total_chars": len(text) if text else 0,
        "selector": selector,
    }


async def _handle_evaluate(
    url: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Navigate to *url*, run *js*, return result."""
    js_code = kwargs.get("js", "")
    if not js_code.strip():
        return {"error": "js parameter is required for action='evaluate'"}

    page = await _browser_manager.ensure_page()

    await page.goto(url, timeout=_NAVIGATE_TIMEOUT, wait_until="load")
    await asyncio.sleep(2)

    result = await page.evaluate(js_code)

    return {
        "ok": True,
        "result": str(result),
        "url": url,
    }


async def _handle_snapshot(
    url: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Navigate to *url*, return accessibility tree text."""
    page = await _browser_manager.ensure_page()

    await page.goto(url, timeout=_NAVIGATE_TIMEOUT, wait_until="load")
    await asyncio.sleep(3)

    # Use Playwright's built-in accessibility snapshot
    snapshot = await page.accessibility.snapshot()

    if snapshot is None:
        return {
            "ok": True,
            "snapshot": "",
            "note": "No accessibility tree available",
        }

    # Format the snapshot as readable text
    lines: list[str] = []
    _format_a11y_node(snapshot, lines, indent=0)

    return {
        "ok": True,
        "snapshot": "\n".join(lines),
        "url": url,
    }


def _format_a11y_node(node: dict[str, Any], lines: list[str], indent: int = 0) -> None:
    """Recursively format an accessibility node into text lines."""
    prefix = "  " * indent
    name = node.get("name", "")
    role = node.get("role", "")
    value = node.get("value", "")
    description = node.get("description", "")

    parts = [f"{prefix}[{role}]"]
    if name:
        parts.append(f"name={name!r}")
    if value:
        parts.append(f"value={value!r}")
    if description:
        parts.append(f"desc={description!r}")

    lines.append(" ".join(parts))

    for child in node.get("children", []):
        _format_a11y_node(child, lines, indent + 1)
