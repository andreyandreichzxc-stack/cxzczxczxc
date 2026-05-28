"""mcp_screenshot tool — registered via @tool decorator.

Take screenshots of web pages via Playwright (headless Chromium).

Actions:
- ``action="url" url="https://example.com" full_page=false``
    — navigate, wait 3 s for load, screenshot the full viewport.
- ``action="element" url="..." selector=".main-content"``
    — screenshot a specific element on the page.

Screenshots are saved to ``data/screenshots/`` with a timestamp-based filename.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from src.config import PROJECT_ROOT
from src.core.actions.tool_registry import tool

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────

_SCREENSHOTS_DIR = PROJECT_ROOT / "data" / "screenshots"
_NAVIGATE_TIMEOUT = 30_000  # ms (30 s)
_LOAD_WAIT_SEC = 3
_BROWSER_ARGS = ["--no-sandbox", "--disable-setuid-sandbox"]

# ══════════════════════════════════════════════════════════════════════════
# Tool: mcp_screenshot
# ══════════════════════════════════════════════════════════════════════════


@tool(
    name="mcp_screenshot",
    description=(
        "Take screenshots of web pages via Playwright. Two actions:\n"
        "- 'url' — navigate to a URL, wait 3 s, screenshot the full viewport.\n"
        "- 'element' — screenshot a specific CSS selector on the page.\n"
        "Saves to data/screenshots/ and returns the local path & file size."
    ),
    category="utility",
    risk="low",
    params={
        "action": "str — 'url' or 'element'",
        "url": "str — page URL to screenshot",
        "selector": "str — CSS selector (required for action='element')",
        "full_page": "bool — capture full scrollable page (default false, used with 'url')",
    },
)
async def mcp_screenshot(
    action: str = "",
    url: str = "",
    selector: str = "",
    full_page: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Take a screenshot of a web page or element via Playwright.

    Args:
        action: ``"url"`` or ``"element"``.
        url: The page URL to screenshot.
        selector: CSS selector (required when ``action="element"``).
        full_page: Capture the full scrollable page (default ``False``,
                   used with ``action="url"``).

    Returns:
        A dict with ``"path"``, ``"size_bytes"``, ``"url"`` or ``"error"``.
    """
    try:
        if action not in ("url", "element"):
            return {"error": f"Unknown action {action!r}. Valid: url, element"}

        if not url or not url.strip():
            return {"error": "url parameter is required"}

        url = url.strip()

        if action == "element" and not selector.strip():
            return {"error": "selector parameter is required for action='element'"}

        return await _take_screenshot(action, url, selector.strip(), full_page)

    except Exception as exc:
        logger.exception("mcp_screenshot(%r) failed", action)
        return {"error": str(exc)}


# ══════════════════════════════════════════════════════════════════════════
# Implementation
# ══════════════════════════════════════════════════════════════════════════


async def _take_screenshot(
    action: str,
    url: str,
    selector: str,
    full_page: bool,
) -> dict[str, Any]:
    """Headless Chromium screenshot via Playwright."""
    # Lazy import — playwright is an optional dependency
    try:
        from playwright.async_api import async_playwright  # type: ignore[import-untyped]
    except ImportError:
        return {
            "error": (
                "playwright not installed: "
                "pip install playwright && playwright install chromium"
            )
        }

    # Ensure output directory exists
    _SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Unique timestamped filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"screenshot_{timestamp}.png"
    output_path = _SCREENSHOTS_DIR / filename

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=_BROWSER_ARGS,
            )
            try:
                page = await browser.new_page()

                # Navigate with timeout
                await page.goto(url, timeout=_NAVIGATE_TIMEOUT, wait_until="load")

                # Brief wait for dynamic content
                await asyncio.sleep(_LOAD_WAIT_SEC)

                if action == "element":
                    element = await page.query_selector(selector)
                    if element is None:
                        return {"error": f"Selector {selector!r} not found on the page"}
                    await element.screenshot(path=str(output_path))
                else:
                    await page.screenshot(path=str(output_path), full_page=full_page)

            finally:
                await browser.close()

    except Exception as exc:
        logger.warning("Screenshot failed: %s", exc)
        # Clean up partial file if it exists
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        return {"error": str(exc)}

    if not output_path.is_file():
        return {"error": "Screenshot file was not created"}

    size_bytes = output_path.stat().st_size
    return {
        "ok": True,
        "path": str(output_path.relative_to(PROJECT_ROOT)),
        "size_bytes": size_bytes,
        "url": url,
    }
