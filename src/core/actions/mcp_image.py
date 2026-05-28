"""mcp_image tool — registered via @tool decorator.

Image analysis via Pillow (PIL).  Read-only image inspection.

Actions:
- ``action="info" path="data/photo.jpg"`` — dimensions, format, mode, DPI, file size
- ``action="exif" path="data/photo.jpg"`` — EXIF metadata (camera, GPS, date)
- ``action="dominant_colors" path="data/photo.jpg" count=5`` — most common colours as hex

All operations are read-only.  Path must be within ``settings.data_dir``.
Pillow is imported lazily.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from src.config import settings
from src.core.actions.tool_registry import tool

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────

_MAX_DOMINANT_COLORS = 20


# ══════════════════════════════════════════════════════════════════════════
# Tool: mcp_image
# ══════════════════════════════════════════════════════════════════════════


@tool(
    name="mcp_image",
    description=(
        "Read-only image analysis.  Supports three actions:\n"
        "- 'info' — dimensions, format, mode, DPI, file size.\n"
        "- 'exif' — EXIF metadata (camera, GPS, date).\n"
        "- 'dominant_colors' — most common colours as hex codes.\n"
        "All operations are read-only.  Path must be within data/ directory."
    ),
    category="utility",
    risk="low",
    params={
        "action": "str — 'info', 'exif', or 'dominant_colors'",
        "path": "str — path to an image file (required)",
        "count": "int — number of dominant colours (default 5, max 20, used with 'dominant_colors')",
    },
)
async def mcp_image(
    action: str,
    path: str = "",
    count: int = 5,
    **kwargs: Any,
) -> dict[str, Any]:
    """Image analysis tool.

    Args:
        action: ``"info"``, ``"exif"``, or ``"dominant_colors"``.
        path: Path to an image file (required).
        count: Number of dominant colours (default 5, max 20).

    Returns:
        A dict with image metadata or an ``"error"`` key.
    """
    try:
        if action not in ("info", "exif", "dominant_colors"):
            return {
                "error": f"Unknown action {action!r}. "
                f"Valid actions: info, exif, dominant_colors"
            }

        if not path or not path.strip():
            return {"error": "path parameter is required"}

        resolved = _safe_image_path(path.strip())
        if resolved is None:
            return {"error": f"Path {path!r} is outside allowed directories"}
        if not resolved.is_file():
            return {"error": f"File not found: {resolved}"}

        if action == "info":
            return await _image_info(resolved)
        elif action == "exif":
            return await _image_exif(resolved)
        else:  # dominant_colors
            n = max(1, min(count, _MAX_DOMINANT_COLORS))
            return await _dominant_colors(resolved, n)
    except Exception as exc:
        logger.exception("mcp_image(%r) failed", action)
        return {"error": str(exc)}


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════


def _safe_image_path(raw: str) -> Path | None:
    """Resolve *raw* to an absolute path within ``settings.data_dir``.

    Returns ``None`` if the path is outside allowed directories or
    contains ``..`` as a path component.
    """
    import os

    normalised = raw.replace("/", os.sep).replace("\\", os.sep)
    if ".." in normalised.split(os.sep):
        return None

    resolved = Path(raw).resolve()
    data_dir = settings.data_dir.resolve()

    try:
        resolved.relative_to(data_dir)
    except ValueError:
        return None

    return resolved


def _import_pil() -> Any:
    """Lazy import of PIL.Image. Raises ImportError if not installed."""
    try:
        from PIL import Image  # type: ignore[import-untyped]

        return Image
    except ImportError:
        raise ImportError("Pillow not installed: pip install Pillow")


# ══════════════════════════════════════════════════════════════════════════
# Action implementations
# ══════════════════════════════════════════════════════════════════════════


async def _image_info(image_path: Path) -> dict[str, Any]:
    """Get basic image metadata."""
    Image = _import_pil()
    loop = asyncio.get_running_loop()

    def _do() -> dict[str, Any]:
        try:
            img = Image.open(str(image_path))
        except Exception as exc:
            raise ValueError(f"Cannot open image: {exc}")

        width, height = img.size
        file_size = image_path.stat().st_size
        dpi = None
        try:
            dpi_info = img.info.get("dpi")
            if dpi_info:
                dpi = round(dpi_info[0], 1)
        except Exception:
            pass

        return {
            "width": width,
            "height": height,
            "format": img.format or "unknown",
            "mode": img.mode,
            "dpi": dpi,
            "file_size_bytes": file_size,
            "file_size_kb": round(file_size / 1024, 1),
        }

    try:
        result = await loop.run_in_executor(None, _do)
    except ImportError as exc:
        return {"error": str(exc)}
    except ValueError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        logger.warning("Image info error: %s", exc)
        return {"error": f"Failed to read image info: {exc}"}

    result["ok"] = True
    return result


async def _image_exif(image_path: Path) -> dict[str, Any]:
    """Extract EXIF metadata from an image."""
    Image = _import_pil()
    loop = asyncio.get_running_loop()

    def _do() -> dict[str, Any]:
        try:
            img = Image.open(str(image_path))
        except Exception as exc:
            raise ValueError(f"Cannot open image: {exc}")

        exif_data = img.getexif()
        if not exif_data:
            return {"exif_present": False, "exif": {}}

        from PIL.ExifTags import TAGS, GPSTAGS

        decoded: dict[str, Any] = {}
        gps: dict[str, Any] = {}

        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, str(tag_id))

            if tag_name == "GPSInfo":
                for gps_tag_id, gps_value in value.items():
                    gps_tag_name = GPSTAGS.get(gps_tag_id, str(gps_tag_id))
                    gps[gps_tag_name] = str(gps_value)
                decoded["GPSInfo"] = gps
            else:
                # Convert bytes to string representation
                if isinstance(value, bytes):
                    try:
                        value = value.decode("utf-8", errors="replace")
                    except Exception:
                        value = str(value)
                decoded[tag_name] = str(value)

        # Extract key fields for the summary
        summary: dict[str, Any] = {
            "Make": decoded.get("Make"),
            "Model": decoded.get("Model"),
            "DateTimeOriginal": decoded.get("DateTimeOriginal"),
            "Software": decoded.get("Software"),
            "GPSInfo": gps if gps else None,
        }

        # Remove None values
        summary = {k: v for k, v in summary.items() if v is not None}

        return {
            "exif_present": True,
            "exif_count": len(decoded),
            "summary": summary,
        }

    try:
        result = await loop.run_in_executor(None, _do)
    except ImportError as exc:
        return {"error": str(exc)}
    except ValueError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        logger.warning("Image EXIF error: %s", exc)
        return {"error": f"Failed to read EXIF: {exc}"}

    result["ok"] = True
    return result


async def _dominant_colors(image_path: Path, count: int) -> dict[str, Any]:
    """Get the most common colours in an image.

    The image is resized to 1 pixel high while keeping aspect ratio,
    then the most common colours are extracted and returned as hex strings.
    """
    Image = _import_pil()
    loop = asyncio.get_running_loop()

    def _do() -> dict[str, Any]:
        try:
            img = Image.open(str(image_path)).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Cannot open image: {exc}")

        # Resize to 1px high, preserving aspect ratio
        w, h = img.size
        if h > 1:
            new_w = max(1, w // h)
            img = img.resize((new_w, 1), Image.Resampling.LANCZOS)

        pixels = list(img.getdata())
        # Count colour frequencies
        colour_counts: dict[tuple[int, int, int], int] = {}
        for pixel in pixels:
            colour_counts[pixel] = colour_counts.get(pixel, 0) + 1

        # Sort by frequency
        sorted_colours = sorted(colour_counts.items(), key=lambda x: -x[1])

        colors: list[dict[str, Any]] = []
        for (r, g, b), freq in sorted_colours[:count]:
            hex_str = f"#{r:02x}{g:02x}{b:02x}"
            total = sum(colour_counts.values())
            percentage = round(freq / total * 100, 1) if total else 0
            colors.append(
                {
                    "hex": hex_str,
                    "rgb": {"r": r, "g": g, "b": b},
                    "percentage": percentage,
                }
            )

        return {
            "colors": colors,
            "count": len(colors),
        }

    try:
        result = await loop.run_in_executor(None, _do)
    except ImportError as exc:
        return {"error": str(exc)}
    except ValueError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        logger.warning("Image dominant_colors error: %s", exc)
        return {"error": f"Failed to extract dominant colours: {exc}"}

    result["ok"] = True
    return result
