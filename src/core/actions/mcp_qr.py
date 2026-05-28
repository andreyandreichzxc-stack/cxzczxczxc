"""mcp_qr tool — registered via @tool decorator.

QR code generation using the ``qrcode`` library (``pip install qrcode[pil]``).

Actions:
  - **generate** — create a QR code PNG image and save it to ``data/qr/``.
"""

from __future__ import annotations

import importlib
import logging
import uuid
from datetime import datetime
from typing import Any

from src.config import settings
from src.core.actions.tool_registry import tool

logger = logging.getLogger(__name__)

_QR_DIR_REL = "qr"


@tool(
    name="mcp_qr",
    description=(
        "Generate QR codes as PNG images.\n\n"
        "Actions:\n"
        "- **generate** — create a QR code and save to data/qr/.\n\n"
        "Requires: qrcode[pil] (``pip install qrcode[pil]``)\n\n"
        "Examples:\n"
        '  action="generate" data="https://example.com" size=300'
    ),
    category="utility",
    risk="low",
    params={
        "action": "str — 'generate'",
        "data": "str — the content to encode (URL, text, etc.)",
        "size": "int — QR code image size in pixels (default 300, min 100, max 2000)",
    },
)
async def mcp_qr(
    action: str = "",
    data: str = "",
    size: int = 300,
    **kwargs: Any,
) -> dict[str, Any]:
    """Generate QR codes using the ``qrcode`` library."""
    try:
        if action == "generate":
            return await _generate_qr(data, size)
        else:
            return {"error": (f"Unknown action {action!r}. Valid actions: generate")}
    except ValueError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        logger.exception("mcp_qr(%r) failed", action)
        return {"error": f"Unexpected error: {exc}"}


# ── Helpers ──────────────────────────────────────────────────────────────


async def _generate_qr(data: str, size: int) -> dict[str, Any]:
    if not data or not data.strip():
        return {"error": "data parameter is required for action='generate'"}

    # Validate size
    if size < 100:
        size = 100
    elif size > 2000:
        size = 2000

    # Lazy import qrcode
    qrcode = _lazy_import_qrcode()
    if qrcode is None:
        return {"error": "qrcode not installed: pip install qrcode[pil]"}

    # Ensure output directory exists
    qr_dir = settings.data_dir / _QR_DIR_REL
    qr_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    filename = f"qr_{timestamp}_{unique_id}.png"
    file_path = qr_dir / filename

    # Generate QR code
    qr = qrcode.QRCode(
        version=None,  # auto-detect
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=4,
    )
    qr.add_data(data.strip())
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")

    # Resize if needed (qrcode generates fixed-size; we scale the final image)
    if size != 300:
        img = img.resize((size, size))

    img.save(file_path, format="PNG")
    size_bytes = file_path.stat().st_size

    return {
        "ok": True,
        "action": "generate",
        "path": str(file_path),
        "filename": filename,
        "size_bytes": size_bytes,
    }


def _lazy_import_qrcode() -> Any:
    """Try to import the ``qrcode`` package.

    Returns the module on success or ``None`` if not installed.
    """
    try:
        return importlib.import_module("qrcode")
    except ImportError:
        return None
