"""Command: /install — auto-install missing dependencies via pip."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from src.bot.filters import OwnerOnly
from src.core.infra.gating import gates

router = Router(name="install_cmd")
router.message.filter(OwnerOnly())
logger = logging.getLogger(__name__)

# Mapping from gate names to pypi packages
_GATE_TO_PACKAGE = {
    "httpx": "httpx",
    "playwright": "playwright",
    "beautifulsoup": "beautifulsoup4",
    "pyyaml": "pyyaml",
    "psutil": "psutil",
    "faster_whisper": "faster-whisper",
}


@router.message(Command("install"))
async def cmd_install(message: Message) -> None:
    """Install missing Python dependencies via pip."""
    missing = gates.missing_install_hints
    if not missing:
        await message.answer("✅ Все зависимости уже установлены.")
        return

    packages = []
    for m in missing:
        pkg = _GATE_TO_PACKAGE.get(m["name"])
        if pkg:
            packages.append(pkg)

    if not packages:
        await message.answer(
            "⚠️ Нет pip-пакетов для установки (только системные зависимости)."
        )
        return

    pkgs_str = " ".join(packages)
    await message.answer(f"📦 Устанавливаю: <code>{pkgs_str}</code>...")

    try:
        proc = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, "-m", "pip", "install"] + packages,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode == 0:
            # Re-run gates to check if install succeeded
            gates.run_all()
            still_missing = gates.missing_install_hints
            if still_missing:
                names = [m["name"] for m in still_missing]
                msg = f"✅ <b>pip install</b> выполнен. Но ещё не хватает: <code>{', '.join(names)}</code>\n\n"
                for m in still_missing:
                    msg += f"• {m['description']}: <code>{m['install_hint']}</code>\n"
                msg += "\nПроверь <code>/gates</code>."
                await message.answer(msg)
            else:
                await message.answer(
                    "✅ <b>Все зависимости установлены!</b> Проверь <code>/gates</code>."
                )
        else:
            await message.answer(
                f"❌ <b>Ошибка pip:</b>\n<code>{proc.stderr[-800:]}</code>"
            )
    except subprocess.TimeoutExpired:
        await message.answer(
            "❌ <b>Таймаут</b>. Попробуй вручную: <code>pip install {pkgs_str}</code>"
        )
    except Exception as e:
        await message.answer(f"❌ <b>Ошибка:</b> {e}")


@router.message(Command("install_playwright"))
async def cmd_install_playwright(message: Message) -> None:
    """Install Playwright browser (chromium)."""
    await message.answer("📦 Устанавливаю <code>playwright chromium</code>...")

    try:
        proc = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, "-m", "playwright", "install", "chromium"],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if proc.returncode == 0:
            gates.run_all()
            await message.answer(
                "✅ <b>Chromium установлен!</b> Проверь <code>/gates</code>."
            )
        else:
            await message.answer(
                f"❌ <b>Ошибка:</b>\n<code>{proc.stderr[-800:]}</code>"
            )
    except subprocess.TimeoutExpired:
        await message.answer(
            "❌ <b>Таймаут</b>. Попробуй вручную: <code>playwright install chromium</code>"
        )
    except Exception as e:
        await message.answer(f"❌ <b>Ошибка:</b> {e}")
