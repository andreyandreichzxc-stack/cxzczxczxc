"""Команда /humanize — проверка текста на AI-шаблонность."""

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from src.bot.filters import OwnerOnly
from src.core.humanizer import analyze_ai_score, humanize_text

router = Router(name="humanize")
router.message.filter(OwnerOnly())


@router.message(Command("humanize"))
async def cmd_humanize(message: Message) -> None:
    """Проверить текст на AI-шаблонность.

    Использование:
    /humanize <текст>
    /humanize fix <текст> — с авто-исправлением
    """
    text = message.text or ""
    args = text.split(maxsplit=1)
    if len(args) < 2:
        await message.answer(
            "🎯 <b>Humanizer</b>\n"
            "Проверяет текст на AI-шаблонность.\n\n"
            "<b>Использование:</b>\n"
            "/humanize &lt;текст&gt; — оценить\n"
            "/humanize fix &lt;текст&gt; — оценить + исправить"
        )
        return

    text = args[1]
    do_fix = text.lower().startswith("fix ")
    if do_fix:
        text = text[4:]

    try:
        score, breakdown = analyze_ai_score(text)
    except Exception:
        await message.answer("❌ Не удалось проанализировать текст.")
        return

    # Формируем ответ
    emoji = "🟢" if score < 0.3 else ("🟡" if score < 0.6 else "🔴")
    pct = int(score * 100)

    lines = [f"{emoji} <b>AI-шаблонность: {pct}%</b>"]

    if breakdown.get("markers"):
        markers_str = ", ".join(m["phrase"] for m in breakdown["markers"][:5])
        lines.append(f"🔤 Маркеры: {markers_str}")

    if breakdown.get("patterns"):
        pat_str = ", ".join(p["label"] for p in breakdown["patterns"][:3])
        lines.append(f"📐 Паттерны: {pat_str}")

    if breakdown.get("repeats"):
        rep_str = ", ".join(
            f"{r['word']} (×{r['count']})" for r in breakdown["repeats"][:3]
        )
        lines.append(f"🔁 Повторы: {rep_str}")

    if breakdown.get("length_note"):
        lines.append(f"📏 {breakdown['length_note']}")

    if do_fix:
        try:
            fixed = humanize_text(text)
            fixed_score, _ = analyze_ai_score(fixed)
            fixed_pct = int(fixed_score * 100)
            lines.append("")
            lines.append(f"✅ <b>Исправлено ({fixed_pct}%):</b>")
            lines.append(f"<code>{fixed[:500]}</code>")
        except Exception:
            lines.append("")
            lines.append("⚠️ Не удалось исправить текст.")

    await message.answer("\n".join(lines))
