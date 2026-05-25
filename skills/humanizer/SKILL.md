# Humanizer

## Purpose
Anti-AI humanizer: заменяет AI-маркеры («конечно», «я понимаю», «в заключение» — 40+ фраз) на человеческие аналоги, удаляет шаблонные концовки, оценивает «AI-шность» текста (0.0–1.0).

## Activation
- **Manual**: `/humanize <текст>` — показать AI-score и исправить текст, `/humanize fix <текст>` — исправить
- **Automatic**: `ANTI_AI_BLOCK` — промпт-блок, инжектируемый в system-промпт Maestro при `anti_ai_enabled=True`; подавляет AI-шаблоны на этапе генерации

## Files
- `src/core/humanizer/humanizer.py` — `humanize_text()` (замена 40+ AI-маркеров на человеческие), `humanize_response()` (удаление шаблонных концовок + контекстные follow-up)
- `src/core/humanizer/scorer.py` — `analyze_ai_score()` (pure function: 0.0–1.0, проверяет маркеры, паттерны, повторы, длину)
- `src/core/humanizer/vocabulary.py` — `AI_MARKERS` (словарь маркеров с весами), `REPEAT_PENALTY`, `REPEAT_THRESHOLD`, `MAX_THEORETICAL_SCORE`
- `src/core/humanizer/patterns.py` — `AI_PATTERNS` (regex-паттерны), `IDEAL_LENGTH_MIN/MAX`
- `src/core/humanizer/stats.py` — runtime-статистика: `record_check()`, `get_stats()`
- `src/bot/handlers/humanize_cmd.py` — обработчик `/humanize`
- `src/core/intelligence/soul_blocks.py` — `ANTI_AI_BLOCK` (промпт-блок)
- `src/core/intelligence/prompt_assembler.py` — инжектит `ANTI_AI_BLOCK` при `anti_ai=True`
- `src/db/models/_auth.py` — `anti_ai_enabled`

## Tools
- `humanize_text` (text, LOW) — замена AI-маркеров
- `analyze_ai_score` (analysis, LOW) — pure function оценки AI-шности
- `ANTI_AI_BLOCK` (prompt, LOW) — превентивный промпт-блок

## Output
Исправленный текст (без AI-маркеров) + AI-score (число 0.0–1.0 и «🥷 Живой» / «🤖 AI»). Статистика использования.

## Dependencies
- **Env**: none — pure Python
- **DB**: `anti_ai_enabled` в UserSettings
- **No services required**
