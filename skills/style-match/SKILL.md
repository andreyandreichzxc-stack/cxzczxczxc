# Style Match

## Purpose
Анализирует стиль общения пользователя (длина сообщений, emoji, тон, лексика) и адаптирует ответы ассистента под этот стиль.

## Activation
- **Manual**: `/style` — показать или обновить стилевой профиль контакта
- **Automatic**: `analyze_user_style()` — вызывается при каждом запросе к Maestro (с TTL-кешем 1 час)
- **Background**: `update_global_style_profile()` — периодическое обновление глобального профиля владельца

## Files
- `src/core/intelligence/style_matcher.py` — `analyze_user_style()` (последние 50 сообщений: длина, emoji_rate, caps_rate, тон, water_tolerance), `get_or_update_style_profile()` → style_match_block для system-промпта
- `src/core/contacts/style_profile.py` — `build_style_profile()` (LLM-анализ), `style_profile_as_prompt_hint()`, `update_global_style_profile()`, JSON-профиль: address, register, length, emoji_usage, punctuation, typical_openings/closings
- `src/core/contacts/style_heuristics.py` — эвристики без LLM: мат-словарь, длина, эмоциональность, пунктуация
- `src/bot/handlers/style_cmd.py` — обработчик `/style`
- `src/core/intelligence/maestro.py` — инжектит style_match_block в system-промпт
- `src/core/intelligence/prompt_assembler.py` — поле style_match_block в AssemblyContext
- `src/userbot/auto_reply.py` — использует style_profile_as_prompt_hint() в авто-ответе
- `src/db/models/_contacts.py` — `style_profile` (JSON)
- `src/db/models/_base.py` — `global_style_profile` (JSON)
- `src/db/models/_learning.py` — `style_profile` в AdaptivePersona

## Tools
- `style_matcher.analyze_user_style()` (analysis, LOW) — профилирование стиля

## Output
`style_match_block` — текстовый блок для system-промпта: «Говори как пользователь: длина ~N слов, emoji часто/редко, тон X, обращения Y».

## Dependencies
- **Env**: LLM-ключ (для build_style_profile)
- **DB**: contacts.style_profile, adaptive_persona.style_profile, global_style_profile
- **Services**: userbot mirror (доступ к исходящим сообщениям)
