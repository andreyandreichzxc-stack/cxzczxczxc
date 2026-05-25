# Draft

## Purpose
Генерирует черновики ответов с учётом стиля собеседника, памяти владельца и контекста отсутствия. Поддерживает 3 тона: тёплый, деловой, сухой. Работает через inline-кнопки в Telegram.

## Activation
- **Manual**: inline-кнопки «Отправить», «Редактировать», «Варианты тона» в сообщениях с черновиками
- **Automatic**: `draft_suggester.should_suggest()` — авто-предложение черновика при входящем сообщении (с rate-limit)
- **Agent**: `draft_agent.draft()` — вызывается Maestro при интенте `draft_reply`

## Files
- `src/agents/draft_agent.py` — `draft()` (генерирует черновик: style_hint + memory_hint + absence_hint → JSON с draft, tone, reasoning), `draft_variants()` (3 варианта тона)
- `src/bot/handlers/draft_actions.py` — роутер inline-кнопок черновиков (отправить, отредактировать, варианты)
- `src/core/contacts/draft_suggester.py` — `should_suggest()` (проверка необходимости), `suggest_draft()` (генерация через `summarizer.draft_reply()`), rate-limit `_check_rate_limit()`
- `src/core/intelligence/summarizer.py` — `draft_reply()` для draft_suggester
- `src/core/intelligence/maestro.py` — `draft` агент в AGENT_REGISTRY, `draft_reply` в интентах
- `src/bot/app.py` — подключение роутера `draft_actions.router`
- `src/core/actions/tool_registry.py` — регистрация `draft_reply`

## Tools
- `draft_reply` (generation, MEDIUM) — генерация черновика через tool_registry
- `draft_agent.draft()` (generation, MEDIUM) — агент черновика для Maestro
- `draft_suggester` (suggestion, LOW) — авто-предложение черновика

## Output
Telegram-сообщение с inline-кнопками, содержащее текст черновика и указание тона. Пользователь может: отправить как есть, отредактировать, или запросить другой тон.

## Dependencies
- **Env**: LLM-ключ (для генерации)
- **DB**: memory (контекст), contacts (style_profile)
- **Services**: memory_recall, style_matcher
