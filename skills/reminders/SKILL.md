# Reminders

## Purpose
Отслеживает обещания и дедлайны владельца: автоматически извлекает из переписки, напоминает с опережением, показывает список активных.

## Activation
- **Manual**: `/todos` — список открытых обязательств с кнопками «Выполнено»/«Отменить»
- **Automatic**: `reminders_loop()` — каждые 300с проверяет просроченные/приближающиеся дедлайны, создаёт Notification
- **Free-text**: интенты `add_reminder`, `remove_reminder`, `add_reminders_from_chat`
- **Agent**: `commitment_agent.extract()` — вызывается Maestro для анализа переписки

## Files
- `src/core/scheduling/reminders.py` — `reminders_loop()`, `_check_once()` — фоновый цикл
- `src/core/actions/mcp_reminders.py` — `mcp_reminders(action="list"/"create")` — инструмент
- `src/agents/commitment_agent.py` — `extract()` — LLM-агент извлечения обещаний
- `src/bot/handlers/todos.py` — обработчик `/todos`
- `src/bot/handlers/free_text_exec.py` — `exec_add_reminder()`, `exec_remove_reminder()`, `exec_add_reminders_from_chat()`
- `src/bot/handlers/free_text_common.py` — маппинг интентов
- `src/bot/handlers/free_text_pipeline.py` — регистрация в пайплайне
- `src/core/actions/tool_registry.py` — регистрация `set_reminder`
- `src/db/models/_messaging.py` — модель `Commitment` (text, deadline_at, status, direction)
- `src/db/repo.py` — `add_commitment()`, `list_open_commitments()`, `update_commitment_status()`
- `src/db/models/_auth.py` — `reminders_enabled`, `reminder_lead_hours`, `reminder_overdue_enabled`
- `src/bot/handlers/settings.py` — настройки в `/settings`
- `src/core/intelligence/guardrails.py` — `add_reminder`/`remove_reminder` → MEDIUM, `schedule_reminder` → HIGH

## Tools
- `mcp_reminders` (data, MEDIUM) — list/create напоминаний
- `set_reminder` (action, MEDIUM) — установка напоминания
- `commitment` agent (analysis, LOW) — извлечение обещаний из переписки

## Output
Список обязательств в Telegram (текст, срок, направление). Уведомления о просроченных/приближающихся дедлайнах.

## Dependencies
- **Env**: LLM-ключ (для commitment_agent)
- **DB**: таблица `commitments`, поля в UserSettings
- **Services**: scheduling loop, notification_queue
