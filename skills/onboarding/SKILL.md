# Onboarding

## Purpose
Визард первичной настройки: проверяет готовность пользователя (`is_onboarded()`), проводит через шаги (логин, LLM-ключ, часовой пояс, синхронизация), блокирует доступ к боту до завершения.

## Activation
- **Manual**: `/start` — точка входа в визард
- **Automatic**: `onboarding_guard_middleware` — outer middleware: перенаправляет не-онбордингнутых на `/start` (пропускает `/start`, `/login`, `/cancel`, активные FSM)
- **Check**: `is_onboarded()` — проверка: есть сессия + есть LLM-ключ + TZ != UTC

## Files
- `src/bot/handlers/start.py` — `cmd_start()` (приветствие, кнопка «Начать»), FSM-шаги (логин, LLM-ключ, TZ, синхронизация), `advance_onboarding_after_login()`, `_finish_onboarding()`
- `src/bot/filters.py` — `is_onboarded()` — проверка готовности
- `src/bot/app.py` — `onboarding_guard_middleware` — outer middleware, блокирует не-готовых
- `src/bot/handlers/login.py` — `/login`: часть онбординга, вызывает `advance_onboarding_after_login()`
- `src/bot/handlers/free_text.py` — атомарная установка флага онбординга, кнопка «Пропустить» для persona
- `src/bot/handlers/settings.py` — секции настроек, настраиваемые в онбординге

## Tools
- `onboarding_guard_middleware` (middleware, MEDIUM) — guard middleware
- `is_onboarded()` (check, LOW) — pure function проверки

## Output
Пошаговый визард в Telegram: приветствие → логин → LLM-ключ → часовой пояс → синхронизация → готово. Сообщение со списком доступных команд после завершения.

## Dependencies
- **Env**: none — полностью через UI
- **DB**: UserSettings (session, llm_key, timezone, onboarded flag)
- **Services**: Telethon (сессия), LLM router (проверка ключа)
