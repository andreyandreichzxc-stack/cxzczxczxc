# Guardrails

## Purpose
Risk-based система безопасности: оценивает риск каждого действия (LOW/MEDIUM/HIGH/CRITICAL), санитизирует параметры, требует подтверждения пользователя для опасных операций.

## Activation
- **Automatic**: вызывается Maestro в tool-loop — `guardrail_evaluate()` перед каждым действием
- **Pipeline**: `free_text_pipeline.py` вызывает `guardrail_evaluate()` для HIGH/CRITICAL действий
- **Coverage**: 60+ интентов в `_INTENT_RISK_MAP`

## Files
- `src/core/intelligence/guardrails.py` — ядро (458 строк): `ActionRisk` enum (LOW/MEDIUM/HIGH/CRITICAL), `_INTENT_RISK_MAP` (60+ интентов), `evaluate()` — главная точка: риск → санитизация → confirm, `get_confirmation_message()` (30+ форматтеров), `sanitize_action()` (чистка по allowed keys), `GuardrailResult`
- `src/core/intelligence/maestro.py` — вызывает `guardrail_evaluate()` в tool-loop; при `needs_confirm=True` возвращает подтверждение пользователю
- `src/bot/handlers/free_text_pipeline.py` — вызывает `guardrail_evaluate()` для опасных действий
- `src/core/actions/mcp_telegram.py` — caller должен спросить пользователя (guardrail)
- `src/core/actions/action_registry.py` — `SAFE_KEYS`, `action_registry` — используются для санитизации

## Tools
- `guardrail_evaluate(action, params)` (security, HIGH) — оценка риска + санитизация + confirm
- `sanitize_action(action, params)` (security, MEDIUM) — чистка параметров
- `get_confirmation_message(action, params)` (UX, LOW) — человекочитаемое подтверждение

## Output
`GuardrailResult`: risk_level, needs_confirm (bool), confirm_message (str), sanitized_params (dict). При `needs_confirm=True` — бот спрашивает пользователя «Точно ли?».

## Dependencies
- **Env**: none — pure Python
- **DB**: none — stateless
- **Services**: none — все данные в коде/конфиге
