# Auto-Reply

## Purpose
Автоматически отвечает в личных сообщениях, когда владелец оффлайн или спит. Учитывает отношения с контактом, стиль общения и память.

## Activation
- **Automatic**: Telethon event handler на входящие ЛС (userbot `attach_auto_reply()`)
- **Manual**: `/mode` — переключение режима (off/close/all), `/settings` → секция auto_reply

## Files
- `src/userbot/auto_reply.py` — основной handler: оффлайн-детекция, вызов `decide()`, генерация ответа через LLM с учётом style_profile и memory_recall. 3 тона: тёплый (друзья), деловой (коллеги), сухой (незнакомцы)
- `src/core/contacts/auto_reply_decision.py` — `decide()` — проверка: бот/группа/архив/кулдаун/спам/оффлайн/лимит → `AutoReplyChoice`
- `src/core/contacts/style_profile.py` — `style_profile_as_prompt_hint()` — подсказка стиля в промпт
- `src/core/memory/memory_recall.py` — `recall()` + `format_recall_for_prompt()` — контекст памяти
- `src/userbot/manager.py` — вызывает `attach_auto_reply()` при старте userbot
- `src/db/models/_auth.py` — `auto_reply_enabled`, `auto_reply_mode`, `auto_reply_text`, `auto_reply_cooldown_min`, `auto_reply_close_contacts`, `notify_on_auto_reply`
- `src/db/models/_contacts.py` — `last_auto_reply_at`
- `src/db/models/_messaging.py` — `auto_reply_logs`
- `src/db/repo.py` — `add_auto_reply_log()`
- `src/bot/handlers/settings.py` — настройки auto_reply, auto_mode
- `src/bot/handlers/mode_cmd.py` — `/mode` переключение режима
- `src/core/intelligence/guardrails.py` — `change_auto_mode` → MEDIUM

## Tools
- `auto_reply_decision.decide()` (decision, MEDIUM) — определяет, отвечать ли
- `auto_reply.generate()` (generation, HIGH) — генерирует ответ через LLM

## Output
Автоматическое сообщение контакту в Telegram: краткое, в стиле владельца, с контекстом из памяти.

## Dependencies
- **Env**: LLM-ключ, Telethon session
- **DB**: UserSettings (auto_reply_*), contacts (style_profile, last_auto_reply_at), memory
- **Services**: Telethon userbot, Qdrant (memory_recall), SmartCache
