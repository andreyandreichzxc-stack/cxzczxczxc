# News Digest

## Purpose
Собирает посты из подписанных Telegram-каналов, семантически фильтрует по теме и генерирует сводку через LLM.

## Activation
- **Manual**: `/news <тема> [--hours=24]`, `/news_channels`, `/news_topics`
- **Automatic**: `news_scheduler_loop()` — фоновый цикл, рассылает дайджест по расписанию (`news_digest_time`), если `news_enabled=True`
- **Free-text**: интент `news_digest` (фразы «что пишут про X», «новости по Y»)

## Files
- `src/bot/handlers/news_cmd.py` — обработчики `/news`, `/news_channels`
- `src/bot/handlers/news_topics.py` — управление темами авто-новостей
- `src/core/scheduling/news.py` — `build_news_digest()`, `news_scheduler_loop()`, `NEWS_SYSTEM`
- `src/agents/digest_agent.py` — `build_digest()` (агент для входящих сообщений)
- `src/bot/handlers/free_text_exec.py` — `exec_classic_news_digest()`
- `src/bot/handlers/free_text_common.py` — маппинг kind → параметры
- `src/bot/handlers/free_text_pipeline.py` — регистрация в пайплайне
- `src/core/intelligence/guardrails.py` — `news_digest` → ActionRisk.LOW
- `src/core/intelligence/agent.py` — спецификация интента `news_digest`
- `src/db/models/_auth.py` — `news_digest_time`, `news_enabled`
- `src/db/models/_contacts.py` — `is_news_source` (каналы-источники)
- `src/bot/handlers/settings.py` — настройки в `/settings`

## Tools
- `build_news_digest` (information, LOW) — сбор и анализ новостей по теме
- `news_digest` intent (action, LOW) — free-text вызов дайджеста

## Output
Telegram-сообщение со сводкой: заголовок темы, количество источников, структурированный обзор с ссылками на каналы.

## Dependencies
- **Env**: LLM-ключ (OpenRouter/Gemini/etc.)
- **DB**: `news_digest_time`, `news_enabled`, `is_news_source`
- **Services**: Telethon userbot (чтение каналов), Qdrant (cosine-фильтр embeddings)
