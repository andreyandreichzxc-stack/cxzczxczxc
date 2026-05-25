# Search

## Purpose
Полнотекстовый поиск по всем сообщениям пользователя: `/search` ищет по всем чатам, `/timeline` показывает хронологию обсуждения темы. Использует FTS5 для быстрого поиска.

## Activation
- **Manual**: `/search <запрос>` — поиск по всем чатам, `/timeline <тема>` — хронология обсуждения
- **Agent**: `search_agent.resolve()` — нечёткий поиск контакта/чата по имени/нику
- **Automatic**: `cross_chat_search()` — используется другими модулями (через tool_registry или напрямую)

## Files
- `src/bot/handlers/search.py` — обработчик `/search`: вызывает `cross_chat_search()`, форматирует результаты по чатам
- `src/bot/handlers/timeline_cmd.py` — обработчик `/timeline`: хронология темы
- `src/db/repo.py` — `cross_chat_search()` (FTS5 по `messages_fts`, возвращает top conversations со сниппетами), `_fts_query_for()` (безопасный FTS5 MATCH), `search_memories()` (FTS5 по `memories_fts`)
- `src/core/actions/cross_search_tool.py` — инструмент `cross_chat_search` для tool_registry
- `src/agents/search_agent.py` — `resolve()` — LLM-агент поиска контакта
- `src/db/session.py` — FTS5 `messages_fts` virtual table + триггеры синхронизации
- `src/core/contacts/chat_finder.py` — поиск чата: keyword expansion (LLM) + локальный FTS5
- `src/userbot/mirror.py` — зеркало сообщений в БД и FTS5 в реальном времени
- `src/userbot/dialogs.py` — `/sync` — заполнение БД/FTS5 для холодного старта

## Tools
- `cross_chat_search` (data, LOW) — полнотекстовый поиск по всем чатам
- `search_agent.resolve()` (analysis, LOW) — нечёткий поиск контакта

## Output
Список сообщений со сниппетами, сгруппированных по чатам, с датами и ссылками. Для `/timeline` — хронологическая лента обсуждения.

## Dependencies
- **DB**: `messages_fts` FTS5 virtual table, `memories_fts`
- **Services**: userbot mirror (синхронизация сообщений), Telethon (заполнение при `/sync`)
- **No env vars required**
