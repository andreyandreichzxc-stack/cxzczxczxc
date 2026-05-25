# Context Files

## Purpose
LLM-WIKI: Markdown-файлы в `data/contexts/` с per-contact знаниями и `_owner.md` для владельца. FTS5-индексация для быстрого поиска. Инжектится в system-промпт Maestro.

## Activation
- **Automatic**: `find_relevant_contexts()` — вызывается при каждом запросе через prompt_assembler; ищет контекст по имени контакта в сообщении
- **Manual**: `search_contexts_tool` — доступен через tool_registry (action="search"/"list")
- **Startup**: `index_contexts_to_fts()`, `init_owner_context()` — при запуске бота

## Files
- `src/core/memory/context_files.py` — ядро: `get_contact_context()`, `save_contact_context()`, `find_relevant_contexts()`, LLM-WIKI API (`save_context`, `get_context`, `append_to_context`, `search_in_contexts`, `list_context_files`), FTS5: `_fts5_simple_query()`, `index_contexts_to_fts()`, `init_owner_context()`
- `src/core/actions/search_contexts_tool.py` — инструмент `search_contexts(action="search"/"list")`
- `src/core/context/providers/wiki_context_provider.py` — провайдер LLM-WIKI для ContextEngine
- `src/core/intelligence/prompt_assembler.py` — использует `find_relevant_contexts()` и `get_context()`
- `src/core/intelligence/character_evolution.py` — `try_extract_context_updates()`
- `src/bot/handlers/contact_cmd.py` — использует `get_contact_context()` в `/contact`
- `src/main.py` (bot) — инициализация FTS5 и owner-контекста
- `data/contexts/_owner.md` — шаблон контекста владельца

## Tools
- `search_contexts` (data, LOW) — поиск/список LLM-WIKI файлов через tool_registry

## Output
Инжектированные в system-промпт знания: «Контекст о контакте X: ... О владельце: ...». Пользователь также может запросить через `/contact` или `search_contexts`.

## Dependencies
- **DB**: FTS5 таблица `context_files_fts`
- **Files**: `data/contexts/` — Markdown-файлы
- **No env vars required** — полностью локально
