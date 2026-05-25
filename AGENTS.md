# AGENTS.md — Системный контракт для AI-агента

## Как работает память

Бот НЕ читает файлы вручную. Для доступа к памяти используй MCP tools:
- `recall_memory` — семантический + ключевой поиск по фактам
- `search_contexts` — гибридный поиск (FTS5 + Qdrant) по LLM-WIKI
- `cross_chat_search` — поиск по истории переписок

Frozen snapshot (топ-3 релевантных факта) уже в system prompt.
Если нужна детальная информация — вызывай `recall_memory`.

## Как бот пишет в память

- Факты о пользователе → `add_memory` (через memory_queue)
- Факты о контакте → `add_memory` с contact_id
- Контекстные заметки → `search_contexts` tool (LLM-WIKI)

## Как бот использует контекст

- System prompt получает: SOUL.md (личность) + frozen snapshot (топ-3 факта)
- При self-reference ("расскажи обо мне") → инжектится `_owner.md`
- При упоминании контакта ("что с Олей?") → инжектится `оля.md`
- Contact digest кеширует факты + обещания + риски на 1 час

## Ночной цикл (dream cycle)

- 03:00 — decay + tier promotion/demotion
- Каждые 6ч — консолидация дубликатов
- 09:00 — утренний брифинг
- Каждые 6ч — burnout detection

## Tools

Все MCP tools зарегистрированы через `@tool` декоратор.
Доступны: recall_memory, search_contexts, cross_chat_search,
mcp_web, mcp_http, mcp_telegram, mcp_reminders, mcp_filesystem,
mcp_system, search_messages, summarize_chat, draft_reply,
set_reminder, list_contacts, execute_code.

## Контракт безопасности

- guardrails.py: LOW=разрешено, MEDIUM=подтверждение для новых, HIGH=всегда подтверждение, CRITICAL=всегда подтверждение
- SDD executor: только allowlist AST, нет доступа к DB/session/user
- Pairing: незнакомцы не получают auto-reply без /approve
- SSRF: блокируются private IP, localhost, metadata endpoint
