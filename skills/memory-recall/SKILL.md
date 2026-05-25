# Memory Recall

## Purpose
Единая точка входа в подсистему памяти: комбинирует Qdrant (семантический поиск), FTS5 (полнотекстовый поиск) и MMR-реранкинг для максимально релевантного и разнообразного recall.

## Activation
- **Manual**: `/memory <запрос>` — ручной поиск по фактам
- **Automatic**: вызывается Maestro, auto-reply, draft_suggester, prompt_assembler при каждом запросе
- **Free-text**: интенты для операций с памятью

## Files
- `src/core/memory/memory_recall.py` — `recall()` — единый вход: contact facts + self facts + Qdrant semantic + pinned + fresh + task-context; `_mmr_rerank()` — Maximal Marginal Relevance; `format_recall_for_prompt()`
- `src/core/memory/hybrid_search.py` — `reciprocal_rank_fusion()` — комбинирует Qdrant (cosine) + FTS5 (BM25) через RRF (k=60)
- `src/core/actions/vector_store.py` — `VectorStore` (Qdrant Embedded): коллекция messages, `search()`, `upsert()`, `rebuild()`
- `src/agents/memory_agent.py` — агент памяти для Maestro
- `src/core/actions/recall_memory_tool.py` — инструмент `recall_memory` для tool_registry
- `src/bot/handlers/memory_cmd.py` — `/memory` обработчик
- `src/bot/handlers/free_text_memory.py` — free-text операции с памятью
- `src/db/models/_memory.py` — модель `Memory` (факты)
- `src/db/repo.py` — `search_memories()` (FTS5), `add_memory()`
- `src/db/session.py` — FTS5: `messages_fts`, `memories_fts`
- `src/core/memory/memory_tagger.py` — авто-тегирование
- `src/core/memory/smart_memory.py` — умная память (связывание)
- `src/core/memory/memory_extractor.py` — извлечение фактов из текста

## Tools
- `recall_memory` (data, LOW) — поиск по памяти через tool_registry
- `memory_agent.recall()` (analysis, LOW) — агент памяти Maestro
- `vector_store.search()` (data, LOW) — Qdrant semantic search

## Output
Список релевантных фактов (JSON) или форматированный текст для промпта: «Ты знаешь о контакте: ... Из своей памяти: ...».

## Dependencies
- **Env**: LLM-ключ (для memory_extractor), embeddings-модель
- **DB**: `memories` таблица, `memories_fts` FTS5, Qdrant коллекция
- **Services**: Qdrant Embedded (data/qdrant/), userbot mirror (источник сообщений)
