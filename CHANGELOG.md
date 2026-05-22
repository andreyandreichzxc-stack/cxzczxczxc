# Changelog

## v2.0 — May 2026
- 🛡️ ~110 crash bugs fixed: NameError, AttributeError, TypeError, UnboundLocalError
- 🔒 20+ XSS/HTML-injection fixes: sanitize_html() everywhere
- ⚡ 6 performance optimizations: batch embeddings, batch Qdrant upserts, N+1→2 SQL
- 🎭 Humanizer: anti-AI block + /humanize command — бот говорит живо
- 🧠 Style matcher: динамически подстраивается под стиль юзера
- 🔮 Skill Creator: автономно предлагает новые навыки
- 💾 Persistent embedding cache: эмбеддинги не теряются после рестарта
- 🔗 Pattern cache: частые intent→action без LLM
- 🧬 Character evolution: бот саморазвивается
- 📦 SmartCache: 3-уровневый кэш (L0 memory → L1 SQLite → L2 Memory)
- 🏗️ 14 race conditions fixed: asyncio.Lock повсюду
- 🧹 7 memory leaks plugged: бесконечно растущие словари
- 🛑 FTS5 injection fix: операторы экранируются
- 📋 /humanize fix <текст> — чистка AI-шаблонов
- 🗃️ Alembic миграции для всех новых таблиц

## v1.0 — Original
- Base Telegram bot with LLM integration
- Memory system with SQLite + Qdrant
- Maestro agent orchestrator
- Smart AutoRouter
- Skills system
