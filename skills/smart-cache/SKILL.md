# Smart Cache

## Purpose
Трёхуровневый умный кеш: L0 (in-memory OrderedDict, 500 записей) → L1 (SQLite SmartCacheEntry, 10k/owner) → L2 (Memory.fact). Автоматический scoring, graduation, anti-bloat.

## Activation
- **Automatic**: каждый вызов `SmartCache.get()`/`.put()`/`.bump()` — прозрачно для вызывающего кода
- **Background**: cleanup stale entries, decay scoring

## Files
- `src/core/cache/smart_cache.py` — основной класс: `SmartCache` с методами `get()`, `put()`, `bump()` (L0→L1→L2 promotion), `SOURCE_WEIGHTS`, `GRADUATION_MAX_PER_OWNER_PER_DAY=50`, `GLOBAL_MAX_GRADUATIONS=100k`
- `src/db/models/_cache.py` — модель `SmartCacheEntry` (cache_key, content_hash, importance_score, access_count, graduated, source)
- `src/core/context_cache.py` — in-memory TTL-кэш (синглтон): `get()`, `put()`, `invalidate(prefix)`, async-safe с `asyncio.Lock`
- `src/core/intelligence/agent_cache.py` — кеш результатов агентов
- `src/core/intelligence/pattern_cache.py` — кеш паттернов
- `src/core/actions/embedding_cache.py` — кеш эмбеддингов
- `src/core/actions/stats_cache.py` — кеш статистики
- `src/db/models/_embedding_cache.py` — модель для кеша эмбеддингов

## Tools
- `SmartCache.get(key, owner_id)` (data, LOW) — чтение из кеша
- `SmartCache.put(key, value, source, owner_id)` (data, LOW) — запись с авто-promotion
- `context_cache.get(key)` / `context_cache.put(key, value, ttl)` (data, LOW) — быстрый in-memory кеш

## Output
Прозрачное ускорение: cache hit — мгновенный ответ, cache miss — вычисление + сохранение. Логируются graduation events.

## Dependencies
- **DB**: таблица `smart_cache_entries`, `embedding_cache`
- **Services**: Memory.fact (L2 storage)
- **No env vars required** — работает автономно
