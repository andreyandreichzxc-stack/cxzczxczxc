"""Runtime-статистика humanizer'а."""


_stats: dict[str, int | float] = {
    "total_checks": 0,
    "total_humanized": 0,
    "avg_score_before": 0.0,
    "avg_score_after": 0.0,
}
_lock = None  # ленивый импорт asyncio.Lock


def record_check(score_before: float, score_after: float, humanized: bool) -> None:
    """Записать результат проверки. Thread-safe через simple int ops."""
    global _stats
    n = _stats["total_checks"] + 1
    _stats["total_checks"] = n
    _stats["avg_score_before"] = (
        _stats["avg_score_before"] * (n - 1) + score_before
    ) / n
    _stats["avg_score_after"] = (_stats["avg_score_after"] * (n - 1) + score_after) / n
    if humanized:
        _stats["total_humanized"] += 1


def get_stats() -> dict:
    return dict(_stats)
