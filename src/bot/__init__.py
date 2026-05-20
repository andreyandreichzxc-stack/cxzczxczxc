"""Bot package — re-exports the application factory.

Note: ``run_bot`` is exported lazily via ``__getattr__``
to avoid a circular import chain (notifier → bot → handlers → core → ...).
"""

__all__ = [
    "run_bot",
]


def __getattr__(name: str):
    if name == "run_bot":
        from src.bot.app import run_bot

        return run_bot
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
