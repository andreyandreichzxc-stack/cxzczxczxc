"""Core package — re-exports cross-cutting utilities.

Note: ``notification_queue`` is exported lazily via ``__getattr__``
to avoid a circular import chain (core → notifier → bot → handlers → core).
"""

from src.core.infra.text_sanitizer import sanitize_html
from src.core.infra.timeutil import HM_RE, get_user_tz, now_in_tz

__all__ = [
    "sanitize_html",
    "now_in_tz",
    "get_user_tz",
    "HM_RE",
    "notification_queue",
]


def __getattr__(name: str):
    if name == "notification_queue":
        from src.core.scheduling.notification_queue import notification_queue

        return notification_queue
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
