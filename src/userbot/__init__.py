"""Userbot manager — singleton access helpers."""

from src.userbot.manager import _MANAGER_SINGLETON as _mgr


def get_userbot_manager():
    """Get the singleton UserbotManager instance, or None if not initialized."""
    return _mgr


def get_active_telethon_client(telegram_id: int):
    """Get active Telethon client for a user, or None."""
    return _mgr.get_client(telegram_id) if _mgr else None
