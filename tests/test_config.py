"""Tests for config helpers — parse_telethon_proxy."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ.setdefault("ENCRYPTION_KEY", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=")
os.environ.setdefault("BOT_TOKEN", "test:token")
os.environ.setdefault("OWNER_TELEGRAM_ID", "123456789")

from src.config import parse_telethon_proxy


def test_parse_telethon_proxy_with_auth():
    """Proxy URL with auth returns full 6-tuple."""
    result = parse_telethon_proxy("socks5://user:pass@host:1080")
    assert result is not None
    assert result == ("socks5", "host", 1080, True, "user", "pass")


def test_parse_telethon_proxy_no_auth():
    """Proxy URL without auth returns 3-tuple."""
    result = parse_telethon_proxy("socks5://host:1080")
    assert result == ("socks5", "host", 1080)


def test_parse_telethon_proxy_http_no_auth():
    """HTTP proxy without auth defaults to port 8080."""
    result = parse_telethon_proxy("http://proxy.example.com")
    assert result == ("http", "proxy.example.com", 8080)


def test_parse_telethon_proxy_empty():
    """Empty string returns None."""
    assert parse_telethon_proxy("") is None


def test_parse_telethon_proxy_no_scheme():
    """URL without // prefix: urlparse treats user as scheme, falls back to 127.0.0.1:8080."""
    result = parse_telethon_proxy("user:pass@somehost:9150")
    # urlparse sees "user" as scheme (no // to separate), somehost is not parsed as hostname
    assert result is not None
    scheme, host, port = result
    # scheme defaults to what urlparse saw: "user"
    assert scheme == "user"
    assert host == "127.0.0.1"
    # scheme != "socks5" → default port is 8080
    assert port == 8080
