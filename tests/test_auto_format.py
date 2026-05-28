"""Tests for auto_format() and auto_format_urls() in src.core.infra.formatting."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("ENCRYPTION_KEY", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=")
os.environ.setdefault("BOT_TOKEN", "test:token")
os.environ.setdefault("OWNER_TELEGRAM_ID", "123456789")

from src.core.infra.formatting import auto_format, auto_format_urls

# ── Helpers ──────────────────────────────────────────────────────────────

_passed = 0
_failed = 0


def check(test_name: str, actual: str, expected_predicate, desc: str = "") -> bool:
    """Check actual result against expected_predicate (string or callable)."""
    global _passed, _failed
    if callable(expected_predicate):
        ok = expected_predicate(actual)
        expected_repr = "<predicate>"
    else:
        ok = actual == expected_predicate
        expected_repr = repr(expected_predicate)
    if ok:
        _passed += 1
        status = "PASS"
    else:
        _failed += 1
        status = "FAIL"
    extra = f"  ({desc})" if desc else ""
    print(f"[{status}] {test_name}{extra}")
    if not ok:
        print(f"       expected: {expected_repr}")
        print(f"       actual:   {repr(actual)}")
    return ok


# ── auto_format() tests ──────────────────────────────────────────────────


def test_auto_format_simple_command():
    """Basic /start command wrapping."""
    result = auto_format("/start")
    check(
        "auto_format: /start -> <code>/start</code>",
        result,
        lambda r: "<code>/start</code>" in r,
    )


def test_auto_format_help_command():
    result = auto_format("/help")
    check(
        "auto_format: /help -> <code>/help</code>",
        result,
        lambda r: "<code>/help</code>" in r,
    )


def test_auto_format_two_commands():
    result = auto_format("Привет, /sync и /keys import")
    check(
        "auto_format: /sync wrapped",
        result,
        lambda r: "<code>/sync</code>" in r,
    )
    check(
        "auto_format: /keys import wrapped as single command",
        result,
        lambda r: "<code>/keys import</code>" in r,
    )


def test_auto_format_url_standalone():
    """URL without commands -- check URL is formatted AND integrity."""
    result = auto_format("Check https://github.com/user/repo")
    check(
        "auto_format: URL -> <a> tag present",
        result,
        lambda r: "<a " in r,
    )
    check(
        "auto_format: URL tag has closing </a>",
        result,
        lambda r: "</a>" in r,
    )
    check(
        "auto_format: has correct link href",
        result,
        lambda r: "https://github.com/user/repo" in r,
    )
    # Check for corruption: /user or /repo inside href should NOT become <code>
    check(
        "auto_format: /user inside href NOT wrapped in <code>",
        result,
        lambda r: "<code>/user</code>" not in r and "<code>/repo</code>" not in r,
    )


def test_auto_format_url_with_command():
    """URL + command same line -- test coexistence behavior."""
    result = auto_format("/start и ссылка https://example.com/page")
    check(
        "auto_format: /start wrapped in <code>",
        result,
        lambda r: "<code>/start</code>" in result,
    )
    check(
        "auto_format: URL href present",
        result,
        lambda r: "https://example.com/page" in result,
    )
    # Verify integrity: no URL path corruption
    url_path_wrapped = (
        "<code>/example</code>" in result or "<code>/page</code>" in result
    )
    if url_path_wrapped:
        print("       BUG: URL path corrupted by command regex!")
        print(f"       actual: {repr(result)}")


def test_auto_format_already_formatted():
    """Text with existing HTML tags must NOT be re-formatted (early return)."""
    check(
        "auto_format: already formatted -- early return (no change)",
        auto_format("уже <b>отформатировано</b>"),
        "уже <b>отформатировано</b>",
    )


def test_auto_format_tme_shortener():
    result = auto_format("http://t.me/channel")
    # t.me is a shortener: URL itself should appear (either as-is or as label)
    check(
        "auto_format: t.me shortener -> has <a> tag",
        result,
        lambda r: "<a " in r,
    )
    check(
        "auto_format: t.me shortener -> href contains t.me",
        result,
        lambda r: 'href="http://t.me/channel"' in r,
    )


def test_auto_format_youtube_shortener():
    result = auto_format("http://youtu.be/abc123")
    check(
        "auto_format: youtu.be shortener -> has <a> tag",
        result,
        lambda r: "<a " in r,
    )


def test_auto_format_bitly_shortener():
    result = auto_format("http://bit.ly/xyz")
    check(
        "auto_format: bit.ly shortener -> has <a> tag",
        result,
        lambda r: "<a " in r,
    )


# ── auto_format_urls() tests (standalone, no command regex interference) ─


def test_auto_format_urls_parentheses():
    """URL in parentheses: (https://example.com/page)
    The URL regex has '(' in the lookbehind exclusion, so it will NOT match.
    This is documented behavior."""
    result = auto_format_urls("см. тут (https://example.com/page)")
    # Since '(' precedes the URL, the regex does NOT match. Text stays as-is.
    check(
        "auto_format_urls: parentheses -- URL preceded by ( --> not matched",
        result,
        lambda r: "<a " not in r,  # No URL formatted
    )


def test_auto_format_urls_parentheses_no_left_paren():
    """URL preceded by space but followed by ) -- should match URL without )."""
    result = auto_format_urls("see https://example.com/page) and more")
    check(
        "auto_format_urls: URL ) excluded from URL",
        result,
        lambda r: 'href="https://example.com/page"' in r,
    )
    check(
        "auto_format_urls: ) preserved after link",
        result,
        lambda r: r.endswith("</a>) and more") or ")" in r,
    )


def test_auto_format_urls_followed_by_angle():
    """URL followed by < -- < is excluded from URL char class."""
    result = auto_format_urls("see https://example.com <-- there")
    check(
        "auto_format_urls: URL followed by <",
        result,
        lambda r: 'href="https://example.com"' in r,
    )


def test_auto_format_urls_multiple_unique_labels():
    """Multiple URLs -- each gets its own label."""
    result = auto_format_urls(
        "a https://github.com/user/repo b https://docs.python.org/3/library/re.html c"
    )
    check(
        "auto_format_urls: two unique URLs -> each wrapped in <a>",
        result,
        lambda r: result.count("<a ") == 2,
    )
    check(
        "auto_format_urls: labels are different (repo vs re)",
        result,
        lambda r: "repo</a>" in r and "re</a>" in r,
    )


def test_auto_format_urls_same_url_twice():
    """Same URL twice -- first formatted, second left bare."""
    url = "https://example.com/page"
    result = auto_format_urls(f"see {url} and also {url}")
    check(
        "auto_format_urls: first URL wrapped",
        result,
        lambda r: result.count(f'href="{url}"') == 1,
    )
    check(
        "auto_format_urls: second URL left bare",
        result,
        lambda r: result.count(url) == 2,  # once in href, once bare
    )


def test_auto_format_urls_already_formatted():
    result = auto_format_urls('already <a href="https://x.com">formatted</a> text')
    check(
        "auto_format_urls: already has <a> -- early return unchanged",
        result,
        'already <a href="https://x.com">formatted</a> text',
    )


def test_auto_format_urls_no_url():
    result = auto_format_urls("просто текст без ссылок")
    check(
        "auto_format_urls: no URL -- unchanged",
        result,
        "просто текст без ссылок",
    )


def test_auto_format_urls_with_ampersand():
    result = auto_format_urls("https://example.com?a=1&b=2")
    check(
        "auto_format_urls: URL with params formatted",
        result,
        lambda r: '<a href="https://example.com?a=1&b=2">' in r,
    )


def test_auto_format_urls_after_comma():
    result = auto_format_urls("вот, https://example.com -- ссылка")
    check(
        "auto_format_urls: URL after comma formatted",
        result,
        lambda r: '<a href="https://example.com">' in r,
    )


# ── Command regex edge cases ─────────────────────────────────────────────


def test_command_regex_api_url_no_match():
    # In http://api.telegram.org/bot123/getUpdates:
    # /getUpdates is preceded by digit (3), which is a word char, so NO match.
    """Command regex should NOT match /getUpdates inside API URL."""
    result = auto_format("http://api.telegram.org/bot123/getUpdates")
    check(
        "command regex: /getUpdates NOT wrapped (preceded by digit)",
        result,
        lambda r: "<code>/getUpdates</code>" not in r,
    )


def test_command_regex_fraction_no_match():
    result = auto_format("100/200")
    check(
        "command regex: /200 NOT wrapped (preceded by digits)",
        result,
        lambda r: "<code>/200</code>" not in r and "100/200" in r,
    )


def test_command_regex_alone():
    check(
        "command regex: /alone wrapped",
        auto_format("/alone"),
        lambda r: "<code>/alone</code>" in r,
    )


def test_command_regex_single_char():
    check(
        "command regex: /a wrapped",
        auto_format("/a"),
        lambda r: "<code>/a</code>" in r,
    )


def test_command_regex_underscore_command():
    check(
        "command regex: /my_command wrapped",
        auto_format("/my_command"),
        lambda r: "<code>/my_command</code>" in r,
    )


# ── URL + command coexistence with auto_format ──────────────────────────


def test_url_then_command_coexist():
    """URL first, then command."""
    result = auto_format("visit https://example.com/page then /start")
    check(
        "coexistence: URL href present",
        result,
        lambda r: "https://example.com/page" in r,
    )
    check(
        "coexistence: /start wrapped",
        result,
        lambda r: "<code>/start</code>" in r,
    )


def test_command_then_url_coexist():
    """Command first, then URL."""
    result = auto_format("/help at https://docs.example.com/guide")
    check(
        "coexistence: /help at wrapped as single command",
        result,
        lambda r: "<code>/help at</code>" in r,
    )
    check(
        "coexistence: URL href present",
        result,
        lambda r: "https://docs.example.com/guide" in r,
    )


def test_url_command_same_line():
    result = auto_format("https://mistral.ai/news /news")
    check(
        "coexistence same line: URL href present",
        result,
        lambda r: "https://mistral.ai/news" in r,
    )
    check(
        "coexistence same line: /news wrapped",
        result,
        lambda r: "<code>/news</code>" in r,
    )


# ── Closing tag integrity (CRITICAL!) ───────────────────────────────────


def test_closing_a_tag_not_corrupted():
    """After auto_format_urls produces <a>...</a>, the command regex
    must NOT match /a inside </a> or path segments inside href."""
    result = auto_format("Go to https://example.com/path now /start")
    check(
        "integrity: </a> not corrupted into <code>/a</code>",
        result,
        lambda r: "<code>/a</code>" not in r,
    )
    check(
        "integrity: href path /path not wrapped",
        result,
        lambda r: "<code>/path</code>" not in r,
    )
    # If there's corruption, report the actual output
    if "<code>/a</code>" in result or "<code>/path</code>" in result:
        print("       BUG: URL tag corrupted by command regex!")
        print(f"       actual: {repr(result)}")


def test_shortener_with_command():
    result = auto_format("http://bit.ly/xyz and /help")
    check(
        "shortener+cmd: bit.ly href intact",
        result,
        lambda r: "http://bit.ly/xyz" in r,
    )
    check(
        "shortener+cmd: /help wrapped",
        result,
        lambda r: "<code>/help</code>" in r,
    )
    # Check for </a> corruption
    if "<code>/a</code>" in result:
        print("       BUG: </a> corrupted in shortener+command test!")
        print(f"       actual: {repr(result)}")


# ── Summary ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    tests = [
        # auto_format basic commands
        test_auto_format_simple_command,
        test_auto_format_help_command,
        test_auto_format_two_commands,
        test_auto_format_url_standalone,
        test_auto_format_url_with_command,
        test_auto_format_already_formatted,
        test_auto_format_tme_shortener,
        test_auto_format_youtube_shortener,
        test_auto_format_bitly_shortener,
        # auto_format_urls standalone
        test_auto_format_urls_parentheses,
        test_auto_format_urls_parentheses_no_left_paren,
        test_auto_format_urls_followed_by_angle,
        test_auto_format_urls_multiple_unique_labels,
        test_auto_format_urls_same_url_twice,
        test_auto_format_urls_already_formatted,
        test_auto_format_urls_no_url,
        test_auto_format_urls_with_ampersand,
        test_auto_format_urls_after_comma,
        # command regex edge cases
        test_command_regex_api_url_no_match,
        test_command_regex_fraction_no_match,
        test_command_regex_alone,
        test_command_regex_single_char,
        test_command_regex_underscore_command,
        # coexistence
        test_url_then_command_coexist,
        test_command_then_url_coexist,
        test_url_command_same_line,
        # integrity (CRITICAL)
        test_closing_a_tag_not_corrupted,
        test_shortener_with_command,
    ]

    for t in tests:
        try:
            t()
        except Exception:
            _failed += 1
            print(f"[FAIL] {t.__name__}  (EXCEPTION)")
            traceback.print_exc()

    total = _passed + _failed
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {_passed}/{total} passed, {_failed}/{total} failed")
    print(f"{'=' * 60}")
