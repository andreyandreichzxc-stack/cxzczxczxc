"""
Tests for _extract_slug_hint() in src.core.infra.formatting.
"""

import os
import sys

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.infra.formatting import _extract_slug_hint


SLUG_TEST_CASES = [
    ("https://github.com/user/repo/issues/123", "repo#123", "GitHub issue"),
    ("https://github.com/user/repo/pull/456", "repo#456", "GitHub PR"),
    ("https://github.com/user/repo", "repo", "GitHub repo"),
    ("https://github.com/user", "user", "GitHub user profile"),
    ("https://docs.python.org/3/library/re.html", "re", "Docs .html"),
    (
        "https://medium.com/@user/why-python-is-great-abc123",
        "why python is great abc123",
        "Medium article",
    ),
    (
        "https://reddit.com/r/Python/comments/abc/title",
        "r/Python",
        "Reddit subreddit",
    ),
    (
        "https://stackoverflow.com/questions/12345/how-to-fix-bug",
        "how to fix bug",
        "StackOverflow question",
    ),
    ("https://pypi.org/project/requests/", "requests", "PyPI project"),
    (
        "https://avito.ru/moskva/telefony/iphone_15_123456",
        "iphone 15 123456",
        "Avito listing with underscores",
    ),
    ("https://example.com/", "example", "Empty path -> netloc fallback"),
    (
        "https://youtube.com/watch?v=dQw4w9WgXcQ",
        "youtu.be: dQw4w9WgXcQ",
        "YouTube video ID from query param",
    ),
]


@pytest.mark.parametrize(
    ("url", "expected", "_description"),
    SLUG_TEST_CASES,
)
def test_extract_slug_hint_cases(
    url: str, expected: str, _description: str
) -> None:
    assert _extract_slug_hint(url) == expected


def run_case(url: str, expected: str | None, description: str = "") -> bool:
    """Run one test case. expected=None means just report, don't judge."""
    try:
        result = _extract_slug_hint(url)
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        return False

    if expected is None:
        print(f"  INFO: {url}")
        print(f"    -> {result!r}")
        return True  # Not a pass/fail, just informational

    if result == expected:
        print(f"  PASS: {url}")
        print(f"    -> {result!r}")
        return True
    else:
        print(f"  FAIL: {url}")
        print(f"    expected: {expected!r}")
        print(f"    got:      {result!r}")
        return False


def main():
    print("=" * 72)
    print("Testing _extract_slug_hint() -- URL -> slug extraction")
    print("=" * 72)

    passed = 0
    failed = 0

    for url, expected, desc in SLUG_TEST_CASES:
        print(f"\n{[desc]}:")
        if run_case(url, expected):
            passed += 1
        else:
            failed += 1

    # -- Edge cases / informational tests ------------------------------
    print(f"\n{'-' * 72}")
    print("EDGE CASES (informational -- no expected value)")
    print(f"{'-' * 72}")

    edge_urls = [
        ("https://t.me/somechannel", "Telegram shortener domain (t.me)"),
        ("https://habr.com/ru/articles/123456/", "Habr numeric slug"),
        ("https://example.com/index.html", "index.html -- should skip 'index'"),
        ("https://example.com/short/ab", "Very short last segment (2 chars)"),
        ("https://docs.python.org/3/library/", "Trailing slash, no filename"),
        ("https://github.com/user/repo/discussions/789", "GitHub discussions"),
        ("https://reddit.com/r/rust", "Reddit just subreddit"),
        ("https://example.com", "Bare domain, no path at all"),
        ("https://sub.domain.co.uk/path/to/page", "Multi-level TLD netloc fallback?"),
    ]

    for url, desc in edge_urls:
        try:
            result = _extract_slug_hint(url)
            print(f"\n  [{desc}]")
            print(f"    URL:  {url}")
            print(f"    slug: {result!r}")
        except Exception as e:
            print(f"\n  [{desc}] EXCEPTION: {e}")

    # -- Summary -------------------------------------------------------
    print(f"\n{'=' * 72}")
    print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed} asserts")
    if failed == 0:
        print("All required tests PASSED!")
    else:
        print(f"WARNING: {failed} test(s) FAILED!")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
