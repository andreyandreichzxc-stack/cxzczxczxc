"""Telegram HTML formatting helpers.

All functions return Telegram-compatible HTML strings.
Bot uses ParseMode.HTML globally.

Supports ALL Telegram formatting options:
    - <b>bold</b>, <i>italic</i>, <u>underline</u>, <s>strikethrough</s>
    - <code>monospace</code>, <pre>code block</pre>
    - <blockquote>quote</blockquote>, <tg-spoiler>spoiler</tg-spoiler>
    - <a href="url">link</a>
"""

import re


# ── Basic formatting ──────────────────────────────────────────────────


def bold(text: str) -> str:
    return f"<b>{text}</b>"


def italic(text: str) -> str:
    return f"<i>{text}</i>"


def underline(text: str) -> str:
    return f"<u>{text}</u>"


def strikethrough(text: str) -> str:
    return f"<s>{text}</s>"


def code(text: str) -> str:
    return f"<code>{text}</code>"


def code_block(text: str, language: str = "") -> str:
    """Wrap in <pre><code class="language-..."> for syntax highlighting."""
    lang_attr = f' class="language-{language}"' if language else ""
    return f"<pre><code{lang_attr}>{text}</code></pre>"


def spoiler(text: str) -> str:
    return f"<tg-spoiler>{text}</tg-spoiler>"


def blockquote(text: str) -> str:
    return f"<blockquote>{text}</blockquote>"


def link(url: str, text: str = "") -> str:
    """Create <a href="url">text</a> — clickable link."""
    display = text or url
    return f'<a href="{url}">{display}</a>'


# ── Compound formatting ───────────────────────────────────────────────


def section(title: str, body: str) -> str:
    """Formatted section with bold title."""
    return f"\n{bold(title)}\n{body}"


def format_list(items: list[str], ordered: bool = False) -> str:
    """Bullet (•) or numbered (1.) list."""
    if not items:
        return ""
    if ordered:
        return "\n".join(f"{i}. {item}" for i, item in enumerate(items, 1))
    return "\n".join(f"• {item}" for item in items)


def format_key_value(kv: dict[str, str]) -> str:
    """Key: value pairs with bold keys."""
    return "\n".join(f"{bold(k)}: {v}" for k, v in kv.items())


def format_memory_fact(fact: str) -> str:
    """Format memory facts with italic quotes."""
    return italic(f"«{fact}»")


# ── Smart link formatting ─────────────────────────────────────────────

# Known URL shorteners — detect, don't describe
_SHORTENER_DOMAINS: frozenset[str] = frozenset(
    {
        "t.co",
        "bit.ly",
        "bitly.com",
        "goo.gl",
        "buff.ly",
        "ow.ly",
        "tinyurl.com",
        "tiny.cc",
        "is.gd",
        "clck.ru",
        "shorturl.at",
        "cutt.ly",
        "rebrand.ly",
        "shorte.st",
        "snip.ly",
        "v.gd",
        "tr.im",
        "short.link",
        "rb.gy",
        "lnkd.in",
        "short.cm",
        "zpr.io",
        "qr.ae",
        "t.me",
        "youtu.be",
        "dlvr.it",
        "vk.cc",
        "s.id",
        "gg.gg",
    }
)


def _extract_slug_hint(url: str) -> str:
    """Extract human-readable hint from URL path.

    Examples:
        github.com/user/repo/issues/123 → "repo#123"
        docs.python.org/3/library/re.html → "re"
        medium.com/@user/why-python-is-great-abc123 → "why-python-is-great"
        reddit.com/r/Python/comments/abc/title → "r/Python"
    """
    from urllib.parse import urlparse, parse_qs

    parsed = urlparse(url)
    path = parsed.path.strip("/")
    if not path:
        return parsed.netloc.split(".")[-2] if parsed.netloc else ""

    parts = path.split("/")

    # GitHub: user/repo/issues/123 → "repo#123"
    if (
        "github.com" in parsed.netloc
        and len(parts) >= 4
        and parts[2] in ("issues", "pull", "discussions")
    ):
        return f"{parts[1]}#{parts[3]}"
    if "github.com" in parsed.netloc and len(parts) >= 2:
        return parts[1] if parts[1] else parts[0]

    # YouTube: watch?v=VIDEO_ID → "youtu.be: VIDEO_ID" (short ID)
    if "youtube.com" in parsed.netloc or "youtu.be" in parsed.netloc:
        qs = parse_qs(parsed.query)
        if "v" in qs:
            vid = qs["v"][0][:11]  # YouTube IDs are 11 chars
            return f"youtu.be: {vid}"
        # youtu.be/VIDEO_ID path
        if "youtu.be" in parsed.netloc and parts:
            return f"youtu.be: {parts[0][:11]}"

    # Docs: last segment without extension
    if len(parts) == 1:
        slug = parts[0].rsplit(".", 1)[0] if "." in parts[0] else parts[0]
        if slug.lower() not in ("index", "default", "main", "home"):
            return slug
        # Fall through to netloc fallback

    # Reddit: r/Subreddit/comments/...
    if len(parts) >= 2 and (parts[0].startswith("r/") or parts[0] == "r"):
        return parts[0] if parts[0] != "r" else f"r/{parts[1]}"

    # General: use last meaningful segment
    for i, part in enumerate(reversed(parts)):
        clean = re.sub(r"[_-]", " ", part.rsplit(".", 1)[0])
        if not clean:
            continue
        # Last segment (i==0) is likely a page/file name — allow shorter names
        if clean.lower() not in ("index", "default", "main", "home"):
            # Skip purely numeric slugs if parent segments exist
            if clean.isdigit() and len(parts) > 1:
                continue
            if i == 0 or len(clean) > 2:
                return clean[:50]

    last = (
        parts[-1].rsplit(".", 1)[0]
        if parts[-1] and "." in parts[-1]
        else (parts[-1] or "")
    )
    if last.lower() in ("index", "default", "main", "home") or not last:
        return parsed.netloc.split(".")[-2] if parsed.netloc else ""
    return last[:50]


def auto_format_urls(text: str) -> str:
    """Convert bare URLs to Telegram <a> links with smart labels.

    Each URL is replaced ONCE (first occurrence).
    Shorteners get a 🔗 warning label. All others get slug-extracted labels.
    """
    if "<a " in text:
        return text

    from urllib.parse import urlparse

    URL_RE = re.compile(r'(?<!["\'>\(])(https?://[^\s<>"\)\]]+)(?!["\'<\]])')
    seen: set[str] = set()

    def _replace(match: re.Match) -> str:
        url = match.group(0)
        if url in seen:
            return url
        seen.add(url)

        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower().lstrip("www.")

            # Shortener → just link as-is
            base_domain = (
                ".".join(domain.split(".")[-2:]) if domain.count(".") > 1 else domain
            )
            if domain in _SHORTENER_DOMAINS or base_domain in _SHORTENER_DOMAINS:
                if len(url) > 50:
                    return f'<a href="{url}">🔗 ссылка</a>'
                return f'<a href="{url}">{url}</a>'

            # Smart label
            label = _extract_slug_hint(url)
            if label:
                if len(label) > 60:
                    label = label[:57] + "..."
                return f'<a href="{url}">{label}</a>'
            else:
                return f'<a href="{url}">{domain}</a>'
        except Exception:
            return url

    return URL_RE.sub(_replace, text)


# ── Auto-formatting ───────────────────────────────────────────────────


def auto_format(text: str) -> str:
    """Apply Telegram HTML formatting to plain text."""
    if any(
        tag in text
        for tag in (
            "<b>",
            "<i>",
            "<code>",
            "<u>",
            "<s>",
            "<tg-spoiler>",
            "<blockquote>",
            "<a ",
        )
    ):
        return text
    # Commands first (before URLs, to avoid <a> tag corruption)
    # (?<![\w:/]) prevents matching inside URLs (://, path segments)
    text = re.sub(
        r"(?<![\w:/])/([a-z0-9_]+(?:\s+[a-z0-9_]+)?)", r"<code>/\1</code>", text
    )
    text = auto_format_urls(text)
    text = re.sub(r"^(.+):$", r"<b>\1:</b>", text, flags=re.MULTILINE)
    text = re.sub(r"\*([^*]+)\*", r"<b>\1</b>", text)
    text = re.sub(r"_([^_]+)_", r"<i>\1</i>", text)
    text = re.sub(r"~([^~]+)~", r"<s>\1</s>", text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return text
