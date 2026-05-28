"""Dynamic HTTP headers mimicking Chrome."""

from .fingerprint import Fingerprint


def build_headers(
    fp: Fingerprint,
    referer: str = "https://www.google.com/",
) -> dict[str, str]:
    """Build a full set of Chrome-like headers from a fingerprint.

    Args:
        fp: Fingerprint profile (UA, sec-ch-ua, platform, languages).
        referer: Referer header value. Defaults to Google.

    Returns:
        Dictionary of HTTP headers ready to pass to httpx/requests.
    """
    return {
        "User-Agent": fp.user_agent,
        "Sec-Ch-Ua": fp.sec_ch_ua,
        "Sec-Ch-Ua-Mobile": fp.sec_ch_ua_mobile,
        "Sec-Ch-Ua-Platform": fp.sec_ch_ua_platform,
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;"
            "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
        ),
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": fp.languages,
        "Cache-Control": "max-age=0",
        "DNT": "1",
        "Referer": referer,
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "cross-site" if "google" in referer else "same-origin",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
    }
