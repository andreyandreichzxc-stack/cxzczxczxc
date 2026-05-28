"""Built-in web connectors for GitHub, X, and 4PDA."""

from __future__ import annotations

import asyncio
import mimetypes
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from .base import (
    ConnectorActionAnnotations,
    ConnectorActionSpec,
    ConnectorResult,
    ConnectorRuntime,
    ConnectorSpec,
)
from .credentials import redact_url_secrets
from .http import ConnectorHttpError, MAX_REDIRECTS, validate_public_url
from .registry import ConnectorRegistry
from .sanitize import sanitize_text


USER_AGENT = "asist-connectors/1.0 (+https://github.com/)"
DOWNLOAD_ROOT = Path("data/connectors").resolve()
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024
SENSITIVE_REDIRECT_HEADERS = {"authorization", "cookie"}
GITHUB_API_HOSTS = {"api.github.com"}
GITHUB_DOWNLOAD_HOSTS = {
    "github.com",
    "objects.githubusercontent.com",
    "release-assets.githubusercontent.com",
    "github-releases.githubusercontent.com",
}
FOURPDA_HOSTS = {"4pda.to", "www.4pda.to"}
X_API_HOSTS = {"api.x.com"}
X_MEDIA_HOSTS = {"pbs.twimg.com", "video.twimg.com"}


def register_site_connectors(registry: ConnectorRegistry) -> None:
    for spec, handler in (
        (_github_spec(), _github_handler),
        (_fourpda_spec(), _fourpda_handler),
        (_x_spec(), _x_handler),
    ):
        if registry.get(spec.name) is None:
            registry.register(spec, handler)


def _standard_actions(*, attachments_auth_note: str = "") -> tuple[ConnectorActionSpec, ...]:
    return (
        ConnectorActionSpec(
            name="search_topics",
            description="Search topics, repositories, issues, posts, or conversations.",
            risk="low",
            annotations=ConnectorActionAnnotations(title="Search Topics", read_only=True),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        ),
        ConnectorActionSpec(
            name="read_topic",
            description="Read a topic/thread/repository/page by URL or source-specific id.",
            risk="low",
            annotations=ConnectorActionAnnotations(title="Read Topic", read_only=True),
            input_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "id": {"type": "string"},
                    "limit": {"type": "integer", "default": 20},
                },
            },
        ),
        ConnectorActionSpec(
            name="read_post",
            description="Read a single post/comment/item by URL or id.",
            risk="low",
            annotations=ConnectorActionAnnotations(title="Read Post", read_only=True),
            input_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "id": {"type": "string"},
                    "repo": {"type": "string"},
                },
            },
        ),
        ConnectorActionSpec(
            name="download_attachments",
            description=f"Download attachments/assets/media into data/connectors. {attachments_auth_note}".strip(),
            risk="high",
            requires_confirmation=True,
            annotations=ConnectorActionAnnotations(
                title="Download Attachments",
                read_only=False,
                destructive=False,
                idempotent=False,
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "id": {"type": "string"},
                    "repo": {"type": "string"},
                    "limit": {"type": "integer", "default": 10},
                },
            },
        ),
    )


def _github_spec() -> ConnectorSpec:
    return ConnectorSpec(
        name="github",
        description="GitHub connector using public REST APIs and optional GITHUB_TOKEN.",
        category="developer",
        auth_mode="api_key",
        docs_url="https://docs.github.com/rest",
        capabilities=("search_topics", "read_topic", "read_post", "download_attachments"),
        actions=_standard_actions(attachments_auth_note="Downloads release assets when a repo/release URL is provided."),
    )


def _fourpda_spec() -> ConnectorSpec:
    return ConnectorSpec(
        name="4pda",
        description="4PDA forum connector for public forum pages; FOURPDA_COOKIE can help with anti-bot/auth pages.",
        category="forum",
        auth_mode="cookie",
        docs_url="https://4pda.to/forum/index.php?act=idx",
        capabilities=("search_topics", "read_topic", "read_post", "download_attachments"),
        actions=_standard_actions(attachments_auth_note="Some attachments require logged-in FOURPDA_COOKIE."),
    )


def _x_spec() -> ConnectorSpec:
    return ConnectorSpec(
        name="x",
        description="X/Twitter connector using X API v2. Set X_BEARER_TOKEN for search/read operations.",
        category="social",
        auth_mode="api_key",
        docs_url="https://developer.x.com/en/docs/x-api",
        capabilities=("search_topics", "read_topic", "read_post", "download_attachments"),
        actions=_standard_actions(attachments_auth_note="Downloads media URLs returned by X API when available."),
    )


def _limit(params: dict[str, Any], default: int = 10, maximum: int = 50) -> int:
    try:
        value = int(params.get("limit") or default)
    except (TypeError, ValueError):
        value = default
    return max(1, min(maximum, value))


def _headers(token_env: str | None = None, cookie_env: str | None = None) -> dict[str, str]:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json, text/html;q=0.9, */*;q=0.8"}
    if token_env and os.getenv(token_env):
        headers["Authorization"] = f"Bearer {os.environ[token_env]}"
    if cookie_env and os.getenv(cookie_env):
        headers["Cookie"] = os.environ[cookie_env]
    return headers


def _validate_connector_url(url: str, *, allowed_hosts: set[str] | None = None) -> str:
    validated = validate_public_url(url)
    host = (urlparse(validated).hostname or "").lower()
    if allowed_hosts and host not in allowed_hosts:
        raise ConnectorHttpError(f"Host is not allowed for this connector: {host}")
    return validated


def _redirect_headers(headers: dict[str, str], source_url: str, target_url: str) -> dict[str, str]:
    if (urlparse(source_url).hostname or "").lower() == (urlparse(target_url).hostname or "").lower():
        return headers
    return {key: value for key, value in headers.items() if key.lower() not in SENSITIVE_REDIRECT_HEADERS}


async def _fetch_response(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    allowed_hosts: set[str] | None = None,
    timeout: float = 25.0,
) -> httpx.Response:
    current_url = _validate_connector_url(url, allowed_hosts=allowed_hosts)
    current_params = params
    current_headers = headers or _headers()

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        for _ in range(MAX_REDIRECTS + 1):
            response = await client.get(current_url, params=current_params, headers=current_headers)
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    raise ConnectorHttpError("Redirect response has no Location header")
                next_url = _validate_connector_url(urljoin(str(response.url), location), allowed_hosts=allowed_hosts)
                current_headers = _redirect_headers(current_headers, str(response.url), next_url)
                current_url = next_url
                current_params = None
                continue
            return response

    raise ConnectorHttpError("Too many redirects")


async def _fetch_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    allowed_hosts: set[str] | None = None,
) -> Any:
    response = await _fetch_response(url, params=params, headers=headers, allowed_hosts=allowed_hosts)
    response.raise_for_status()
    return response.json()


async def _fetch_text(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    allowed_hosts: set[str] | None = None,
) -> str:
    response = await _fetch_response(url, params=params, headers=headers, allowed_hosts=allowed_hosts)
    if response.status_code in {401, 403}:
        raise PermissionError(f"HTTP {response.status_code}: source requires auth/cookies or blocked automated access")
    response.raise_for_status()
    if "charset=windows-1251" in response.headers.get("content-type", "").lower():
        response.encoding = "cp1251"
    return response.text


def _github_token_headers() -> dict[str, str]:
    headers = _headers("GITHUB_TOKEN")
    headers["Accept"] = "application/vnd.github+json"
    headers["X-GitHub-Api-Version"] = "2022-11-28"
    return headers


def _parse_github_repo(value: str) -> tuple[str, str] | None:
    value = value.strip()
    if re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", value):
        owner, repo = value.split("/", 1)
        return owner, repo
    parsed = urlparse(value)
    if parsed.netloc.lower() != "github.com":
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return None


async def _github_handler(action: str, params: dict[str, Any], runtime: ConnectorRuntime) -> ConnectorResult:
    if action == "search_topics":
        query = str(params.get("query") or "").strip()
        if not query:
            return ConnectorResult(False, error="query is required")
        limit = _limit(params)
        repos, issues = await asyncio.gather(
            _fetch_json(
                "https://api.github.com/search/repositories",
                params={"q": query, "per_page": min(limit, 20)},
                headers=_github_token_headers(),
                allowed_hosts=GITHUB_API_HOSTS,
            ),
            _fetch_json(
                "https://api.github.com/search/issues",
                params={"q": query, "per_page": min(limit, 20)},
                headers=_github_token_headers(),
                allowed_hosts=GITHUB_API_HOSTS,
            ),
        )
        return ConnectorResult(
            True,
            data={
                "repositories": [
                    {
                        "id": item["id"],
                        "full_name": item["full_name"],
                        "title": item["full_name"],
                        "url": item["html_url"],
                        "description": item.get("description"),
                        "stars": item.get("stargazers_count"),
                    }
                    for item in repos.get("items", [])[:limit]
                ],
                "issues_and_prs": [
                    {
                        "id": item["id"],
                        "title": item.get("title"),
                        "url": item.get("html_url"),
                        "state": item.get("state"),
                        "repository_url": item.get("repository_url"),
                    }
                    for item in issues.get("items", [])[:limit]
                ],
            },
            metadata={"rate_limit_remaining": repos.get("rate", {}).get("remaining")},
        )

    if action == "read_topic":
        value = str(params.get("repo") or params.get("url") or params.get("id") or "").strip()
        repo = _parse_github_repo(value)
        if not repo:
            return ConnectorResult(False, error="repo/url/id must identify a GitHub repository")
        owner, name = repo
        repo_data = await _fetch_json(
            f"https://api.github.com/repos/{owner}/{name}",
            headers=_github_token_headers(),
            allowed_hosts=GITHUB_API_HOSTS,
        )
        readme = None
        try:
            readme_data = await _fetch_json(
                f"https://api.github.com/repos/{owner}/{name}/readme",
                headers=_github_token_headers(),
                allowed_hosts=GITHUB_API_HOSTS,
            )
            readme = readme_data.get("download_url") or readme_data.get("html_url")
        except Exception:
            readme = None
        return ConnectorResult(
            True,
            data={
                "full_name": repo_data.get("full_name"),
                "url": repo_data.get("html_url"),
                "description": repo_data.get("description"),
                "default_branch": repo_data.get("default_branch"),
                "stars": repo_data.get("stargazers_count"),
                "forks": repo_data.get("forks_count"),
                "open_issues": repo_data.get("open_issues_count"),
                "readme_url": readme,
                "topics": repo_data.get("topics", []),
            },
        )

    if action == "read_post":
        value = str(params.get("url") or params.get("id") or "").strip()
        parsed = urlparse(value)
        match = re.match(r"^/([^/]+)/([^/]+)/(issues|pull)/(\d+)", parsed.path)
        if not match:
            return ConnectorResult(False, error="url/id must be a GitHub issue or pull request URL")
        owner, repo, kind, number = match.groups()
        issue = await _fetch_json(
            f"https://api.github.com/repos/{owner}/{repo}/issues/{number}",
            headers=_github_token_headers(),
            allowed_hosts=GITHUB_API_HOSTS,
        )
        return ConnectorResult(
            True,
            data={
                "repo": f"{owner}/{repo}",
                "number": issue.get("number"),
                "title": issue.get("title"),
                "url": issue.get("html_url"),
                "state": issue.get("state"),
                "author": (issue.get("user") or {}).get("login"),
                "body": issue.get("body"),
                "comments": issue.get("comments"),
                "kind": kind,
            },
        )

    if action == "download_attachments":
        value = str(params.get("repo") or params.get("url") or params.get("id") or "").strip()
        repo = _parse_github_repo(value)
        if not repo:
            return ConnectorResult(False, error="repo/url/id must identify a GitHub repository or release URL")
        owner, name = repo
        limit = _limit(params, maximum=20)
        releases = await _fetch_json(
            f"https://api.github.com/repos/{owner}/{name}/releases",
            params={"per_page": min(limit, 20)},
            headers=_github_token_headers(),
            allowed_hosts=GITHUB_API_HOSTS,
        )
        urls: list[tuple[str, str]] = []
        for release in releases:
            for asset in release.get("assets", []):
                urls.append((asset.get("browser_download_url"), asset.get("name") or "asset"))
                if len(urls) >= limit:
                    break
            if len(urls) >= limit:
                break
        files = []
        for url, filename in urls:
            if url:
                files.append(
                    await _download_url(
                        url,
                        "github",
                        filename,
                        _github_token_headers(),
                        allowed_hosts=GITHUB_DOWNLOAD_HOSTS,
                    )
                )
        return ConnectorResult(True, data={"files": files, "count": len(files)})

    return ConnectorResult(False, error=f"Unsupported GitHub action: {action}")


def _fourpda_base_url(value: str) -> str | None:
    value = value.strip()
    if not value:
        return None
    if value.isdigit():
        return f"https://4pda.to/forum/index.php?showtopic={value}"
    if value.startswith("http://") or value.startswith("https://"):
        parsed = urlparse(value)
        host = (parsed.hostname or "").lower()
        if parsed.scheme != "https" or host not in FOURPDA_HOSTS:
            raise ValueError("4PDA URL must use https://4pda.to or https://www.4pda.to")
        return value
    return f"https://4pda.to/forum/index.php?{value.lstrip('?')}"


def _extract_4pda_posts(html: str, base_url: str, limit: int) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    candidates = soup.select("[id^=post-], [id^=entry], .post-block, .post")
    posts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in candidates:
        post_id = item.get("id") or ""
        if post_id in seen:
            continue
        seen.add(post_id)
        text = sanitize_text(item.get_text("\n", strip=True), max_length=6000)
        if not text:
            continue
        attachments = _extract_links(item, base_url, attachment_only=True)
        posts.append({"id": post_id, "text": text, "attachments": attachments})
        if len(posts) >= limit:
            break
    if posts:
        return posts

    text = sanitize_text(soup.get_text("\n", strip=True), max_length=12000)
    return [{"id": "", "text": text, "attachments": _extract_links(soup, base_url, attachment_only=True)}] if text else []


def _extract_links(soup: BeautifulSoup, base_url: str, *, attachment_only: bool = False) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    for anchor in soup.find_all("a", href=True):
        href = urljoin(base_url, anchor["href"])
        text = sanitize_text(anchor.get_text(" ", strip=True), max_length=200)
        lowered = href.lower()
        is_attachment = any(marker in lowered for marker in ("act=attach", "attach_id=", "dl=1", "/forum/dl/post/"))
        if attachment_only and not is_attachment:
            continue
        links.append({"url": href, "text": text})
    return links[:100]


async def _fourpda_handler(action: str, params: dict[str, Any], runtime: ConnectorRuntime) -> ConnectorResult:
    headers = _headers(cookie_env="FOURPDA_COOKIE")
    if action == "search_topics":
        query = str(params.get("query") or "").strip()
        if not query:
            return ConnectorResult(False, error="query is required")
        limit = _limit(params)
        html = await _fetch_text(
            "https://4pda.to/forum/index.php",
            params={"act": "search", "source": "all", "forums": "all", "subforums": "1", "query": query},
            headers=headers,
            allowed_hosts=FOURPDA_HOSTS,
        )
        soup = BeautifulSoup(html, "html.parser")
        topics = []
        for link in _extract_links(soup, "https://4pda.to/forum/index.php"):
            href = link["url"]
            if "showtopic=" not in href and "act=findpost" not in href and "showpost=" not in href:
                continue
            if not link["text"]:
                continue
            topics.append({"title": link["text"], "url": href})
            if len(topics) >= limit:
                break
        return ConnectorResult(True, data={"results": topics})

    if action == "read_topic":
        try:
            url = _fourpda_base_url(str(params.get("url") or params.get("id") or ""))
            if not url:
                return ConnectorResult(False, error="url or id is required")
        except ValueError as exc:
            return ConnectorResult(False, error=str(exc))
        limit = _limit(params, default=20, maximum=100)
        html = await _fetch_text(url, headers=headers, allowed_hosts=FOURPDA_HOSTS)
        soup = BeautifulSoup(html, "html.parser")
        title = sanitize_text((soup.find("title").get_text(" ", strip=True) if soup.find("title") else ""), max_length=300)
        return ConnectorResult(True, data={"title": title, "url": url, "posts": _extract_4pda_posts(html, url, limit)})

    if action == "read_post":
        try:
            url = _fourpda_base_url(str(params.get("url") or params.get("id") or ""))
            if not url:
                return ConnectorResult(False, error="url or id is required")
        except ValueError as exc:
            return ConnectorResult(False, error=str(exc))
        html = await _fetch_text(url, headers=headers, allowed_hosts=FOURPDA_HOSTS)
        posts = _extract_4pda_posts(html, url, 1)
        return ConnectorResult(True, data={"url": url, "post": posts[0] if posts else None})

    if action == "download_attachments":
        try:
            url = _fourpda_base_url(str(params.get("url") or params.get("id") or ""))
            if not url:
                return ConnectorResult(False, error="url or id is required")
        except ValueError as exc:
            return ConnectorResult(False, error=str(exc))
        limit = _limit(params, maximum=20)
        html = await _fetch_text(url, headers=headers, allowed_hosts=FOURPDA_HOSTS)
        soup = BeautifulSoup(html, "html.parser")
        attachments = _extract_links(soup, url, attachment_only=True)[:limit]
        if os.getenv("FOURPDA_ALLOW_RESTRICTED_DOWNLOADS") != "1":
            return ConnectorResult(
                True,
                data={
                    "files": [],
                    "count": 0,
                    "attachments": attachments,
                    "note": (
                        "4PDA attachment download is disabled by default because "
                        "the site restricts automated dl/attach endpoints. Set "
                        "FOURPDA_ALLOW_RESTRICTED_DOWNLOADS=1 to opt in explicitly."
                    ),
                },
            )
        files = [
            await _download_url(
                item["url"],
                "4pda",
                item["text"] or "attachment",
                headers,
                allowed_hosts=FOURPDA_HOSTS,
            )
            for item in attachments
        ]
        return ConnectorResult(True, data={"files": files, "count": len(files), "attachments": attachments})

    return ConnectorResult(False, error=f"Unsupported 4PDA action: {action}")


async def _x_handler(action: str, params: dict[str, Any], runtime: ConnectorRuntime) -> ConnectorResult:
    token = os.getenv("X_BEARER_TOKEN")
    if not token:
        return ConnectorResult(False, error="X_BEARER_TOKEN is required for X/Grok connector")
    headers = _headers("X_BEARER_TOKEN")
    if action == "search_topics":
        query = str(params.get("query") or "").strip()
        if not query:
            return ConnectorResult(False, error="query is required")
        limit = _limit(params, default=10, maximum=100)
        payload = await _fetch_json(
            "https://api.x.com/2/tweets/search/recent",
            params={
                "query": query,
                "max_results": max(10, min(100, limit)),
                "tweet.fields": "created_at,author_id,attachments,conversation_id,public_metrics",
                "expansions": "author_id,attachments.media_keys",
                "media.fields": "url,preview_image_url,type",
            },
            headers=headers,
            allowed_hosts=X_API_HOSTS,
        )
        return ConnectorResult(True, data=payload)

    if action in {"read_topic", "read_post"}:
        tweet_id = _x_tweet_id(str(params.get("id") or params.get("url") or ""))
        if not tweet_id:
            return ConnectorResult(False, error="tweet id or status URL is required")
        payload = await _fetch_json(
            f"https://api.x.com/2/tweets/{tweet_id}",
            params={
                "tweet.fields": "created_at,author_id,attachments,conversation_id,public_metrics,referenced_tweets",
                "expansions": "author_id,attachments.media_keys",
                "media.fields": "url,preview_image_url,type",
            },
            headers=headers,
            allowed_hosts=X_API_HOSTS,
        )
        return ConnectorResult(True, data=payload)

    if action == "download_attachments":
        tweet_id = _x_tweet_id(str(params.get("id") or params.get("url") or ""))
        if not tweet_id:
            return ConnectorResult(False, error="tweet id or status URL is required")
        payload = await _x_handler("read_post", {"id": tweet_id}, runtime)
        if not payload.ok:
            return payload
        includes = (payload.data or {}).get("includes", {})
        media = includes.get("media", [])
        files = []
        for item in media:
            media_url = item.get("url") or item.get("preview_image_url")
            if media_url:
                files.append(
                    await _download_url(
                        media_url,
                        "x",
                        Path(urlparse(media_url).path).name or "media",
                        headers,
                        allowed_hosts=X_MEDIA_HOSTS,
                    )
                )
        return ConnectorResult(True, data={"files": files, "count": len(files), "tweet": payload.data})

    return ConnectorResult(False, error=f"Unsupported X action: {action}")


def _x_tweet_id(value: str) -> str:
    value = value.strip()
    if value.isdigit():
        return value
    parsed = urlparse(value)
    match = re.search(r"/status(?:es)?/(\d+)", parsed.path)
    return match.group(1) if match else ""


async def _download_url(
    url: str,
    connector: str,
    filename_hint: str,
    headers: dict[str, str] | None = None,
    *,
    allowed_hosts: set[str] | None = None,
) -> dict[str, Any]:
    current_url = _validate_connector_url(url, allowed_hosts=allowed_hosts)
    parsed = urlparse(current_url)
    safe_name = _safe_filename(filename_hint or Path(parsed.path).name or "download")
    target_dir = (DOWNLOAD_ROOT / connector).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = (target_dir / safe_name).resolve()
    try:
        target.relative_to(target_dir)
    except ValueError:
        raise ValueError("Unsafe download target")
    current_headers = headers or _headers()
    content_type = ""

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=False) as client:
        for _ in range(MAX_REDIRECTS + 1):
            async with client.stream("GET", current_url, headers=current_headers) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise ConnectorHttpError("Redirect response has no Location header")
                    next_url = _validate_connector_url(urljoin(str(response.url), location), allowed_hosts=allowed_hosts)
                    current_headers = _redirect_headers(current_headers, str(response.url), next_url)
                    current_url = next_url
                    continue

                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                if "." not in target.name:
                    ext = mimetypes.guess_extension(content_type.split(";")[0].strip()) or ".bin"
                    target = target.with_suffix(ext)
                total = 0
                with target.open("wb") as fh:
                    async for chunk in response.aiter_bytes():
                        total += len(chunk)
                        if total > MAX_DOWNLOAD_BYTES:
                            fh.close()
                            target.unlink(missing_ok=True)
                            raise ValueError("Download exceeds size limit")
                        fh.write(chunk)
                return {
                    "url": redact_url_secrets(url),
                    "path": str(target),
                    "bytes": target.stat().st_size,
                    "content_type": content_type,
                }

    raise ConnectorHttpError("Too many redirects")


def _safe_filename(value: str) -> str:
    value = sanitize_text(value, max_length=120) or "download"
    value = re.sub(r"[^\w.\-()+ ]+", "_", value, flags=re.UNICODE).strip(" .")
    return value or "download"
