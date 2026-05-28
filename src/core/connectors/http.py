"""HTTP helpers for future site connectors."""

from __future__ import annotations

import ipaddress
import socket
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx


DEFAULT_TIMEOUT = 20.0
MAX_REDIRECTS = 5
ALLOWED_SCHEMES = {"http", "https"}


class ConnectorHttpError(RuntimeError):
    pass


def _is_public_ip(host: str) -> bool:
    try:
        addresses = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ConnectorHttpError(f"Cannot resolve host: {host}") from exc

    for address in addresses:
        ip_text = address[4][0]
        ip = ipaddress.ip_address(ip_text)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    return True


def validate_public_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise ConnectorHttpError("Only http and https URLs are allowed")
    if not parsed.hostname:
        raise ConnectorHttpError("URL must include a hostname")
    if not _is_public_ip(parsed.hostname):
        raise ConnectorHttpError("URL resolves to a non-public address")
    return url


async def request_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    json: Any = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Any:
    current_url = validate_public_url(url)
    current_method = method.upper()
    current_json = json

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        for _ in range(MAX_REDIRECTS + 1):
            response = await client.request(
                current_method,
                current_url,
                headers=headers,
                params=params,
                json=current_json,
            )
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    raise ConnectorHttpError("Redirect response has no Location header")
                current_url = validate_public_url(urljoin(str(response.url), location))
                if response.status_code in {301, 302, 303}:
                    current_method = "GET"
                    current_json = None
                continue
            response.raise_for_status()
            return response.json()

    raise ConnectorHttpError("Too many redirects")
