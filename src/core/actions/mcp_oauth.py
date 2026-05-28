"""MCP OAuth 2.1 client — connect to hosted MCP servers with OAuth.

Provides the :class:`MCPOAuthClient` class for the full OAuth 2.1 + PKCE
flow (Dynamic Client Registration → authorization → token exchange) plus
token persistence and automatic refresh.

Usage::

    from src.core.actions.mcp_oauth import mcp_oauth

    result = await mcp_oauth.connect("linear", "https://mcp.linear.app")
    token = await mcp_oauth.get_token("linear")
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse, parse_qs

import httpx

from src.config import settings

logger = logging.getLogger(__name__)

TOKENS_DIR: Path = settings.data_dir / "mcp-tokens"
CODE_VERIFIER_LENGTH = 64
_LOCAL_CALLBACK_PORT = 18080
_LOCAL_CALLBACK_PATH = "/callback"
_CALLBACK_TIMEOUT = 120.0  # seconds to wait for user authorisation
_REFRESH_BUFFER = 60  # refresh when < 60 s remaining


@dataclass
class OAuthToken:
    """Serialisable OAuth 2.1 token with optional refresh support."""

    access_token: str
    token_type: str = "Bearer"
    expires_at: float = 0.0  # asyncio.get_event_loop().time() + expires_in
    refresh_token: str | None = None
    server_url: str = ""


class MCPOAuthClient:
    """OAuth 2.1 + PKCE client for MCP servers.

    Supports:
    - PKCE (S256) code challenge / verifier
    - Dynamic Client Registration (RFC 7591) when available
    - Local HTTP callback server for the authorisation redirect
    - Token persistence as JSON files in ``TOKENS_DIR``
    - Automatic token refresh before expiration
    """

    def __init__(self) -> None:
        TOKENS_DIR.mkdir(parents=True, exist_ok=True)
        self._http = httpx.AsyncClient(timeout=30.0)

    async def close(self) -> None:
        """Close the underlying HTTP client session."""
        await self._http.aclose()

    # ── Public API ──────────────────────────────────────────────────────────

    async def connect(self, server_name: str, server_url: str) -> dict[str, Any]:
        """Full OAuth 2.1 + PKCE flow against an MCP server.

        Steps:
        1. Fetch OAuth metadata from ``.well-known``.
        2. Generate PKCE code verifier / challenge.
        3. Dynamic client registration (optional — falls back to server
           name as client_id).
        4. Start local HTTP server and build the authorisation URL.
        5. Exchange the authorisation code for tokens.
        6. Persist the token for future ``get_token()`` calls.

        Returns:
            ``{"ok": True, "token_type": "Bearer"}`` on success, or
            ``{"error": "<reason>"}`` on failure.
        """
        try:
            # 1. Fetch OAuth metadata
            metadata = await self._fetch_metadata(server_url)
            if "error" in metadata:
                return metadata

            # 2. Generate PKCE
            code_verifier = self._generate_code_verifier()
            code_challenge = self._compute_code_challenge(code_verifier)

            # 3. Dynamic Client Registration (if supported)
            client_id = await self._register_client(metadata, server_name)

            # 4. Build authorisation URL
            redirect_uri = (
                f"http://localhost:{_LOCAL_CALLBACK_PORT}{_LOCAL_CALLBACK_PATH}"
            )
            state = secrets.token_hex(16)
            auth_url = self._build_auth_url(
                metadata,
                client_id,
                redirect_uri,
                code_challenge,
                state,
            )

            # 5. Start local server for callback
            token = await self._wait_for_callback(
                metadata,
                client_id,
                redirect_uri,
                code_verifier,
                state,
                auth_url,
            )
            if "error" in token:
                return token

            # 6. Save token for future use
            self._save_token(server_name, token, server_url)
            return {"ok": True, "token_type": token.get("token_type", "Bearer")}

        except Exception as exc:
            logger.error("OAuth connect failed for %s: %s", server_name, exc)
            return {"error": str(exc)}

    async def get_token(self, server_name: str) -> str | None:
        """Return a valid access token for *server_name*, refreshing if needed.

        Returns ``None`` if no token is stored or the token cannot be
        refreshed.
        """
        token_data = self._load_token(server_name)
        if not token_data:
            return None

        token = OAuthToken(**token_data)
        now = asyncio.get_event_loop().time()

        # Refresh if expired or about to expire
        if token.expires_at and token.expires_at - now < _REFRESH_BUFFER:
            if token.refresh_token:
                new_token = await self._refresh_token(server_name, token)
                if new_token:
                    return new_token.access_token
            return None

        return token.access_token

    def is_connected(self, server_name: str) -> bool:
        """Check whether a token file exists for *server_name*."""
        return (TOKENS_DIR / f"{server_name}.json").exists()

    # ── OAuth metadata discovery ──────────────────────────────────────────

    async def _fetch_metadata(self, server_url: str) -> dict[str, Any]:
        """Fetch OAuth metadata from the server's well-known endpoints.

        Tries ``oauth-protected-resource`` first, then falls back to
        ``openid-configuration`` (some servers expose OAuth metadata there).
        """
        base = server_url.rstrip("/")

        # Primary: OAuth Resource Server metadata (RFC 8414-style)
        try:
            resp = await self._http.get(
                f"{base}/.well-known/oauth-protected-resource",
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass

        # Fallback: OpenID Connect discovery
        try:
            resp = await self._http.get(
                f"{base}/.well-known/openid-configuration",
            )
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "authorization_endpoint": data.get("authorization_endpoint"),
                    "token_endpoint": data.get("token_endpoint"),
                    "registration_endpoint": data.get("registration_endpoint"),
                }
        except Exception:
            pass

        return {"error": f"No OAuth metadata found at {server_url}"}

    # ── PKCE helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _generate_code_verifier() -> str:
        """Generate a PKCE code verifier (high-entropy random string)."""
        return secrets.token_urlsafe(CODE_VERIFIER_LENGTH)[:128]

    @staticmethod
    def _compute_code_challenge(verifier: str) -> str:
        """Compute S256 PKCE code challenge from *verifier*."""
        digest = hashlib.sha256(verifier.encode()).digest()
        return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()

    # ── Dynamic Client Registration ───────────────────────────────────────

    async def _register_client(self, metadata: dict[str, Any], name: str) -> str:
        """Register this client via DCR (RFC 7591), or fallback to *name*."""
        reg_url = metadata.get("registration_endpoint")
        if not reg_url:
            return name

        try:
            resp = await self._http.post(
                reg_url,
                json={
                    "client_name": f"TelegramHelper-{name}",
                    "redirect_uris": [
                        f"http://localhost:{_LOCAL_CALLBACK_PORT}{_LOCAL_CALLBACK_PATH}",
                    ],
                    "grant_types": ["authorization_code", "refresh_token"],
                    "token_endpoint_auth_method": "none",
                },
            )
            if resp.status_code in (200, 201):
                data = resp.json()
                return data.get("client_id", name)
        except Exception:
            logger.debug("DCR failed for %s, using fallback client_id", name)

        return name

    # ── Authorisation URL builder ─────────────────────────────────────────

    @staticmethod
    def _build_auth_url(
        metadata: dict[str, Any],
        client_id: str,
        redirect_uri: str,
        code_challenge: str,
        state: str,
    ) -> str:
        """Build the authorisation URL with PKCE parameters."""
        auth_endpoint = metadata.get("authorization_endpoint", "")
        if not auth_endpoint:
            return ""

        params = {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "state": state,
            "scope": "mcp",
        }
        return f"{auth_endpoint}?{urlencode(params)}"

    # ── Local callback server + code exchange ─────────────────────────────

    async def _wait_for_callback(
        self,
        metadata: dict[str, Any],
        client_id: str,
        redirect_uri: str,
        code_verifier: str,
        state: str,
        auth_url: str,
    ) -> dict[str, Any]:
        """Start a local HTTP server, wait for the OAuth callback, exchange code.

        Logs the authorisation URL so the operator can open it in a browser.
        """
        result: dict[str, Any] = {"error": "timeout"}
        event = asyncio.Event()

        async def handle_callback(
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
        ) -> None:
            try:
                request_raw = await asyncio.wait_for(
                    reader.read(4096),
                    timeout=5.0,
                )
                request_line = request_raw.decode("utf-8", errors="ignore").split(
                    "\r\n"
                )[0]
                path = request_line.split(" ")[1] if " " in request_line else "/"
                parsed = urlparse(path)
                params = parse_qs(parsed.query)

                if params.get("state", [""])[0] == state:
                    code = params.get("code", [None])[0]
                    if code:
                        token_resp = await self._exchange_code(
                            metadata,
                            client_id,
                            code,
                            code_verifier,
                            redirect_uri,
                        )
                        result.clear()
                        result.update(token_resp)
                    else:
                        result["error"] = params.get(
                            "error_description", ["no code received"]
                        )[0]

                body = (
                    "<html><body>"
                    "<h1>Authorization complete</h1>"
                    "<p>You can close this window.</p>"
                    "</body></html>"
                )
                response = (
                    f"HTTP/1.1 200 OK\r\n"
                    f"Content-Length: {len(body)}\r\n"
                    f"Content-Type: text/html\r\n\r\n"
                    f"{body}"
                )
                writer.write(response.encode())
                await writer.drain()
            except Exception:
                pass
            finally:
                writer.close()
                event.set()

        # Parse redirect_uri for host/port
        parsed_redirect = urlparse(redirect_uri)
        host = parsed_redirect.hostname or "localhost"
        port = parsed_redirect.port or _LOCAL_CALLBACK_PORT

        server = await asyncio.start_server(handle_callback, host, port)

        logger.info(
            "OAuth: open this URL in your browser to authorise:\n  %s",
            auth_url or "(no auth URL — metadata missing authorization_endpoint)",
        )

        try:
            await asyncio.wait_for(event.wait(), timeout=_CALLBACK_TIMEOUT)
        except asyncio.TimeoutError:
            result["error"] = "timeout waiting for OAuth callback"
        finally:
            server.close()

        return result

    async def _exchange_code(
        self,
        metadata: dict[str, Any],
        client_id: str,
        code: str,
        code_verifier: str,
        redirect_uri: str,
    ) -> dict[str, Any]:
        """Exchange authorisation *code* for an access token."""
        token_endpoint = metadata.get("token_endpoint", "")
        if not token_endpoint:
            return {"error": "no token_endpoint in metadata"}

        try:
            resp = await self._http.post(
                token_endpoint,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                    "client_id": client_id,
                    "code_verifier": code_verifier,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                expires_in = data.get("expires_in", 3600)
                return {
                    "access_token": data["access_token"],
                    "token_type": data.get("token_type", "Bearer"),
                    "expires_at": asyncio.get_event_loop().time() + expires_in,
                    "refresh_token": data.get("refresh_token"),
                }
            return {"error": f"token exchange failed: {resp.status_code}"}
        except Exception as exc:
            return {"error": str(exc)}

    # ── Token refresh ────────────────────────────────────────────────────

    async def _refresh_token(
        self,
        server_name: str,
        token: OAuthToken,
    ) -> OAuthToken | None:
        """Refresh an expired token using the stored refresh token."""
        token_data = self._load_token(server_name)
        if not token_data:
            return None

        metadata = await self._fetch_metadata(token_data.get("server_url", ""))
        token_endpoint = metadata.get("token_endpoint", "")
        if not token_endpoint:
            return None

        try:
            resp = await self._http.post(
                token_endpoint,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": token.refresh_token,
                    "client_id": server_name,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                expires_in = data.get("expires_in", 3600)
                new_token = OAuthToken(
                    access_token=data["access_token"],
                    token_type=data.get("token_type", "Bearer"),
                    expires_at=asyncio.get_event_loop().time() + expires_in,
                    refresh_token=data.get("refresh_token", token.refresh_token),
                    server_url=token.server_url,
                )
                self._save_token(server_name, vars(new_token), token.server_url)
                return new_token
        except Exception:
            pass

        return None

    # ── Token persistence ─────────────────────────────────────────────────

    def _save_token(
        self, server_name: str, token_data: dict[str, Any], server_url: str
    ) -> None:
        """Persist *token_data* as a JSON file keyed by *server_name*."""
        data = dict(token_data)
        data["server_url"] = server_url
        path = TOKENS_DIR / f"{server_name}.json"
        path.write_text(json.dumps(data, indent=2))

    def _load_token(self, server_name: str) -> dict[str, Any] | None:
        """Load persisted token data for *server_name*, or ``None``."""
        path = TOKENS_DIR / f"{server_name}.json"
        if path.exists():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                pass
        return None


# Module-level singleton — imported by tools and other modules
mcp_oauth = MCPOAuthClient()
