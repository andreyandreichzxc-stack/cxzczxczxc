"""mcp_gmail tool — registered via @tool decorator.

Check Gmail inbox via the Gmail API (OAuth 2.0).

Requires:
- A Gmail OAuth 2.0 credentials file at ``data/gmail_credentials.json``
- ``google-auth`` and ``google-api-python-client`` packages installed.

Risk: **high** — requires explicit user confirmation before execution.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.config import settings
from src.core.actions.tool_registry import tool

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────

_CREDENTIALS_PATH: Path = settings.data_dir / "gmail_credentials.json"

_GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

# Sane limit for results
_MAX_RESULTS_MAX = 50


# ══════════════════════════════════════════════════════════════════════════
# Tool: mcp_gmail
# ══════════════════════════════════════════════════════════════════════════


@tool(
    name="mcp_gmail",
    description=(
        "Check Gmail inbox for messages matching a query. "
        "Supports one action:\n"
        "- 'check' — return recent messages from Gmail inbox.\n"
        "Requires: gmail_credentials.json in data/ directory.  "
        "Risk: high — user confirmation is always required."
    ),
    category="email",
    risk="high",
    requires_confirmation=True,
    params={
        "action": "str — 'check'",
        "max_results": "int — max messages to return (default 5, max 50)",
        "query": "str — Gmail search query (default 'is:unread')",
    },
)
async def mcp_gmail(
    action: str,
    max_results: int = 5,
    query: str = "is:unread",
    **kwargs: Any,
) -> dict[str, Any]:
    """Gmail inbox check tool.

    Args:
        action: ``"check"``.
        max_results: Maximum number of messages to return (1–50, default 5).
        query: Gmail search query (default ``"is:unread"``).

    Returns:
        A dict with a list of messages or an ``"error"`` key on failure.
    """
    try:
        if action == "check":
            return await _gmail_check(max_results, query)
        else:
            return {"error": f"Unknown action {action!r}. Valid actions: check"}
    except Exception as exc:
        logger.exception("mcp_gmail(%r) failed", action)
        return {"error": str(exc)}


# ══════════════════════════════════════════════════════════════════════════
# Action implementation
# ══════════════════════════════════════════════════════════════════════════


async def _gmail_check(max_results: int, query: str) -> dict[str, Any]:
    """Fetch messages from Gmail inbox.

    Loads OAuth credentials from ``data/gmail_credentials.json``, builds
    the Gmail service, and queries for messages matching *query*.
    """
    # Validate credentials file exists
    if not _CREDENTIALS_PATH.is_file():
        return {
            "error": (
                "Gmail not configured. "
                "Place OAuth 2.0 credentials at data/gmail_credentials.json"
            )
        }

    # Clamp max_results
    max_results = max(1, min(max_results, _MAX_RESULTS_MAX))

    # Lazy import Google API libs
    try:
        from google.auth.transport.requests import Request as GoogleAuthRequest
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        return {
            "error": (
                "google-auth and google-api-python-client required. "
                "Install with: pip install google-auth google-api-python-client "
                "google-auth-oauthlib"
            )
        }

    try:
        creds = None

        # Token file for persistent auth
        token_path = _CREDENTIALS_PATH.with_name("gmail_token.json")

        if token_path.is_file():
            creds = Credentials.from_authorized_user_file(
                str(token_path), _GMAIL_SCOPES
            )

        # If no valid credentials, run OAuth flow
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(GoogleAuthRequest())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(_CREDENTIALS_PATH), _GMAIL_SCOPES
                )
                creds = flow.run_local_server(port=0)

            # Save token for next run
            with open(token_path, "w") as token_file:
                token_file.write(creds.to_json())

        # Build service and fetch messages
        service = build("gmail", "v1", credentials=creds)

        # List messages matching query
        results = (
            service.users()
            .messages()
            .list(userId="me", q=query, maxResults=max_results)
            .execute()
        )

        messages = results.get("messages", [])
        if not messages:
            return {"ok": True, "messages": [], "count": 0}

        # Fetch full details for each message
        output: list[dict[str, str]] = []
        for msg in messages[:max_results]:
            msg_data = (
                service.users()
                .messages()
                .get(userId="me", id=msg["id"], format="metadata")
                .execute()
            )

            headers = msg_data.get("payload", {}).get("headers", [])
            header_map = {h["name"].lower(): h["value"] for h in headers}

            output.append(
                {
                    "from": header_map.get("from", ""),
                    "subject": header_map.get("subject", ""),
                    "snippet": msg_data.get("snippet", ""),
                    "date": header_map.get("date", ""),
                }
            )

        return {
            "ok": True,
            "messages": output,
            "count": len(output),
            "query": query,
        }

    except Exception as exc:
        logger.exception("Gmail API call failed")
        return {"error": f"Gmail API error: {exc}"}
