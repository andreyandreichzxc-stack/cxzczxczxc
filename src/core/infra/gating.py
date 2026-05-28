"""Gating — check runtime dependencies and enable/disable features gracefully."""

from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)


class Gates:
    """Runtime dependency checker with graceful degradation."""

    def __init__(self) -> None:
        self._checks: list[dict] = []
        self._passed: set[str] = set()
        self._failed: dict[str, str] = {}

    def register(
        self,
        name: str,
        *,
        check: Callable[[], bool],
        fallback: str | None = None,
        description: str = "",
        install_hint: str = "",
    ) -> "Gates":
        """Register a dependency check.

        Args:
            name: feature name, e.g. "whisper_transcription"
            check: callable that returns True if dependency is available
            fallback: what to use if check fails, e.g. "openai_whisper_api", None = disable
            description: human-readable name
            install_hint: pip install command for the missing dependency
        """
        self._checks.append(
            {
                "name": name,
                "check": check,
                "fallback": fallback,
                "description": description or name,
                "install_hint": install_hint,
                "result": None,
            }
        )
        return self

    def run_all(self) -> None:
        """Run all registered checks. Log results."""
        for entry in self._checks:
            name = entry["name"]
            try:
                if entry["check"]():
                    self._passed.add(name)
                    entry["result"] = "passed"
                    logger.info("✅ Gate passed: %s", entry["description"])
                else:
                    self._failed[name] = "check returned False"
                    entry["result"] = "failed"
                    fallback = entry["fallback"]
                    if fallback:
                        logger.warning(
                            "⏭️ Gate failed: %s → fallback: %s",
                            entry["description"],
                            fallback,
                        )
                    else:
                        logger.warning(
                            "❌ Gate failed: %s → DISABLED (no fallback)",
                            entry["description"],
                        )
            except Exception as exc:
                self._failed[name] = str(exc)
                entry["result"] = "error"
                logger.warning("⚠️ Gate error: %s → %s", entry["description"], exc)

    def is_available(self, name: str) -> bool:
        return name in self._passed

    def get_fallback(self, name: str) -> str | None:
        for entry in self._checks:
            if entry["name"] == name:
                return entry.get("fallback") if entry["result"] != "passed" else None
        return None

    def get_install_hint(self, name: str) -> str:
        for entry in self._checks:
            if entry["name"] == name:
                return entry.get("install_hint", "")
        return ""

    @property
    def missing_install_hints(self) -> list[dict]:
        """Returns [{name, description, install_hint}] for all failed gates with hints."""
        hints = []
        for entry in self._checks:
            if entry["result"] != "passed" and entry.get("install_hint"):
                hints.append(
                    {
                        "name": entry["name"],
                        "description": entry["description"],
                        "install_hint": entry["install_hint"],
                    }
                )
        return hints
        return None

    @property
    def status(self) -> dict:
        return {
            "passed": sorted(self._passed),
            "failed": dict(self._failed),
            "total": len(self._checks),
        }


# Module-level singleton
gates = Gates()
