"""Audit: maps agentmemory concepts to TelegramHelper implementations.

This file serves as a living registry that tracks parity between the
53 agentmemory tools and what TelegramHelper implements for each one.

Usage:
    from src.core.memory.tools_audit import AGENTMEMORY_MAPPING, missing_nl, missing_impl
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Master mapping: agentmemory tool → TelegramHelper status
# ---------------------------------------------------------------------------

AGENTMEMORY_MAPPING: dict[str, dict[str, Any]] = {
    # ── Memory CRUD ──────────────────────────────────────────────────
    "create_memory": {
        "exists": True,
        "impl": "memory_repo.add_memory() / add_memory_candidate()",
        "nl": "store_memory",
    },
    "read_memory": {
        "exists": True,
        "impl": "memory_repo.list_memories() / search_memories()",
        "nl": "list_memories",
    },
    "search_memory": {
        "exists": True,
        "impl": "memory_recall.recall() / search_memories() / fts_search()",
        "nl": "recall_memory (tool only, auto in pipeline)",
    },
    "update_memory": {
        "exists": True,
        "impl": "memory_repo.update_memory() via inline SQLAlchemy",
        "nl": "update_memory",  # NEW in Phase 5.2
    },
    "delete_memory": {
        "exists": True,
        "impl": "memory_repo.delete_memory()",
        "nl": "forget_memory",
    },
    "list_memories": {
        "exists": True,
        "impl": "memory_repo.list_memories()",
        "nl": "list_memories",
    },
    # ── Graph / Relations ────────────────────────────────────────────
    "create_relation": {
        "exists": True,
        "impl": "memory_repo.link_memories()",
        "nl": "link_memories",  # NEW in Phase 5.2
    },
    "get_relations": {
        "exists": True,
        "impl": "memory_repo.get_memory_graph() / get_linked_memories()",
        "nl": "show_memory_graph",  # NEW in Phase 5.2
    },
    "delete_relation": {
        "exists": True,
        "impl": "memory_repo.unlink_memories()",
        "nl": None,  # Not exposed via NL (advanced)
    },
    # ── Timeline ─────────────────────────────────────────────────────
    "get_timeline": {
        "exists": True,
        "impl": "memory_cmd._format_timeline() via /memory --timeline",
        "nl": "show_memory_timeline (via /memory command)",
    },
    # ── Contradiction ────────────────────────────────────────────────
    "check_contradiction": {
        "exists": True,
        "impl": "contradiction_detector.detect_contradiction()",
        "nl": "contradiction (auto — runs in pipeline)",
    },
    # ── Session ──────────────────────────────────────────────────────
    "get_session": {
        "exists": True,
        "impl": "session_recorder.get_session_history()",
        "nl": "show_sessions",  # NEW in Phase 5.2
    },
    # ── Health ───────────────────────────────────────────────────────
    "get_health": {
        "exists": True,
        "impl": "memory_health.calculate_health_score()",
        "nl": "show_memory_health",  # NEW in Phase 5.2
    },
    # ── Proactive / Suggestions ──────────────────────────────────────
    "get_suggestions": {
        "exists": True,
        "impl": "memory_patterns.detect_patterns()",
        "nl": "show_suggestions",  # NEW in Phase 5.2
    },
    # ── Export / Import (MISSING — not implemented) ──────────────────
    "export_memory": {
        "exists": False,
        "impl": None,
        "nl": None,
    },
    "import_memory": {
        "exists": False,
        "impl": None,
        "nl": None,
    },
    # ── Versioning (MISSING — not implemented) ───────────────────────
    "get_memory_versions": {
        "exists": False,
        "impl": None,
        "nl": None,
    },
    "restore_memory_version": {
        "exists": False,
        "impl": None,
        "nl": None,
    },
    # ── Batch operations (MISSING — not implemented) ─────────────────
    "bulk_delete_memories": {
        "exists": False,
        "impl": None,
        "nl": None,
    },
    "bulk_update_memories": {
        "exists": False,
        "impl": None,
        "nl": None,
    },
}


# ── Derived helpers ──────────────────────────────────────────────────────


def missing_nl() -> list[str]:
    """Return agentmemory tool names that exist but lack NL intent."""
    return [
        name
        for name, info in AGENTMEMORY_MAPPING.items()
        if info["exists"] and info["nl"] is None
    ]


def missing_impl() -> list[str]:
    """Return agentmemory tool names that have no implementation at all."""
    return [name for name, info in AGENTMEMORY_MAPPING.items() if not info["exists"]]


def summary() -> str:
    """Return a human-readable parity summary."""
    total = len(AGENTMEMORY_MAPPING)
    existing = sum(1 for v in AGENTMEMORY_MAPPING.values() if v["exists"])
    with_nl = sum(
        1 for v in AGENTMEMORY_MAPPING.values() if v["exists"] and v["nl"] is not None
    )
    missing_impl_count = total - existing
    missing_nl_count = existing - with_nl
    return (
        f"agentmemory parity: {total} tools tracked.\n"
        f"  ✅ Implemented: {existing} ({with_nl} with NL intent)\n"
        f"  ❌ Missing impl: {missing_impl_count}\n"
        f"  ⚠️  Missing NL: {missing_nl_count}"
    )
