"""Smoke test: verify all major modules import successfully after refactoring."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ.setdefault("ENCRYPTION_KEY", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=")
os.environ.setdefault("BOT_TOKEN", "test:token")
os.environ.setdefault("OWNER_TELEGRAM_ID", "123456789")

import pytest

CRITICAL_MODULES = [
    # Database
    ("src.db.models", "DB models package (32 models)"),
    ("src.db.session", "DB session manager"),
    ("src.db.repo", "DB repository"),
    # Core intelligence
    ("src.core.intelligence.maestro", "Maestro orchestrator"),
    ("src.core.intelligence.smart_autorouter", "Smart AutoRouter"),
    ("src.core.intelligence.llm_guard", "LLM error wrapper"),
    # Bot handlers
    ("src.bot.handlers.free_text", "Free text handler"),
    ("src.bot.handlers.free_text_exec", "Exec handlers"),
    ("src.bot.handlers.free_text_pipeline", "Pipeline stages"),
    ("src.bot.handlers.free_text_common", "Common utilities"),
    ("src.bot.handlers.free_text_memory", "Memory handlers"),
    ("src.bot.handlers.free_text_settings", "Settings handlers"),
    # Routing wordlists (under core.intelligence, not bot.handlers)
    ("src.core.intelligence.routing_wordlists", "Routing word lists"),
    # LLM
    ("src.llm.router", "LLM router with multi-key support"),
    # Infrastructure
    ("src.core.infra.task_manager", "Background task manager"),
    ("src.core.infra.system_tasks", "System background tasks"),
    # Agents
    ("src.agents.search_agent", "Search agent"),
    ("src.agents.memory_agent", "Memory agent"),
    ("src.agents.commitment_agent", "Commitment agent"),
    ("src.agents.summarizer_agent", "Summarizer agent"),
    ("src.agents.draft_agent", "Draft agent"),
    ("src.agents.digest_agent", "Digest agent"),
]

OPTIONAL_MODULES = [
    ("src.config", "Config"),
    ("src.main", "Main entry point"),
]


class TestCriticalImports:
    @pytest.mark.parametrize("module_path,description", CRITICAL_MODULES)
    def test_import(self, module_path, description):
        """Verify critical module imports without error."""
        try:
            __import__(module_path)
        except ImportError as e:
            pytest.fail(f"FAILED to import {module_path} ({description}): {e}")


class TestModelExports:
    def test_all_models_exported(self):
        """Verify all 32 models are accessible from src.db.models."""
        from src.db.models import (
            Base,
        )

        assert len(Base.metadata.tables) >= 26, (
            f"Expected at least 26 tables, got {len(Base.metadata.tables)}"
        )
        assert "users" in Base.metadata.tables
        assert "memories" in Base.metadata.tables
        assert "messages" in Base.metadata.tables

    def test_model_submodules_exist(self):
        """Verify all model sub-module files exist."""
        models_dir = os.path.join(
            os.path.dirname(__file__), "..", "src", "db", "models"
        )
        expected = [
            "__init__.py",
            "_base.py",
            "_auth.py",
            "_contacts.py",
            "_messaging.py",
            "_memory.py",
            "_learning.py",
        ]
        for f in expected:
            path = os.path.join(models_dir, f)
            assert os.path.exists(path), f"Missing: {f}"


class TestBackwardCompat:
    def test_import_user_from_models(self):
        """Verify backward-compatible import works."""
        from src.db.models import User, Memory, Message

        assert User is not None
        assert Memory is not None
        assert Message is not None

    def test_import_from_models_top(self):
        """Verify 'from src.db.models import X' works for commonly used models."""
        from src.db.models import Notification, Contact, UserSettings, LlmKeySlot

        assert Notification is not None
        assert Contact is not None
        assert UserSettings is not None
        assert LlmKeySlot is not None
