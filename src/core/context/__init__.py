"""Pluggable context engine — unified API for all context sources."""

from src.core.context.engine import ContextEngine, engine
from src.core.context.spec import ContextChunk, ContextProvider

__all__ = ["ContextEngine", "engine", "ContextChunk", "ContextProvider"]
