"""Subdirectory hint provider — progressive AGENTS.md discovery."""

from __future__ import annotations
import logging
import re
from pathlib import Path

from src.core.context.spec import ContextChunk

logger = logging.getLogger(__name__)

_HINT_FILES = ["AGENTS.md", "CLAUDE.md", ".cursorrules", "README.md"]


class SubdirectoryHintProvider:
    """Loads AGENTS.md from directories as agent navigates to them."""

    name = "subdirectory_hints"

    def __init__(self) -> None:
        self.root = Path(__file__).resolve().parent.parent.parent.parent
        self.visited: set[Path] = {self.root}

    async def get_context(self, query, *, telegram_id, contact_id=None, limit=8):
        # This provider doesn't use query — it returns hints accumulated
        # from tool calls. For the engine, return empty.
        return []

    def on_tool_args(self, tool_name: str, args: dict) -> list[ContextChunk]:
        """Called after each tool use — extracts paths and discovers hints."""
        if args is None or not isinstance(args, dict):
            return []
        paths: list[str] = []

        # Extract paths from common arguments
        for arg_name in ["path", "workdir", "filePath", "file_path", "directory"]:
            if arg_name in args and isinstance(args[arg_name], str):
                paths.append(args[arg_name])

        # Parse shell commands
        if tool_name == "bash" and "command" in args:
            cmd = str(args["command"])
            # Windows: C:\path, Unix: /path
            found = re.findall(r'(?:["\'])?([A-Za-z]:[/\\][\w/\\\.-]+)', cmd)
            paths.extend(found)
            found = re.findall(r'(?:["\'])?(/[\w/\.-]+)', cmd)
            paths.extend(found)

        return self._find_new_hints(paths)

    def _find_new_hints(self, paths: list[str]) -> list[ContextChunk]:
        """Scan new directories for AGENTS.md files."""
        hints: list[ContextChunk] = []
        for p in paths:
            try:
                path = Path(p)
            except Exception:
                continue
            if not path.exists():
                continue
            if path.is_absolute():
                current = path.parent if path.is_file() else path
            else:
                current = self.root / (path.parent if path.is_file() else path)

            # Walk up to 5 parent levels
            for _ in range(5):
                if current in self.visited:
                    break
                if current == self.root.parent:
                    break
                self.visited.add(current)

                for hint_file in _HINT_FILES:
                    hf = current / hint_file
                    if hf.exists():
                        try:
                            content = hf.read_text(encoding="utf-8")
                            hints.append(
                                ContextChunk(
                                    text=f"### {hint_file} ({current.relative_to(self.root)})\n{content[:1500]}",
                                    source="subdirectory_hint",
                                    reason=hint_file,
                                )
                            )
                            logger.debug("Discovered hint: %s", hf)
                        except Exception:
                            logger.debug("Failed to read hint: %s", hf)
                current = current.parent
        return hints


# Module-level singleton for tracking across tool calls
subdirectory_provider = SubdirectoryHintProvider()
