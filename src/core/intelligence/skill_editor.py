"""Skill Editor — bounded text-space edits for skill evolution.

Inspired by SkillOpt's ReflACT training loop. Instead of replacing entire
skills, applies minimal targeted edits (append, insert_after, replace, delete)
with a configurable edit budget (textual learning rate).

Protected regions (marked with <!-- PROTECTED_START --> / <!-- PROTECTED_END -->)
are never touched by step-level edits — only slow_update (manual review) can
modify them.

Operations:
- append: add content at the end of body
- insert_after: insert content after a marker line
- replace: replace old content with new content
- delete: remove content matching a pattern
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# ── Constants ──

PROTECTED_START = "<!-- PROTECTED_START -->"
PROTECTED_END = "<!-- PROTECTED_END -->"

# Default edit budget (textual learning rate)
DEFAULT_EDIT_BUDGET = 3


class EditOp(str, Enum):
    APPEND = "append"
    INSERT_AFTER = "insert_after"
    REPLACE = "replace"
    DELETE = "delete"


@dataclass
class SkillEdit:
    """A single bounded edit to a skill body."""

    op: EditOp
    target: str | None = None  # marker for insert_after, old text for replace/delete
    content: str = ""  # new content for append/insert_after/replace
    reason: str = ""  # why this edit is proposed

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op.value,
            "target": self.target,
            "content": self.content,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillEdit:
        return cls(
            op=EditOp(data["op"]),
            target=data.get("target"),
            content=data.get("content", ""),
            reason=data.get("reason", ""),
        )


@dataclass
class EditResult:
    """Result of applying edits to a skill body."""

    success: bool
    new_body: str
    applied_edits: list[SkillEdit] = field(default_factory=list)
    rejected_edits: list[tuple[SkillEdit, str]] = field(
        default_factory=list
    )  # (edit, reason)
    version_bump: str = "patch"  # patch/minor/major


def _is_in_protected_region(body: str, position: int) -> bool:
    """Check if a position in the body is inside a protected region."""
    # Find all protected regions
    start_positions = [m.start() for m in re.finditer(re.escape(PROTECTED_START), body)]
    end_positions = [m.start() for m in re.finditer(re.escape(PROTECTED_END), body)]

    for start, end in zip(start_positions, end_positions):
        if start <= position <= end:
            return True
    return False


def _get_protected_regions(body: str) -> list[tuple[int, int]]:
    """Return list of (start, end) positions of protected regions."""
    regions = []
    start_positions = [m.start() for m in re.finditer(re.escape(PROTECTED_START), body)]
    end_positions = [m.start() for m in re.finditer(re.escape(PROTECTED_END), body)]

    for start, end in zip(start_positions, end_positions):
        regions.append((start, end))
    return regions


def _count_changes(old_body: str, new_body: str) -> int:
    """Count the number of line-level changes between old and new body."""
    old_lines = old_body.splitlines()
    new_lines = new_body.splitlines()

    # Simple diff: count lines that changed
    changes = 0
    max_len = max(len(old_lines), len(new_lines))
    for i in range(max_len):
        old_line = old_lines[i] if i < len(old_lines) else ""
        new_line = new_lines[i] if i < len(new_lines) else ""
        if old_line != new_line:
            changes += 1
    return changes


def bump_version(current: str, bump_type: str = "patch") -> str:
    """Bump semver version string.

    Args:
        current: Current version string (e.g., "1.0.0")
        bump_type: Type of bump ("major", "minor", "patch")

    Returns:
        New version string
    """
    parts = current.split(".")
    if len(parts) != 3:
        return "1.0.1"

    try:
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return "1.0.1"

    if bump_type == "major":
        return f"{major + 1}.0.0"
    elif bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    else:  # patch
        return f"{major}.{minor}.{patch + 1}"


# Keep old name as alias for backward compatibility
_bump_version = bump_version


def apply_edits(
    body: str,
    edits: list[SkillEdit],
    *,
    edit_budget: int = DEFAULT_EDIT_BUDGET,
    current_version: str = "1.0.0",
) -> EditResult:
    """Apply bounded edits to a skill body.

    Rules:
    1. Total edits cannot exceed edit_budget (textual learning rate)
    2. Protected regions are never modified
    3. Each edit is validated before application
    4. Rejected edits are collected with reasons

    Args:
        body: Current skill body text
        edits: List of proposed edits
        edit_budget: Maximum number of edits allowed (default 3)
        current_version: Current semver version string

    Returns:
        EditResult with new body, applied/rejected edits, and version bump type
    """
    if not edits:
        return EditResult(success=True, new_body=body, version_bump="patch")

    # Clip edits to budget
    edits_to_apply = edits[:edit_budget]
    budget_rejected = edits[edit_budget:]
    rejected: list[tuple[SkillEdit, str]] = [
        (e, f"exceeds edit budget ({edit_budget})") for e in budget_rejected
    ]

    current_body = body
    applied: list[SkillEdit] = []
    version_bump = "patch"

    for edit in edits_to_apply:
        try:
            # Check protected regions FIRST (before applying)
            if _touches_protected(current_body, edit):
                rejected.append((edit, "touches protected region"))
                continue

            result_body = _apply_single_edit(current_body, edit)
            if result_body is not None:
                current_body = result_body
                applied.append(edit)

                # Determine version bump type
                if edit.op == EditOp.REPLACE and len(edit.content) > 100:
                    version_bump = "minor"
                elif edit.op == EditOp.DELETE:
                    version_bump = "minor"
            else:
                rejected.append((edit, "edit could not be applied (target not found)"))
        except Exception as e:
            rejected.append((edit, f"error: {e!s}"))

    # Determine if we need a major bump (significant content change)
    if applied:
        changes = _count_changes(body, current_body)
        if changes > 10:
            version_bump = "major"

    return EditResult(
        success=len(applied) > 0,
        new_body=current_body,
        applied_edits=applied,
        rejected_edits=rejected,
        version_bump=version_bump,
    )


def _touches_protected(body: str, edit: SkillEdit) -> bool:
    """Check if an edit would modify content inside a protected region."""
    if edit.op == EditOp.APPEND:
        # Append never touches existing content
        return False

    if edit.op == EditOp.INSERT_AFTER:
        if not edit.target:
            return False
        idx = body.find(edit.target)
        if idx < 0:
            return False
        insert_pos = idx + len(edit.target)
        return _is_in_protected_region(body, insert_pos)

    if edit.op in (EditOp.REPLACE, EditOp.DELETE):
        if not edit.target:
            return False
        idx = body.find(edit.target)
        if idx < 0:
            return False
        return _is_in_protected_region(body, idx)

    return False


def _apply_single_edit(body: str, edit: SkillEdit) -> str | None:
    """Apply a single edit to the body. Returns new body or None if cannot apply."""
    if edit.op == EditOp.APPEND:
        content = edit.content.rstrip() + "\n"
        return body.rstrip() + "\n" + content

    elif edit.op == EditOp.INSERT_AFTER:
        if not edit.target:
            return None
        idx = body.find(edit.target)
        if idx < 0:
            return None
        insert_pos = idx + len(edit.target)
        return body[:insert_pos] + "\n" + edit.content + body[insert_pos:]

    elif edit.op == EditOp.REPLACE:
        if not edit.target:
            return None
        if edit.target not in body:
            return None
        return body.replace(edit.target, edit.content, 1)

    elif edit.op == EditOp.DELETE:
        if not edit.target:
            return None
        if edit.target not in body:
            return None
        return body.replace(edit.target, "", 1)

    return None


def format_edit_history(history: list[dict[str, Any]] | None) -> str:
    """Format edit history for prompt injection."""
    if not history:
        return ""

    lines = ["<edit_history>"]
    for entry in history[-5:]:  # Last 5 edits
        op = entry.get("op", "unknown")
        timestamp = entry.get("timestamp", "")
        reason = entry.get("reason", "")
        lines.append(f"  - [{op}] {timestamp}: {reason[:100]}")
    lines.append("</edit_history>")
    return "\n".join(lines)


def format_rejected_edits(rejected: list[dict[str, Any]] | None) -> str:
    """Format rejected edits as negative feedback for prompt injection."""
    if not rejected:
        return ""

    lines = ["<rejected_edits_feedback>"]
    lines.append(
        "The following edits were previously rejected. DO NOT propose similar changes:"
    )
    for entry in rejected[-5:]:  # Last 5 rejected
        op = entry.get("op", "unknown")
        reason = entry.get("reason", "")
        target = str(entry.get("target", "") or "")[:50]
        lines.append(f"  - [{op}] target={target!r}: {reason}")
    lines.append("</rejected_edits_feedback>")
    return "\n".join(lines)


def create_edit_from_proposal(
    proposal: dict[str, Any],
) -> SkillEdit | None:
    """Convert an LLM proposal dict into a SkillEdit."""
    op_str = proposal.get("op", "")
    try:
        op = EditOp(op_str)
    except ValueError:
        return None

    return SkillEdit(
        op=op,
        target=proposal.get("target"),
        content=proposal.get("content", ""),
        reason=proposal.get("reason", ""),
    )
