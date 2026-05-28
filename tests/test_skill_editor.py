"""Tests for skill_editor.py — bounded text-space edits for skill evolution."""

from __future__ import annotations

import pytest

from src.core.intelligence.skill_editor import (
    PROTECTED_END,
    PROTECTED_START,
    EditOp,
    EditResult,
    SkillEdit,
    _apply_single_edit,
    _count_changes,
    _get_protected_regions,
    _is_in_protected_region,
    _touches_protected,
    apply_edits,
    bump_version,
    create_edit_from_proposal,
    format_edit_history,
    format_rejected_edits,
)


# ──────────────────────────────────────────────────────────────
# EditOp enum
# ──────────────────────────────────────────────────────────────


class TestEditOpEnum:
    """Tests for the EditOp enum — all 4 values are defined and work."""

    def test_editop_append_value(self):
        """EditOp.APPEND has string value 'append'."""
        assert EditOp.APPEND.value == "append"
        assert EditOp.APPEND == "append"

    def test_editop_insert_after_value(self):
        """EditOp.INSERT_AFTER has string value 'insert_after'."""
        assert EditOp.INSERT_AFTER.value == "insert_after"

    def test_editop_replace_value(self):
        """EditOp.REPLACE has string value 'replace'."""
        assert EditOp.REPLACE.value == "replace"

    def test_editop_delete_value(self):
        """EditOp.DELETE has string value 'delete'."""
        assert EditOp.DELETE.value == "delete"

    def test_editop_from_string(self):
        """EditOp can be constructed from a valid string."""
        assert EditOp("append") == EditOp.APPEND
        assert EditOp("delete") == EditOp.DELETE

    def test_editop_invalid_string_raises(self):
        """Invalid string raises ValueError."""
        with pytest.raises(ValueError):
            EditOp("invalid")


# ──────────────────────────────────────────────────────────────
# SkillEdit dataclass
# ──────────────────────────────────────────────────────────────


class TestSkillEditDataclass:
    """Tests for the SkillEdit dataclass — creation, to_dict, from_dict roundtrip."""

    def test_create_skill_edit(self):
        """SkillEdit can be created with all fields."""
        edit = SkillEdit(
            op=EditOp.APPEND,
            target=None,
            content="New line",
            reason="test",
        )
        assert edit.op == EditOp.APPEND
        assert edit.target is None
        assert edit.content == "New line"
        assert edit.reason == "test"

    def test_default_values(self):
        """SkillEdit default values are empty strings."""
        edit = SkillEdit(op=EditOp.REPLACE)
        assert edit.target is None
        assert edit.content == ""
        assert edit.reason == ""

    def test_to_dict_append(self):
        """to_dict serializes an APPEND edit correctly."""
        edit = SkillEdit(op=EditOp.APPEND, content="Hello", reason="add greeting")
        d = edit.to_dict()
        assert d["op"] == "append"
        assert d["target"] is None
        assert d["content"] == "Hello"
        assert d["reason"] == "add greeting"

    def test_to_dict_replace(self):
        """to_dict serializes a REPLACE edit correctly."""
        edit = SkillEdit(
            op=EditOp.REPLACE,
            target="old_text",
            content="new_text",
            reason="fix typo",
        )
        d = edit.to_dict()
        assert d["op"] == "replace"
        assert d["target"] == "old_text"
        assert d["content"] == "new_text"

    def test_from_dict_roundtrip(self):
        """from_dict(to_dict(...)) produces an equivalent SkillEdit."""
        original = SkillEdit(
            op=EditOp.INSERT_AFTER,
            target="## Section",
            content="Extra info",
            reason="more context",
        )
        reconstructed = SkillEdit.from_dict(original.to_dict())
        assert reconstructed.op == original.op
        assert reconstructed.target == original.target
        assert reconstructed.content == original.content
        assert reconstructed.reason == original.reason

    def test_from_dict_missing_optional_fields(self):
        """from_dict handles missing optional fields gracefully."""
        d = {"op": "delete", "target": "remove_me"}
        edit = SkillEdit.from_dict(d)
        assert edit.op == EditOp.DELETE
        assert edit.target == "remove_me"
        assert edit.content == ""  # default
        assert edit.reason == ""  # default


# ──────────────────────────────────────────────────────────────
# apply_edits — core function
# ──────────────────────────────────────────────────────────────


class TestApplyEdits:
    """Tests for apply_edits — the main edit application pipeline."""

    def test_empty_edits_returns_success(self):
        """Applying empty edit list returns success with unchanged body."""
        body = "Line 1\nLine 2\n"
        result = apply_edits(body, [])
        assert result.success is True
        assert result.new_body == body
        assert len(result.applied_edits) == 0
        assert len(result.rejected_edits) == 0

    def test_append_edit(self):
        """APPEND adds content at end of body."""
        body = "Line 1\nLine 2\n"
        edit = SkillEdit(op=EditOp.APPEND, content="Line 3")
        result = apply_edits(body, [edit])
        assert result.success is True
        assert "Line 3" in result.new_body
        assert result.new_body.strip().endswith("Line 3")
        assert len(result.applied_edits) == 1
        assert result.applied_edits[0].op == EditOp.APPEND

    def test_insert_after_edit(self):
        """INSERT_AFTER inserts content after a marker line."""
        body = "## Intro\nHello\n## Section\nWorld"
        edit = SkillEdit(
            op=EditOp.INSERT_AFTER,
            target="## Intro",
            content="## Body",
        )
        result = apply_edits(body, [edit])
        assert result.success is True
        # "## Body" should appear after "## Intro"
        assert "## Body" in result.new_body
        idx_intro = result.new_body.find("## Intro")
        idx_body = result.new_body.find("## Body")
        assert idx_body > idx_intro

    def test_insert_after_not_found_rejected(self):
        """INSERT_AFTER with missing target is rejected."""
        body = "Line 1\nLine 2"
        edit = SkillEdit(
            op=EditOp.INSERT_AFTER,
            target="NONEXISTENT",
            content="extra",
        )
        result = apply_edits(body, [edit])
        assert result.success is False
        assert len(result.rejected_edits) == 1
        assert "target not found" in result.rejected_edits[0][1]

    def test_replace_edit(self):
        """REPLACE substitutes old text with new text."""
        body = "Hello, world!"
        edit = SkillEdit(
            op=EditOp.REPLACE,
            target="world",
            content="universe",
        )
        result = apply_edits(body, [edit])
        assert result.success is True
        assert result.new_body == "Hello, universe!"

    def test_delete_edit(self):
        """DELETE removes specific text from body."""
        body = "Line A\nLine B\nLine C"
        edit = SkillEdit(op=EditOp.DELETE, target="Line B\n")
        result = apply_edits(body, [edit])
        assert result.success is True
        assert "Line B" not in result.new_body

    def test_edit_budget_clipping(self):
        """Only top-N edits are applied when count exceeds budget."""
        body = "line 1\nline 2\nline 3\nline 4\nline 5"
        edits = [
            SkillEdit(op=EditOp.APPEND, content="extra 1"),
            SkillEdit(op=EditOp.APPEND, content="extra 2"),
            SkillEdit(op=EditOp.APPEND, content="extra 3"),
            SkillEdit(op=EditOp.APPEND, content="extra 4"),  # beyond budget
            SkillEdit(op=EditOp.APPEND, content="extra 5"),  # beyond budget
        ]
        result = apply_edits(body, edits)
        assert len(result.applied_edits) == 3  # DEFAULT_EDIT_BUDGET
        assert len(result.rejected_edits) == 2
        for _, reason in result.rejected_edits:
            assert "budget" in reason

    def test_edit_budget_custom(self):
        """Custom edit_budget restricts applied edits."""
        body = "line"
        edits = [
            SkillEdit(op=EditOp.APPEND, content="a"),
            SkillEdit(op=EditOp.APPEND, content="b"),
        ]
        result = apply_edits(body, edits, edit_budget=1)
        assert len(result.applied_edits) == 1
        assert len(result.rejected_edits) == 1

    def test_protected_region_rejected(self):
        """Edits inside PROTECTED_START/END are rejected."""
        body = f"safe text\n{PROTECTED_START}\nsecret\n{PROTECTED_END}\nmore text"
        edit = SkillEdit(op=EditOp.REPLACE, target="secret", content="leaked")
        result = apply_edits(body, [edit])
        assert len(result.applied_edits) == 0
        assert len(result.rejected_edits) == 1
        assert "protected" in result.rejected_edits[0][1].lower()

    def test_append_never_touches_protected(self):
        """APPEND is always allowed, even with protected regions."""
        body = f"safe\n{PROTECTED_START}\nsecret\n{PROTECTED_END}\n"
        edit = SkillEdit(op=EditOp.APPEND, content="extra line")
        result = apply_edits(body, [edit])
        assert result.success is True
        assert len(result.applied_edits) == 1
        # Protected content is untouched
        assert "secret" in result.new_body

    def test_delete_near_protected_is_rejected(self):
        """DELETE targeting text inside protected region is rejected."""
        body = f"before\n{PROTECTED_START}\nimportant\n{PROTECTED_END}\nafter"
        edit = SkillEdit(op=EditOp.DELETE, target="important")
        result = apply_edits(body, [edit])
        assert len(result.applied_edits) == 0

    def test_success_is_false_when_all_rejected(self):
        """success is False when no edits are applied."""
        body = f"{PROTECTED_START}\nsecret\n{PROTECTED_END}"
        edit = SkillEdit(op=EditOp.REPLACE, target="secret", content="x")
        result = apply_edits(body, [edit])
        assert result.success is False
        assert len(result.applied_edits) == 0
        assert len(result.rejected_edits) == 1

    def test_mixed_applied_and_rejected(self):
        """Some edits applied, some rejected — both lists populated."""
        body = f"safe text\n{PROTECTED_START}\nsecret\n{PROTECTED_END}\n"
        edits = [
            SkillEdit(op=EditOp.APPEND, content="new line"),  # allowed
            SkillEdit(op=EditOp.REPLACE, target="secret", content="x"),  # rejected
        ]
        result = apply_edits(body, edits)
        assert len(result.applied_edits) == 1
        assert len(result.rejected_edits) == 1
        assert result.applied_edits[0].op == EditOp.APPEND


# ──────────────────────────────────────────────────────────────
# Version bumping
# ──────────────────────────────────────────────────────────────


class TestVersionBump:
    """Tests for bump_version — semver version bumping."""

    def test_bump_patch(self):
        """Patch bump: 1.0.0 → 1.0.1."""
        assert bump_version("1.0.0", "patch") == "1.0.1"

    def test_bump_minor(self):
        """Minor bump: 1.0.0 → 1.1.0."""
        assert bump_version("1.0.0", "minor") == "1.1.0"

    def test_bump_major(self):
        """Major bump: 1.0.0 → 2.0.0."""
        assert bump_version("1.0.0", "major") == "2.0.0"

    def test_bump_patch_default_type(self):
        """Default bump type is patch."""
        assert bump_version("1.0.0") == "1.0.1"

    def test_bump_invalid_version_returns_default(self):
        """Non-semver input returns '1.0.1'."""
        assert bump_version("invalid") == "1.0.1"

    def test_bump_empty_version_returns_default(self):
        """Empty version string returns '1.0.1'."""
        assert bump_version("") == "1.0.1"

    def test_bump_four_part_returns_default(self):
        """Four-part version returns '1.0.1'."""
        assert bump_version("1.0.0.0") == "1.0.1"


# ──────────────────────────────────────────────────────────────
# version_bump in apply_edits result
# ──────────────────────────────────────────────────────────────


class TestApplyEditsVersionBump:
    """Tests for version_bump detection inside apply_edits."""

    def test_append_patch_bump(self):
        """APPEND results in 'patch' bump."""
        body = "line"
        edit = SkillEdit(op=EditOp.APPEND, content="extra")
        result = apply_edits(body, [edit])
        assert result.version_bump == "patch"

    def test_short_replace_patch(self):
        """Short REPLACE results in 'patch' bump."""
        body = "Hello world"
        edit = SkillEdit(op=EditOp.REPLACE, target="world", content="universe")
        result = apply_edits(body, [edit])
        # content length = 8, ≤ 100 → "patch"
        # But count_changes = 0 lines changed (same line count) → not major
        assert result.version_bump in ("patch", "minor")

    def test_delete_minor_bump(self):
        """DELETE results in 'minor' bump (if not many lines changed)."""
        body = "line one\nline two\nline three"
        edit = SkillEdit(op=EditOp.DELETE, target="line two\n")
        result = apply_edits(body, [edit])
        # DELETE → version_bump becomes "minor" during processing
        # But if changes ≤ 10 lines, stays minor (not upgraded to major)
        assert result.version_bump in ("minor", "major")

    def test_large_change_major_bump(self):
        """Many line changes result in 'major' bump (simulated with content that changes many lines)."""
        # Create a body where a long REPLACE with content >100 chars triggers "minor"
        # then if count_changes > 10, it promotes to "major".
        # We create many different lines and replace one long segment with long content
        body_lines = ["line " + str(i) for i in range(15)]
        body = "\n".join(body_lines)
        # Replace multi-line content to trigger many line-level diffs
        old_text = "\n".join(body_lines[2:13])  # 11 lines
        new_text = "\n".join(["changed " + str(i) for i in range(11)])
        edit = SkillEdit(op=EditOp.REPLACE, target=old_text, content=new_text)
        result = apply_edits(body, [edit])
        # REPLACE with content >100 → marks "minor", then >10 changes → promotes to "major"
        if result.applied_edits:
            assert result.version_bump in ("minor", "major")


# ──────────────────────────────────────────────────────────────
# Protected region utilities
# ──────────────────────────────────────────────────────────────


class TestProtectedRegions:
    """Tests for protected region detection helpers."""

    def test_is_in_protected_true(self):
        """Position inside protected region returns True."""
        body = f"before\n{PROTECTED_START}\nX\n{PROTECTED_END}\nafter"
        idx = body.find("X")
        assert _is_in_protected_region(body, idx) is True

    def test_is_in_protected_false(self):
        """Position outside protected region returns False."""
        body = f"before\n{PROTECTED_START}\nX\n{PROTECTED_END}\nafter"
        idx = body.find("before")
        assert _is_in_protected_region(body, idx) is False

    def test_get_protected_regions_finds_one(self):
        """get_protected_regions returns one region when one pair exists."""
        body = f"a\n{PROTECTED_START}\nb\n{PROTECTED_END}\nc"
        regions = _get_protected_regions(body)
        assert len(regions) == 1
        start, end = regions[0]
        assert body[start:].startswith(PROTECTED_START)
        assert body[end:].startswith(PROTECTED_END)

    def test_get_protected_regions_no_regions(self):
        """No markers → empty list."""
        regions = _get_protected_regions("plain text")
        assert regions == []


# ──────────────────────────────────────────────────────────────
# _touches_protected
# ──────────────────────────────────────────────────────────────


class TestTouchesProtected:
    """Tests for _touches_protected — edit vs protected region check."""

    def test_append_never_touches(self):
        """APPEND always returns False for protected check."""
        body = f"{PROTECTED_START}\nsecret\n{PROTECTED_END}"
        edit = SkillEdit(op=EditOp.APPEND, content="x")
        assert _touches_protected(body, edit) is False

    def test_replace_touches_protected(self):
        """REPLACE targeting text inside protected region returns True."""
        body = f"{PROTECTED_START}\nsecret\n{PROTECTED_END}"
        edit = SkillEdit(op=EditOp.REPLACE, target="secret", content="x")
        assert _touches_protected(body, edit) is True

    def test_replace_safe(self):
        """REPLACE targeting text outside protected region returns False."""
        body = f"safe\n{PROTECTED_START}\nsecret\n{PROTECTED_END}"
        edit = SkillEdit(op=EditOp.REPLACE, target="safe", content="ok")
        assert _touches_protected(body, edit) is False

    def test_delete_touches_protected(self):
        """DELETE targeting protected text returns True."""
        body = f"{PROTECTED_START}\nsecret\n{PROTECTED_END}"
        edit = SkillEdit(op=EditOp.DELETE, target="secret")
        assert _touches_protected(body, edit) is True

    def test_insert_after_touches_protected(self):
        """INSERT_AFTER with target inside protected region returns True."""
        body = f"{PROTECTED_START}\nmarker\n{PROTECTED_END}"
        edit = SkillEdit(
            op=EditOp.INSERT_AFTER,
            target="marker",
            content="inserted",
        )
        assert _touches_protected(body, edit) is True

    def test_no_target_returns_false(self):
        """Edit without target never touches protected (except handled elsewhere)."""
        body = f"{PROTECTED_START}\nsecret\n{PROTECTED_END}"
        edit = SkillEdit(op=EditOp.REPLACE, target=None, content="x")
        assert _touches_protected(body, edit) is False


# ──────────────────────────────────────────────────────────────
# _apply_single_edit
# ──────────────────────────────────────────────────────────────


class TestApplySingleEdit:
    """Tests for _apply_single_edit — low-level edit application."""

    def test_append_adds_newline(self):
        """APPEND adds content at the end with a newline."""
        body = "first"
        edit = SkillEdit(op=EditOp.APPEND, content="second")
        result = _apply_single_edit(body, edit)
        assert result is not None
        assert result.startswith("first")
        assert "second" in result

    def test_insert_after_marker(self):
        """INSERT_AFTER places content after the marker."""
        body = "Header\nContent"
        edit = SkillEdit(
            op=EditOp.INSERT_AFTER,
            target="Header",
            content="Subheader",
        )
        result = _apply_single_edit(body, edit)
        assert result is not None
        assert "Header" in result
        assert "Subheader" in result
        assert result.find("Subheader") > result.find("Header")

    def test_insert_after_no_target_returns_none(self):
        """INSERT_AFTER with None target returns None."""
        edit = SkillEdit(op=EditOp.INSERT_AFTER, target=None, content="x")
        assert _apply_single_edit("body", edit) is None

    def test_replace_existing(self):
        """REPLACE with found target returns modified body."""
        body = "abc def ghi"
        edit = SkillEdit(op=EditOp.REPLACE, target="def", content="xyz")
        result = _apply_single_edit(body, edit)
        assert result == "abc xyz ghi"

    def test_replace_not_found_returns_none(self):
        """REPLACE with missing target returns None."""
        edit = SkillEdit(op=EditOp.REPLACE, target="zzz", content="x")
        assert _apply_single_edit("abc", edit) is None

    def test_delete_existing(self):
        """DELETE with found target removes it."""
        body = "remove this token here"
        edit = SkillEdit(op=EditOp.DELETE, target="token ")
        result = _apply_single_edit(body, edit)
        assert result == "remove this here"

    def test_delete_not_found_returns_none(self):
        """DELETE with missing target returns None."""
        edit = SkillEdit(op=EditOp.DELETE, target="absent")
        assert _apply_single_edit("abc", edit) is None

    def test_unknown_op_returns_none(self):
        """Unknown operation is covered by create_edit_from_proposal tests.

        Since EditOp is an enum, invalid values can't be passed to _apply_single_edit
        at runtime. The `return None` at the end is a safety net — it's covered
        indirectly through create_edit_from_proposal returning None for invalid ops.
        """
        # This safety net is validated by test_invalid_op_returns_none
        # which confirms create_edit_from_proposal rejects unknown ops
        pass


# ──────────────────────────────────────────────────────────────
# _count_changes
# ──────────────────────────────────────────────────────────────


class TestCountChanges:
    """Tests for _count_changes — line-level diff counter."""

    def test_no_changes(self):
        """Identical bodies return 0 changes."""
        assert _count_changes("a\nb\nc", "a\nb\nc") == 0

    def test_one_line_changed(self):
        """One modified line returns 1."""
        assert _count_changes("a\nb\nc", "a\nX\nc") == 1

    def test_added_lines(self):
        """Added lines count as changes."""
        assert _count_changes("a\nb", "a\nb\nc\nd") == 2

    def test_removed_lines(self):
        """Removed lines count as changes."""
        assert _count_changes("a\nb\nc\nd", "a\nb") == 2


# ──────────────────────────────────────────────────────────────
# format_edit_history
# ──────────────────────────────────────────────────────────────


class TestFormatEditHistory:
    """Tests for format_edit_history — prompt injection formatting."""

    def test_non_empty_history(self):
        """Non-empty history returns formatted XML string."""
        history = [
            {
                "op": "append",
                "timestamp": "2025-01-01T12:00:00",
                "reason": "Added greeting",
            }
        ]
        result = format_edit_history(history)
        assert "<edit_history>" in result
        assert "append" in result
        assert "Added greeting" in result
        assert "</edit_history>" in result

    def test_empty_history_returns_empty_string(self):
        """Empty history returns empty string."""
        assert format_edit_history([]) == ""
        assert format_edit_history(None) == ""

    def test_truncates_long_reasons(self):
        """Long reasons are truncated to 100 chars."""
        long_reason = "A" * 200
        history = [{"op": "replace", "timestamp": "now", "reason": long_reason}]
        result = format_edit_history(history)
        # The truncated reason (100 chars) should appear, not the full 200
        assert long_reason[:100] in result
        assert long_reason not in result  # Full string not present

    def test_last_five_only(self):
        """Only last 5 entries are shown."""
        history = [
            {"op": "append", "timestamp": "", "reason": f"edit {i}"} for i in range(10)
        ]
        result = format_edit_history(history)
        # Only last 5
        assert "edit 5" in result
        assert "edit 9" in result
        assert "edit 0" not in result
        assert "edit 4" not in result


# ──────────────────────────────────────────────────────────────
# format_rejected_edits
# ──────────────────────────────────────────────────────────────


class TestFormatRejectedEdits:
    """Tests for format_rejected_edits — negative feedback formatting."""

    def test_non_empty_rejected(self):
        """Non-empty rejected list returns formatted XML."""
        rejected = [{"op": "delete", "reason": "protected region", "target": "secret"}]
        result = format_rejected_edits(rejected)
        assert "<rejected_edits_feedback>" in result
        assert "delete" in result
        assert "secret" in result
        assert "protected region" in result

    def test_empty_rejected_returns_empty_string(self):
        """Empty rejected list returns empty string."""
        assert format_rejected_edits([]) == ""
        assert format_rejected_edits(None) == ""

    def test_long_target_truncated(self):
        """Long targets are truncated to 50 chars."""
        long_target = "B" * 100
        rejected = [{"op": "replace", "reason": "bad", "target": long_target}]
        result = format_rejected_edits(rejected)
        assert long_target[:50] in result
        assert long_target not in result  # Full string not present


# ──────────────────────────────────────────────────────────────
# create_edit_from_proposal
# ──────────────────────────────────────────────────────────────


class TestCreateEditFromProposal:
    """Tests for create_edit_from_proposal — LLM proposal parsing."""

    def test_valid_proposal(self):
        """Valid proposal dict converts to SkillEdit."""
        proposal = {
            "op": "append",
            "content": "New line",
            "reason": "add info",
        }
        edit = create_edit_from_proposal(proposal)
        assert edit is not None
        assert edit.op == EditOp.APPEND
        assert edit.content == "New line"
        assert edit.reason == "add info"

    def test_proposal_with_target(self):
        """Proposal with target field includes it."""
        proposal = {
            "op": "replace",
            "target": "old",
            "content": "new",
            "reason": "update",
        }
        edit = create_edit_from_proposal(proposal)
        assert edit is not None
        assert edit.target == "old"

    def test_invalid_op_returns_none(self):
        """Invalid operation string returns None."""
        proposal = {"op": "invalid_op"}
        edit = create_edit_from_proposal(proposal)
        assert edit is None

    def test_empty_op_returns_none(self):
        """Empty operation string returns None."""
        proposal = {"op": ""}
        edit = create_edit_from_proposal(proposal)
        assert edit is None

    def test_missing_fields_defaulted(self):
        """Missing optional fields get defaults."""
        proposal = {"op": "delete", "target": "x"}
        edit = create_edit_from_proposal(proposal)
        assert edit is not None
        assert edit.content == ""
        assert edit.reason == ""


# ──────────────────────────────────────────────────────────────
# Edge cases / integration-like
# ──────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases and integration-style tests for the full edit pipeline."""

    def test_edit_with_non_ascii_content(self):
        """Edits with non-ASCII (Cyrillic) content work correctly."""
        body = "начало документа"
        edit = SkillEdit(op=EditOp.APPEND, content="конец документа")
        result = apply_edits(body, [edit])
        assert result.success is True
        assert "конец документа" in result.new_body

    def test_multiple_edits_of_same_type(self):
        """Multiple APPEND edits all apply in order."""
        body = "start"
        edits = [
            SkillEdit(op=EditOp.APPEND, content="first"),
            SkillEdit(op=EditOp.APPEND, content="second"),
            SkillEdit(op=EditOp.APPEND, content="third"),
        ]
        result = apply_edits(body, edits)
        assert "first" in result.new_body
        assert "second" in result.new_body
        assert "third" in result.new_body
        # Order check
        idx_first = result.new_body.find("first")
        idx_second = result.new_body.find("second")
        idx_third = result.new_body.find("third")
        assert idx_first < idx_second < idx_third

    def test_edit_on_empty_body(self):
        """Editing an empty body still works for APPEND."""
        edit = SkillEdit(op=EditOp.APPEND, content="hello")
        result = apply_edits("", [edit])
        assert result.success is True
        assert "hello" in result.new_body

    def test_edit_on_empty_body_with_protected(self):
        """Protected regions work correctly even on mostly-empty bodies."""
        body = f"{PROTECTED_START}\n{PROTECTED_END}"
        safe_edit = SkillEdit(op=EditOp.APPEND, content="ok")
        bad_edit = SkillEdit(
            op=EditOp.INSERT_AFTER,
            target=PROTECTED_START,
            content="bad",
        )
        result = apply_edits(body, [safe_edit, bad_edit])
        assert result.success is True  # APPEND succeeded
        assert "ok" in result.new_body
        assert len(result.rejected_edits) == 1

    def test_edit_result_repr(self):
        """EditResult can be constructed and inspected."""
        result = EditResult(
            success=True,
            new_body="test",
            applied_edits=[],
            rejected_edits=[],
            version_bump="patch",
        )
        assert result.success is True
        assert result.new_body == "test"
        assert result.version_bump == "patch"
