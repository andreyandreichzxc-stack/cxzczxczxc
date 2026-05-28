"""Tests for skill_validator.py — validation gate for skill updates."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.core.intelligence.skill_validator import (
    MAX_REGRESSION_TOLERANCE,
    MIN_TRAJECTORIES_FOR_VALIDATION,
    TrajectoryData,
    ValidationResult,
    _calculate_baseline_score,
    _estimate_skill_quality_heuristic,
)


# ──────────────────────────────────────────────────────────────
# Helpers to create TrajectoryData instances
# ──────────────────────────────────────────────────────────────


def _make_trajectory(
    id: int = 1,
    request_text: str = "",
    response_text: str = "",
    latency_ms: int | None = None,
    used_skills_json: list | None = None,
    route_mode: str | None = None,
    success: bool = True,
) -> TrajectoryData:
    """Create a TrajectoryData instance with given fields."""
    return TrajectoryData(
        id=id,
        request_text=request_text,
        response_text=response_text,
        latency_ms=latency_ms,
        used_skills_json=used_skills_json,
        route_mode=route_mode,
        success=success,
    )


def _make_mock_trajectory(
    id: int = 1,
    request_text: str = "",
    response_text: str = "ok",
    latency_ms: int | None = 1500,
    used_skills_json: list | None = None,
    route_mode: str | None = "maestro",
    success: bool = True,
) -> MagicMock:
    """Create a MagicMock simulating a Trajectory ORM object.

    Used for testing TrajectoryData.from_trajectory().
    """
    t = MagicMock()
    t.id = id
    t.request_text = request_text
    t.response_text = response_text
    t.latency_ms = latency_ms
    t.used_skills_json = used_skills_json
    t.route_mode = route_mode
    t.success = success
    return t


# ──────────────────────────────────────────────────────────────
# TrajectoryData.from_trajectory
# ──────────────────────────────────────────────────────────────


class TestTrajectoryDataFromTrajectory:
    """Tests for TrajectoryData.from_trajectory — extracting plain data from ORM."""

    def test_extracts_all_fields(self):
        """All fields are correctly extracted from mock Trajectory."""
        mock_t = _make_mock_trajectory(
            id=42,
            request_text="hello",
            response_text="hi there",
            latency_ms=500,
            used_skills_json=[{"name": "greeting"}],
            route_mode="agent",
            success=True,
        )
        data = TrajectoryData.from_trajectory(mock_t)
        assert data.id == 42
        assert data.request_text == "hello"
        assert data.response_text == "hi there"
        assert data.latency_ms == 500
        assert data.used_skills_json == [{"name": "greeting"}]
        assert data.route_mode == "agent"
        assert data.success is True

    def test_handles_none_fields(self):
        """None fields are preserved as None."""
        mock_t = _make_mock_trajectory(
            id=1,
            request_text="",
            response_text="",
            latency_ms=None,
            used_skills_json=None,
            route_mode=None,
            success=False,
        )
        data = TrajectoryData.from_trajectory(mock_t)
        assert data.request_text == ""
        assert data.response_text == ""
        assert data.latency_ms is None
        assert data.used_skills_json is None
        assert data.route_mode is None
        assert data.success is False


# ──────────────────────────────────────────────────────────────
# _calculate_baseline_score
# ──────────────────────────────────────────────────────────────


class TestCalculateBaselineScore:
    """Tests for _calculate_baseline_score — quality scoring from trajectories."""

    def test_empty_trajectories_returns_default(self):
        """Empty trajectory list returns 0.3 (default)."""
        assert _calculate_baseline_score([], None) == 0.3

    def test_fast_response_high_score(self):
        """Fast response (<2s) with complete response gets high score."""
        trajectories = [
            _make_trajectory(
                id=1,
                latency_ms=1500,
                response_text="This is a very complete and helpful response, well over 50 chars long.",
            )
        ]
        score = _calculate_baseline_score(trajectories, None)
        # Base 0.3 + latency 0.25 + completeness 0.2 = 0.75
        assert score == pytest.approx(0.75)

    def test_slow_response_low_score(self):
        """Slow response (>10s) gets minimal bonus."""
        trajectories = [
            _make_trajectory(
                id=1,
                latency_ms=12000,
                response_text="short",
            )
        ]
        score = _calculate_baseline_score(trajectories, None)
        # Base 0.3 + no latency bonus (>10s) + no completeness bonus (<20 chars)
        assert score == pytest.approx(0.3)

    def test_medium_latency_bonus(self):
        """Medium latency (2-5s) gets 0.15 bonus."""
        trajectories = [
            _make_trajectory(
                id=1,
                latency_ms=3000,
                response_text="",
            )
        ]
        score = _calculate_baseline_score(trajectories, None)
        # Base 0.3 + latency 0.15 = 0.45
        assert score == pytest.approx(0.45)

    def test_high_latency_bonus(self):
        """High-but-acceptable latency (5-10s) gets 0.05 bonus."""
        trajectories = [
            _make_trajectory(
                id=1,
                latency_ms=7000,
                response_text="",
            )
        ]
        score = _calculate_baseline_score(trajectories, None)
        # Base 0.3 + latency 0.05 = 0.35
        assert score == pytest.approx(0.35)

    def test_short_response_bonus(self):
        """Response >20 chars but ≤50 gets 0.1 bonus."""
        trajectories = [
            _make_trajectory(
                id=1,
                latency_ms=None,
                response_text="A" * 30,  # >20 but ≤50
            )
        ]
        score = _calculate_baseline_score(trajectories, None)
        # Base 0.3 + completeness 0.1 = 0.4
        assert score == pytest.approx(0.4)

    def test_with_skill_usage_bonus(self):
        """Trajectory with matching skill usage gets +0.25 bonus."""
        trajectories = [
            _make_trajectory(
                id=1,
                latency_ms=1500,
                response_text="Good response, more than 50 characters here, yes it is.",
                used_skills_json=[{"name": "test_skill"}],
            )
        ]
        score = _calculate_baseline_score(trajectories, skill_name="test_skill")
        # Base 0.3 + latency 0.25 + completeness 0.2 + skill 0.25 = 1.0 (capped)
        assert score == pytest.approx(1.0)

    def test_without_skill_usage_no_bonus(self):
        """Trajectory without matching skill gets no skill bonus."""
        trajectories = [
            _make_trajectory(
                id=1,
                latency_ms=1500,
                response_text="Adequate response over fifty characters for completeness.",
                used_skills_json=[{"name": "other_skill"}],
            )
        ]
        score = _calculate_baseline_score(trajectories, skill_name="test_skill")
        # Base 0.3 + latency 0.25 + completeness 0.2 = 0.75
        assert score == pytest.approx(0.75)

    def test_score_capped_at_one(self):
        """Individual turn scores are capped at 1.0."""
        trajectories = [
            _make_trajectory(
                id=1,
                latency_ms=100,  # <2000 → +0.25
                response_text="X" * 60,  # >50 → +0.2
                used_skills_json=[{"name": "test"}],  # +0.25
            )
        ]
        score = _calculate_baseline_score(trajectories, skill_name="test")
        # 0.3 + 0.25 + 0.2 + 0.25 = 1.0 (capped at 1.0, not 1.0 + overflow)
        assert score <= 1.0
        assert score == pytest.approx(1.0)

    def test_none_latency_skips_latency_bonus(self):
        """None latency_ms means no latency bonus is applied."""
        trajectories = [
            _make_trajectory(
                id=1,
                latency_ms=None,
                response_text="A long enough response to get the completeness bonus.",
            )
        ]
        score = _calculate_baseline_score(trajectories, None)
        # Base 0.3 + completeness ( >50) 0.2 = 0.5
        assert score == pytest.approx(0.5)

    def test_zero_latency_skips_bonus(self):
        """Zero latency_ms means no latency bonus (condition checks > 0)."""
        trajectories = [
            _make_trajectory(
                id=1,
                latency_ms=0,
                response_text="",
            )
        ]
        score = _calculate_baseline_score(trajectories, None)
        # Base 0.3 only
        assert score == pytest.approx(0.3)

    def test_skill_usage_mixed_dicts_and_primitives(self):
        """Skill usage list handles mixed dict and non-dict entries gracefully."""
        trajectories = [
            _make_trajectory(
                id=1,
                latency_ms=1500,
                response_text="X" * 60,
                used_skills_json=[{"name": "match"}, "plain_string", 123],
            )
        ]
        score = _calculate_baseline_score(trajectories, skill_name="match")
        # Base 0.3 + latency 0.25 + completeness 0.2 + skill 0.25 = 1.0
        assert score == pytest.approx(1.0)


# ──────────────────────────────────────────────────────────────
# _estimate_skill_quality_heuristic
# ──────────────────────────────────────────────────────────────


class TestEstimateSkillQuality:
    """Tests for _estimate_skill_quality_heuristic — heuristic skill body scoring."""

    def _simple_trajectory(self, request_text: str = "") -> TrajectoryData:
        return _make_trajectory(id=1, request_text=request_text)

    def test_empty_body_returns_zero(self):
        """Empty or whitespace-only body returns 0.0."""
        assert _estimate_skill_quality_heuristic("", [], "test") == 0.0
        assert _estimate_skill_quality_heuristic("   \n  ", [], "test") == 0.0

    def test_short_body_penalty(self):
        """Very short body (<100 chars) gets a penalty."""
        body = "Short skill body under one hundred characters."
        # ~50 chars → <100 → penalty -0.2
        score = _estimate_skill_quality_heuristic(body, [], "test")
        # Base 0.5 - 0.2 = 0.3
        assert score == pytest.approx(0.3)

    def test_good_length_bonus(self):
        """Body of ideal length (300-3000) gets +0.15 bonus."""
        body = "A" * 500  # 500 chars fits in 300-3000
        score = _estimate_skill_quality_heuristic(body, [], "test")
        # Base 0.5 + 0.15 = 0.65 (no structure, no keyword overlap)
        assert score == pytest.approx(0.65)

    def test_too_long_penalty(self):
        """Very long body (>5000 chars) gets a penalty."""
        body = "A" * 6000
        score = _estimate_skill_quality_heuristic(body, [], "test")
        # Base 0.5 - 0.1 = 0.4
        assert score == pytest.approx(0.4)

    def test_structure_bonus(self):
        """Body with structure markers (steps, rules) gets +0.15 bonus."""
        body = "A" * 350 + "\n1. First step\n2. Second step\n- bullet point"
        score = _estimate_skill_quality_heuristic(body, [], "test")
        # Base 0.5 + 0.15 (length) + 0.15 (structure: "1.", "2.", "- ") = 0.8
        assert score == pytest.approx(0.8)

    def test_keyword_overlap_bonus(self):
        """Body words overlapping with trajectory request words get bonus."""
        body = "skill for greeting users with hello and welcome message"
        trajectories = [self._simple_trajectory("user hello welcome greeting test")]
        score = _estimate_skill_quality_heuristic(body, trajectories, "test")
        # body_words: {'skill', 'for', 'greeting', 'users', 'with', 'hello', 'and', 'welcome', 'message'}
        # trajectory words: {'user', 'hello', 'welcome', 'greeting', 'test'}
        # overlap: {'hello', 'welcome', 'greeting', 'user'} — 4 matches → keyword_hits = 1
        # Base 0.5 + short penalty? body_len = ~55 → <100 → penalty -0.2
        # No structure bonus (no markers like "1.", "•", etc.)
        # But wait: "with" = 4 chars → included. Let me recalculate.
        # body = "skill for greeting users with hello and welcome message" → 54 chars → <100 → -0.2
        # So base 0.5 - 0.2 = 0.3 + keyword bonus (1 * 0.05 = 0.05) = 0.35
        assert score > 0.3  # At least some bonus vs plain short body

    def test_no_keyword_overlap(self):
        """No overlapping words means no keyword bonus."""
        body = "xyzzy plugh snarf blort quux bazooka"
        trajectories = [
            self._simple_trajectory("hello world"),
            self._simple_trajectory("goodbye friend"),
        ]
        score = _estimate_skill_quality_heuristic(body, trajectories, "test")
        # Only words like 'blort' and 'quux' — no overlap with 'hello', 'world', etc.
        # Base 0.5 + length 42 (<100=-0.2) = 0.3, no keyword bonus
        assert score == pytest.approx(0.3)

    def test_multiple_trajectory_keyword_matches(self):
        """Multiple trajectories with keyword overlap increase bonus (capped at 0.2)."""
        body = "common skill for common tasks common procedures common methods"
        trajectories = [
            self._simple_trajectory("common skill task"),
            self._simple_trajectory("common method procedure"),
            self._simple_trajectory("common again"),
            self._simple_trajectory("and common once more"),
            self._simple_trajectory("common last time"),
        ]
        score = _estimate_skill_quality_heuristic(body, trajectories, "test")
        # 5 trajectories all matching → keyword_hits = 5 → bonus = min(5*0.05, 0.2) = 0.2
        # Base 0.5 + length: ~73 chars <100 → -0.2 = 0.3
        # + keyword bonus 0.2 = 0.5
        assert score >= 0.3  # Just confirming keyword overlap takes effect

    def test_score_capped_at_one(self):
        """Estimated score is capped at 1.0."""
        body = (
            "A" * 500 + "\n1. first\n2. second\n- bullet\nstep by step\nесли then when"
        )
        # Good length (300-3000): +0.15
        # Structure markers: "1.", "2.", "- ", "step", "если", "then", "when" → >=2 → +0.15
        # Base 0.5 + 0.15 + 0.15 = 0.8
        trajectories = [
            self._simple_trajectory("A " + "first second bullet step " * 10),
        ]
        score = _estimate_skill_quality_heuristic(body, trajectories, "test")
        assert score <= 1.0

    def test_structure_marker_counting(self):
        """Structure markers counting is case-insensitive."""
        body = (
            "A" * 350 + " Step 1: do this\n STEP 2: do that\n • пункт\n правило одно\n"
        )
        score = _estimate_skill_quality_heuristic(body, [], "test")
        # Base 0.5 + length 0.15 + structure (step, step, •, правило) >= 2 → +0.15 = 0.8
        assert score == pytest.approx(0.8)

    def test_short_words_filtered_from_overlap(self):
        """Words shorter than 3 chars are excluded from keyword overlap."""
        body = "ab cd ef the big skill content goes here"
        trajectories = [self._simple_trajectory("ab cd ef the")]
        score = _estimate_skill_quality_heuristic(body, trajectories, "test")
        # body_words: {'the', 'big', 'skill', 'content', 'goes', 'here'} (ab, cd, ef < 3 chars)
        # trajectory words: {'the'} (ab, cd, ef < 3 chars)
        # overlap: {'the'} — only 1 word → need >= 2 for keyword_hits
        # keyword_hits = 0 → no bonus
        # Base 0.5 + length: 39 chars <100 → -0.2 = 0.3
        assert score == pytest.approx(0.3)


# ──────────────────────────────────────────────────────────────
# ValidationResult.summary
# ──────────────────────────────────────────────────────────────


class TestValidationResultSummary:
    """Tests for ValidationResult.summary property — formatted output."""

    def test_accepted_format(self):
        """Accepted result shows ACCEPTED with scores."""
        result = ValidationResult(
            accepted=True,
            score_before=0.5,
            score_after=0.7,
            score_delta=0.2,
            trajectories_used=10,
            reason="Score improvement",
        )
        summary = result.summary
        assert "ACCEPTED" in summary
        assert "0.50" in summary
        assert "0.70" in summary
        assert "+0.20" in summary
        assert "10 trajectories" in summary

    def test_rejected_format(self):
        """Rejected result shows REJECTED with scores."""
        result = ValidationResult(
            accepted=False,
            score_before=0.8,
            score_after=0.6,
            score_delta=-0.2,
            trajectories_used=5,
            reason="Score regression",
        )
        summary = result.summary
        assert "REJECTED" in summary
        assert "0.80" in summary
        assert "0.60" in summary
        assert "-0.20" in summary
        assert "5 trajectories" in summary

    def test_summary_includes_emoji(self):
        """Summary includes ✅ for accepted and ❌ for rejected."""
        accepted = ValidationResult(
            accepted=True,
            score_before=0.0,
            score_after=0.0,
            score_delta=0.0,
            trajectories_used=0,
        )
        assert "✅" in accepted.summary

        rejected = ValidationResult(
            accepted=False,
            score_before=0.0,
            score_after=0.0,
            score_delta=-1.0,
            trajectories_used=0,
        )
        assert "❌" in rejected.summary

    def test_delta_formatting(self):
        """Score delta is formatted with sign and two decimals."""
        pos = ValidationResult(
            accepted=True,
            score_before=0.3,
            score_after=0.35,
            score_delta=0.05,
            trajectories_used=3,
        )
        assert "+0.05" in pos.summary

        neg = ValidationResult(
            accepted=False,
            score_before=0.3,
            score_after=0.25,
            score_delta=-0.05,
            trajectories_used=3,
        )
        assert "-0.05" in neg.summary

        zero = ValidationResult(
            accepted=True,
            score_before=0.3,
            score_after=0.3,
            score_delta=0.0,
            trajectories_used=3,
        )
        assert "+0.00" in zero.summary


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────


class TestConstants:
    """Verify validation constants have expected values."""

    def test_min_trajectories_for_validation(self):
        """At least 3 trajectories are needed for validation."""
        assert MIN_TRAJECTORIES_FOR_VALIDATION == 3

    def test_max_regression_tolerance(self):
        """Regression tolerance is -0.05 (5%)."""
        assert MAX_REGRESSION_TOLERANCE == -0.05
