"""Unit tests for the pillar-composite blend (config#2678 deliverable 2).

Pins the go-live magnitude/guardrail decision: uniform pillar weighting,
``PILLAR_BLEND_WEIGHT`` fraction on the composite, and a hard
``MAX_RATING_CHANGE`` cap on how far the blend can move the rating.
"""

from __future__ import annotations

from nousergon_lib.pillars import MoatAssessment, PillarSubscore, QualitativePillarAssessment

from thinktank.pillars import MAX_RATING_CHANGE, PILLAR_BLEND_WEIGHT, blend_rating, pillar_composite_score


def _assessment(scores: dict[str, int], catalyst: int = 0) -> QualitativePillarAssessment:
    def sub(pillar: str) -> PillarSubscore:
        return PillarSubscore(pillar=pillar, score=scores[pillar], confidence="medium", evidence=[])

    return QualitativePillarAssessment(
        quality=sub("quality"),
        quality_moat=MoatAssessment(
            primary_type="none", width="none", durability_years=0, trend="stable"
        ),
        value=sub("value"),
        momentum=sub("momentum"),
        growth=sub("growth"),
        stewardship=sub("stewardship"),
        defensiveness=sub("defensiveness"),
        catalyst_horizon_modulation=catalyst,
    )


_UNIFORM_50 = dict.fromkeys(("quality", "value", "momentum", "growth", "stewardship", "defensiveness"), 50)


def test_pillar_composite_score_is_uniform_mean_plus_catalyst():
    scores = {"quality": 80, "value": 60, "momentum": 50, "growth": 50, "stewardship": 50, "defensiveness": 50}
    assessment = _assessment(scores, catalyst=10)
    # mean(80,60,50,50,50,50) = 56.666.. -> derive_legacy_qual_score rounds to 57; +10 = 67
    assert pillar_composite_score(assessment) == 67


def test_pillar_composite_score_clamped_to_0_100():
    assessment = _assessment(_UNIFORM_50, catalyst=20)
    assert pillar_composite_score(assessment) == 70
    hot = _assessment({**_UNIFORM_50, "quality": 100, "value": 100, "momentum": 100}, catalyst=20)
    assert pillar_composite_score(hot) <= 100


def test_blend_rating_none_pillar_assessment_passes_through():
    assert blend_rating(72, None) == 72


def test_blend_rating_matching_composite_is_a_no_op():
    # pillar composite == raw rating -> blend leaves it unchanged.
    assessment = _assessment(_UNIFORM_50)
    assert blend_rating(50, assessment) == 50


def test_blend_rating_moves_toward_composite_by_blend_weight():
    # raw=50, composite=90 (all pillars 90) -> blended = 0.85*50 + 0.15*90 = 56.0
    assessment = _assessment(dict.fromkeys(_UNIFORM_50, 90))
    expected_unclamped = round((1 - PILLAR_BLEND_WEIGHT) * 50 + PILLAR_BLEND_WEIGHT * 90)
    assert blend_rating(50, assessment) == expected_unclamped == 56


def test_blend_rating_never_exceeds_max_rating_change():
    # At the current PILLAR_BLEND_WEIGHT (0.15), the largest possible
    # unclamped swing (composite maximally far from raw, 0 vs 100) is
    # exactly MAX_RATING_CHANGE (15) — this pins that the two constants
    # stay in that relationship; a future PILLAR_BLEND_WEIGHT bump without
    # also revisiting MAX_RATING_CHANGE would make this guardrail bite for
    # real rather than just bound the theoretical max.
    assessment = _assessment(dict.fromkeys(_UNIFORM_50, 0))
    blended = blend_rating(100, assessment)
    assert abs(blended - 100) == MAX_RATING_CHANGE

    assessment_low = _assessment(dict.fromkeys(_UNIFORM_50, 100))
    blended_low = blend_rating(0, assessment_low)
    assert abs(blended_low - 0) == MAX_RATING_CHANGE


def test_blend_rating_stays_within_0_100_bounds():
    assessment = _assessment(dict.fromkeys(_UNIFORM_50, 100), catalyst=20)
    assert blend_rating(95, assessment) <= 100
    assessment_low = _assessment(dict.fromkeys(_UNIFORM_50, 0), catalyst=-20)
    assert blend_rating(5, assessment_low) >= 0
