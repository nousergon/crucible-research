"""Pillar-composite blend into Think Tank's operative rating.

Deliverable 2 of config#2678 (Brian's 2026-07-16 operator ruling): rather
than leave the qualitative pillar/moat signal permanently unused a second
time (the Phase 4 -> Phase 6 silent-zero pattern on the now-dormant
scanner/CIO composite — see ``private-docs/attractiveness-pillars-arc-
status.md``), the pillar composite goes live directly into Think Tank's
own rating — the value ``challenger_selection.py`` ranks by and the
leaderboard shadow view scores with (see ``thinktank/ratings.py::
_row_from_thesis``).

Go-live magnitude (Brian's ruling): start conservative. There is no
informative shadow-mode history to seed a magnitude from — the only
production window with non-zero pillar weights was the ~1hr 2026-05-21
AQR-cutover, which hit a 0%-coverage parse bug before any real signal
accumulated (see the incident section of the arc-status doc). So this
uses a small, UNIFORM weight across all 6 pillars (equal-weight mean via
``QualitativePillarAssessment.derive_legacy_qual_score()``) — deliberately
NOT the AQR quality-tilted split implicated in that incident, since this
is a fresh go-live on a different extraction substrate (Think Tank's own
``PILLAR_TIER`` call) with no validated tilt to reuse.
"""

from __future__ import annotations

from nousergon_lib.pillars import QualitativePillarAssessment

PILLAR_BLEND_WEIGHT = 0.15
"""Weight on the pillar composite in the blended rating; the remaining
(1 - PILLAR_BLEND_WEIGHT) stays on the analyst's own raw LLM rating.
Conservative starting magnitude per the operator ruling above — a future
re-tune should be informed by realized rating_minus_attractiveness /
challenger-arm performance data, not by guessing a bigger number."""

MAX_RATING_CHANGE = 15
"""Guardrail: the blended rating can never move more than this many points
away from the raw LLM rating, regardless of how extreme the pillar
composite or catalyst_horizon_modulation is. Mirrors the fleet's other
weight-optimizer guardrail patterns (e.g. weight_optimizer.py's
_MAX_SINGLE_CHANGE) — a fresh go-live gets a magnitude cap even though
this isn't itself an auto-tuned weight."""


def pillar_composite_score(pillar_assessment: QualitativePillarAssessment) -> float:
    """Uniform-weight mean across the 6 pillars plus catalyst horizon
    modulation, clamped to [0, 100]."""
    base = pillar_assessment.derive_legacy_qual_score()
    modulated = base + pillar_assessment.catalyst_horizon_modulation
    return float(max(0, min(100, modulated)))


def blend_rating(
    raw_llm_rating: int,
    pillar_assessment: QualitativePillarAssessment | None,
) -> int:
    """Blend ``PILLAR_BLEND_WEIGHT`` of the pillar composite into the raw
    LLM rating, capped at ``MAX_RATING_CHANGE`` points of movement.

    ``pillar_assessment`` is ``None`` for theses that predate config#2678's
    port (self-healed onto the ratings board from an old artifact) — in
    that case the rating passes through unchanged, exactly matching
    pre-2678 behavior.
    """
    if pillar_assessment is None:
        return raw_llm_rating
    composite = pillar_composite_score(pillar_assessment)
    blended = (1 - PILLAR_BLEND_WEIGHT) * raw_llm_rating + PILLAR_BLEND_WEIGHT * composite
    delta = max(-MAX_RATING_CHANGE, min(MAX_RATING_CHANGE, blended - raw_llm_rating))
    return round(max(0.0, min(100.0, raw_llm_rating + delta)))
