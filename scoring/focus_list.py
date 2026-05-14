"""
Focus list — regime-blended factor-driven per-team prescreen.

Builds a 15-20 name per-sector-team focus list from the within-sector
percentile-ranked factor composites produced by Phase 1c
(``scoring/factor_scoring.py``). The blend formula mirrors Phase 3's
``compute_factor_subscore`` (``scoring/composite.py``) so the focus-list
ordering and the downstream composite-score factor subscore are driven
by the same regime-conditional weights — tuning one tunes both.

Shadow substrate as of 2026-05-14:
  PR 1 of the scanner-placement plan (``alpha-engine-docs/private/scanner-260514.md``).
  This module is a pure helper — it does not wire into ``fetch_data_node``
  or change agent behavior. Wiring lives in subsequent PRs in the arc.

Sequencing note: closes ROADMAP L329 deferred PR 2 ("scanner momentum/
deep-value path strip + per-team Quant Analyst first-stage rank-and-narrow")
substrate side. Agent contract change is gated behind factor-substrate
Phase 2 (``@tool get_factor_profile``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

from scoring.composite import compute_factor_subscore

logger = logging.getLogger(__name__)


# Hard cap per the scanner-placement plan (§5.1). 20 is the institutional
# focus-list sweet spot — wide enough that the agent retains optionality,
# narrow enough that prompt-attention dilution doesn't degrade ranking
# quality. Per-team override accepted but never exceeds this ceiling.
FOCUS_LIST_HARD_CAP: int = 20

# Default size when no per-team override is supplied. 18 = midpoint of
# the 15-20 band specified in the plan doc.
FOCUS_LIST_DEFAULT_SIZE: int = 18

# Sectors with fewer than this many candidates pass through entirely
# (a degenerate top-N on a 3-ticker sector is just "all 3 of them").
FOCUS_LIST_MIN_SECTOR_SIZE: int = 3


# The four factor composites Phase 1c emits, indexed by the stance label
# we attach to each focus-list row. Mirrors ``_FACTOR_SCORE_KEYS`` in
# ``scoring/composite.py`` — keep these aligned.
_STANCE_BY_FACTOR: dict[str, str] = {
    "momentum_score": "momentum",
    "quality_score": "quality",
    "value_score": "value",
    "low_vol_score": "low_vol",
}


@dataclass
class FocusListEntry:
    """One row of the per-team focus list handed to the quant agent."""

    ticker: str
    sector: str
    team_id: str
    focus_score: float
    stance: str
    rank_in_sector: int
    rank_in_team: int
    quality_score: float | None = None
    momentum_score: float | None = None
    value_score: float | None = None
    low_vol_score: float | None = None
    factor_blend_breakdown: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _assign_stance(profile: dict) -> str:
    """Pick the dominant factor for stance routing.

    Returns the stance label of the composite with the highest score in
    the profile. Ties broken by ``_STANCE_BY_FACTOR`` iteration order
    (momentum → quality → value → low_vol) which corresponds to a mild
    pro-growth tie-break. Tickers with no factor scores → ``"unknown"``.
    """
    best_key: str | None = None
    best_val: float = float("-inf")
    for key in _STANCE_BY_FACTOR:
        score = profile.get(key)
        if score is None:
            continue
        if float(score) > best_val:
            best_val = float(score)
            best_key = key
    return _STANCE_BY_FACTOR.get(best_key, "unknown") if best_key else "unknown"


def compute_focus_scores(
    factor_profiles: dict[str, dict] | None,
    market_regime: str | None,
    regime_weights: dict[str, dict[str, float]] | None,
) -> dict[str, dict]:
    """Apply the Phase 3 regime-conditional blend to each ticker's factor profile.

    Args:
        factor_profiles: ``{ticker: {sector, quality_score, momentum_score, ...}}``
            as returned by ``read_factor_profiles_from_s3``. ``None`` or empty
            → empty result (graceful degrade — caller treats as "no focus list
            available, fall through to full sector slice").
        market_regime: ``"bull" | "bear" | "neutral"``.
        regime_weights: ``FACTOR_BLEND_REGIME_WEIGHTS`` config block —
            ``{regime: {factor_score_key: weight}}``. Signed weights.

    Returns:
        ``{ticker: {sector, focus_score, stance, quality_score, momentum_score,
        value_score, low_vol_score, factor_blend_breakdown}}`` for every
        ticker whose blend yielded a non-``None`` subscore. Tickers with no
        contributing factors are skipped.
    """
    if not factor_profiles:
        return {}

    out: dict[str, dict] = {}
    for ticker, profile in factor_profiles.items():
        subscore, details = compute_factor_subscore(
            profile, market_regime, regime_weights,
        )
        if subscore is None:
            continue
        out[ticker] = {
            "sector": profile.get("sector", "Unknown"),
            "focus_score": subscore,
            "stance": _assign_stance(profile),
            "quality_score": profile.get("quality_score"),
            "momentum_score": profile.get("momentum_score"),
            "value_score": profile.get("value_score"),
            "low_vol_score": profile.get("low_vol_score"),
            "factor_blend_breakdown": details.get("breakdown", {}),
        }
    return out


def build_focus_list(
    focus_scores: dict[str, dict],
    sector_team_map: dict[str, str],
    per_team_size: dict[str, int] | None = None,
    default_size: int = FOCUS_LIST_DEFAULT_SIZE,
    hard_cap: int = FOCUS_LIST_HARD_CAP,
) -> dict[str, list[FocusListEntry]]:
    """Group focus scores by team and take top-N per team.

    Args:
        focus_scores: output of ``compute_focus_scores``.
        sector_team_map: ``{gics_sector: team_id}`` (use ``team_config.SECTOR_TEAM_MAP``).
        per_team_size: optional ``{team_id: N}`` overrides. Each ``N`` is
            clamped to ``[0, hard_cap]``.
        default_size: per-team size when no override is supplied. Clamped
            to ``hard_cap``.
        hard_cap: absolute ceiling per team. Default ``FOCUS_LIST_HARD_CAP``.

    Returns:
        ``{team_id: [FocusListEntry, ...]}``. Entries ranked by
        ``focus_score`` descending. ``rank_in_sector`` is the ticker's
        rank among ALL same-sector candidates (not just same-team — sectors
        and teams are 1:1 in current config but kept distinct for future-
        proofing). ``rank_in_team`` is 1-indexed across the team's focus
        list. Teams with no qualifying tickers map to an empty list.
    """
    if not focus_scores:
        return {team_id: [] for team_id in set(sector_team_map.values())}

    per_team_size = per_team_size or {}
    effective_default = min(max(0, default_size), hard_cap)

    # Bucket by sector first — produces sector-level rank, then re-bucket
    # to team. With current config sector ↔ team is many-to-one (multiple
    # GICS sectors can map to one team, e.g. "Consumer Discretionary" +
    # "Consumer Staples" → "consumer"), so we keep both ranks.
    by_sector: dict[str, list[tuple[str, dict]]] = {}
    for ticker, entry in focus_scores.items():
        sector = entry["sector"]
        by_sector.setdefault(sector, []).append((ticker, entry))

    # Rank within each sector
    sector_ranks: dict[str, dict[str, int]] = {}
    for sector, items in by_sector.items():
        items.sort(key=lambda kv: kv[1]["focus_score"], reverse=True)
        sector_ranks[sector] = {t: i + 1 for i, (t, _) in enumerate(items)}

    # Bucket sector-ranked items into teams
    by_team: dict[str, list[tuple[str, dict]]] = {
        tid: [] for tid in set(sector_team_map.values())
    }
    for sector, items in by_sector.items():
        team_id = sector_team_map.get(sector)
        if team_id is None:
            continue
        by_team[team_id].extend(items)

    # For each team: rank by focus_score, take top-N, emit FocusListEntry
    result: dict[str, list[FocusListEntry]] = {}
    for team_id, items in by_team.items():
        items.sort(key=lambda kv: kv[1]["focus_score"], reverse=True)
        team_size = min(
            max(0, per_team_size.get(team_id, effective_default)),
            hard_cap,
        )

        # Min-size carve-out: if the team has fewer total candidates than
        # FOCUS_LIST_MIN_SECTOR_SIZE, pass through whatever exists (avoids
        # a degenerate "all 2 candidates dropped because rank > N" cut).
        if 0 < len(items) < FOCUS_LIST_MIN_SECTOR_SIZE:
            top_items = items
        else:
            top_items = items[:team_size]

        result[team_id] = [
            FocusListEntry(
                ticker=ticker,
                sector=entry["sector"],
                team_id=team_id,
                focus_score=entry["focus_score"],
                stance=entry["stance"],
                rank_in_sector=sector_ranks[entry["sector"]][ticker],
                rank_in_team=i + 1,
                quality_score=entry.get("quality_score"),
                momentum_score=entry.get("momentum_score"),
                value_score=entry.get("value_score"),
                low_vol_score=entry.get("low_vol_score"),
                factor_blend_breakdown=entry.get("factor_blend_breakdown", {}),
            )
            for i, (ticker, entry) in enumerate(top_items)
        ]

    return result


def summarize_focus_list(focus_list: dict[str, list[FocusListEntry]]) -> dict:
    """Compact summary for telemetry / logging.

    Returns ``{team_id: {n, top_3, stance_mix}}``. ``top_3`` is the
    three highest-ranked tickers (handy for log lines). ``stance_mix`` is
    a count of how many entries fall into each stance lane — useful for
    spotting regime-conditioning mismatches (e.g. a BULL run that surfaces
    mostly low_vol stances suggests the blend weights need re-tuning).
    """
    summary: dict[str, dict] = {}
    for team_id, entries in focus_list.items():
        stance_mix: dict[str, int] = {}
        for e in entries:
            stance_mix[e.stance] = stance_mix.get(e.stance, 0) + 1
        summary[team_id] = {
            "n": len(entries),
            "top_3": [e.ticker for e in entries[:3]],
            "stance_mix": stance_mix,
        }
    return summary
