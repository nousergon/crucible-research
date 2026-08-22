"""
Universe membership — the versioned, single source of truth for WHICH names are
in WHICH cut on a given cycle, published as one typed S3 artifact.

Motivation (alpha-engine-config-I4818 / I4819). Cut membership was, until this
module, implicit: reconstructable only by reading three unrelated artifacts with
three different shapes (``candidates/{date}/candidates.json::scanner_tickers``
for the scanner cut, ``scanner/universe/{date}/universe.json`` for attractiveness
ranks, ``population/{date}.json`` for the tracked population) and applying set
arithmetic in each consumer. That implicitness produced two live defects:

  1. The predictor resolved its daily universe from ``population/latest.json``,
     whose sole producer (the multi-agent research graph's ``archive_writer``)
     was retired with the research state — leaving the predictor scoring a
     FROZEN 2026-07-10 list for three weekly cycles with no error anywhere.
  2. Any membership-over-time question (churn, tenure, who-survives) required a
     consumer to re-derive the cuts itself, and the obvious source — the board's
     ``gate.quant_filter_pass`` — is unreliable (2026-07-02 reported 0 passing
     while ``candidates.json`` carried the normal 60).

This module makes membership a PRODUCT CONTRACT: one artifact, versioned, with
every cut named, sized, and carrying its own provenance, plus the full-universe
attractiveness rank table so any consumer can reconstruct a top-N at any N
without re-reading the board.

LOAD-BEARING — this is NOT best-effort observability. The predictor resolves its
daily scoring universe from this artifact, so its producer fails LOUD (see
``lambda/scanner_handler.py``): a missing membership artifact must be a red
Scanner run, never a silently-stale consumer. This is the deliberate difference
from the sibling universe-board write, which is dashboard-only and fail-soft.

Cuts emitted (each carries ``basis`` = how membership was decided):

  ``scanner_champion_60``
                             The scanner's live candidate cut,
                             ``candidates.json::scanner_tickers`` verbatim,
                             basis=``scanner_champion_rank``. This is the
                             sector teams' input set (via candidates.json —
                             ``graph.research_graph._resolve_agent_input_set``),
                             NOT the predictor universe, NOT the RAG corpus
                             scope, NOT Think Tank's coverage window. Group B
                             of SCANNER_CONTRACT.md §1.
                             Renamed from ``scanner_gate_baseline_60``
                             (alpha-engine-config-I7818): that name read as a
                             ``tech_score`` gate and has not been one since the
                             2026-07-22 config#1186 cutover
                             (alpha-engine-config-I7808) — the live ranking is
                             the scanner slot's champion arm. ``basis`` and
                             ``role`` always carried the truth; the key now
                             does too. ``scanner_gate_baseline_60`` is still
                             emitted as a deprecated alias for one window
                             (I7818 follow-up tracks its removal).
                             ``scanner_candidates`` — the alias this replaced
                             (alpha-engine-config-I7578) — is retired outright
                             as of this change and is no longer emitted.
  ``tech_score_top_60``      The DISPLACED INCUMBENT baseline, basis=
                             ``tech_score_rank``: top 60 by ``tech_score``
                             over every row the momentum path admitted
                             (``scan_path == "momentum"``), i.e. what the live
                             cut was before 2026-07-22. Feeds nothing. Emitted
                             so the cutover stays measurable and so "the
                             tech_score top 60" names something real —
                             ``scanner_top_20`` is scoped WITHIN the champion's
                             cut and does not answer that question
                             (alpha-engine-config-I7809).
  ``attractiveness_top_20``  top 20 by cross-sectional attractiveness rank.
                             **This is the cut the predictor resolves from**
                             (alpha-engine-config-I4983) — see
                             ``PREDICTOR_UNIVERSE_CUT`` below for why.
  ``scanner_top_20``         the INCUMBENT challenger arm — top 20 of the
                             scanner cut by ``tech_score``, i.e. what the
                             pre-I4983 rule would have picked at the champion's
                             width. basis=``tech_score_rank``. Emitted only when
                             the caller supplies the eval log's tech scores.
  ``attractiveness_top_25``  top 25 by the same rank.
  ``attractiveness_top_60``  top 60 by the same rank — count-matched to the
                             scanner cut so the two are directly comparable
                             (the comparison is the point: the ranking is
                             stable week-over-week while the gate cut churns
                             55-78%, i.e. the gate drives turnover, not the
                             score).

Deliberately NOT emitted: held positions. Research does not know the executor's
book, and inventing a stale copy here would recreate exactly the multi-writer
drift this artifact exists to end. The predictor unions its own holdings from
``trades/eod_pnl.csv::positions_snapshot`` (the executor's published surface) at
resolution time.

Output (versioned — consumers pin on ``schema_version``):
  ``s3://{bucket}/universe_membership/{run_date}/membership.json``  (pointer)
  ``s3://{bucket}/universe_membership/latest.json``                 (pointer)
  ``s3://{bucket}/universe_membership/{run_date}/runs/{stamp}.json`` (immutable)

The ``runs/`` copy exists because the first two are POINTERS and the Scanner
runs more than once per trading day (alpha-engine-config-I6785). The weekday
preopen SF invokes it in the morning; the postclose-chained weekly *exercise*
run (``pipeline_role=exercise``, config#6658) invokes it again after the close
and normalizes to the same ``trading_day``, so it rewrites the same key.

Measured 2026-08-07: ``predictor/predictions/2026-08-07.json`` was written at
17:16Z; ``universe_membership/2026-08-07/membership.json`` carries
``generated_at`` 2026-08-08T10:13Z. The surviving object POSTDATES the
predictions it supposedly fed by 17 hours, and four names stamped
``attractiveness_top_20`` in those predictions (FFIV, PFGC, SN, TREX) are absent
from it. Bucket versioning is off, so the morning cut was unrecoverable.

That is not only an audit gap. ``crucible-dashboard/loaders/universe_churn.py``
builds its whole churn/tenure/survivor series from the dated keys, so every
double-written cycle contributed the EVENING cut to a series presented as the
cut being traded. ``runs/`` is keyed on the artifact's own ``generated_at`` and
never overwritten: two writers on one trading day produce two objects, and the
question "what did the predictor actually read this morning" stays answerable.

Schema::

  {
    "schema_version": 1,
    "producer": "crucible-research/scoring/universe_membership.py",
    "run_date": "YYYY-MM-DD",
    "generated_at": "ISO-8601 UTC",
    "universe_count": int,             # SCANNED names with a rankable attractiveness
                                       # score. Equals the board's universe_count
                                       # minus population_reconciliation.unrankable.
    "predictor_universe_cut": "attractiveness_top_20",  # names the cut the predictor resolves from
    "cuts": {
      "<cut_name>": {
        "basis": "scanner_gate" | "attractiveness_rank",
        "size": int,
        "tickers": ["AAPL", ...],      # SORTED — set semantics, not rank order
        "source": "<provenance string>"
      },
      ...
    },
    "ranks": {                          # full universe, rank 1 = most attractive
      "AAPL": {"attractiveness_rank": int, "attractiveness_score": float},
      ...
    },
    "tech_score_ranks": {               # FULL momentum-path universe, rank 1 = highest
      "AAPL": {"tech_score_rank": int, "tech_score": float},
      ...                               # Absent when no eval log was supplied.
    },
    "scanner_ranks": {                  # SCANNER CUT only, rank 1 = highest
      "AAPL": {"tech_score_rank": int, "tech_score": float},   # tech_score.
      ...                               # Absent when no eval log was supplied.
    },
    "rank_tables": {                    # basis -> which field holds its FULL table
      "attractiveness_rank": {"field": "ranks", "rank_key": ..., "score_key": ...,
                              "size": int, "population": int,
                              "serves_rank_ceiling": bool, "eligibility": str},
      "tech_score_rank":     {"field": "tech_score_ranks", ...},
    },
    "population_reconciliation": {      # this artifact vs the universe board
      "source": "candidates/{date}/candidates.json::scanner_eval_log::ticker",
      "scanned_universe_size": int | null,   # null on a historical backfill
      "ranked": int,
      "unrankable": ["VMRK", ...],           # scanned, no rankable score
      "ranked_outside_scanned_universe": [], # ALWAYS empty — asserted
      "unrankable_reason": str
    },
    "turnover": {                       # null when no prior artifact was readable
      "prior_run_date": "YYYY-MM-DD",
      "prior_generated_at": "ISO-8601 UTC",
      "per_cut": {
        "<cut_name>": {"size": int, "retained": int, "added": int,
                       "dropped": int, "retention_pct": float}
      }
    }
  }

``ranks`` is the reconstruction substrate: a consumer wanting top-N at an N this
module does not emit slices it directly, with no board read and no re-derivation
of the ranking method. Since alpha-engine-config-I7843 there is one such table
per PROMOTABLE BASIS, indexed by ``rank_tables``: a consumer resolves the rank
ceiling of whichever arm is CHAMPION, so a basis served by only a 60-name
within-cut table could not answer a ceiling of 150 and the arm was unpromotable.

``schema_version`` stays 1 through both changes (2026-08-20). Everything added
is additive — ``ranks``, ``scanner_ranks``, ``cuts`` and every existing key keep
their name, type and meaning, so a consumer reading
``ranks[t]["attractiveness_rank"]`` is untouched. The RANKED POPULATION narrows
by the names the scanner never evaluated (alpha-engine-config-I7844), which is a
defect fix inside the field's declared meaning rather than a new contract, and
it is made visible by ``population_reconciliation`` rather than by a version
bump that would tell every consumer to change code none of them has to change.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, date, datetime
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
PRODUCER = "crucible-research/scoring/universe_membership.py"

# The cut the predictor resolves its daily scoring universe from. Named in the
# artifact (``predictor_universe_cut``) rather than hardcoded on the consumer
# side, so changing which cut drives inference is a producer-side, versioned,
# reviewable decision — not a silent constant edit in another repo.
#
# alpha-engine-config-I4983 (Brian, 2026-07-28): moved from ``scanner_candidates``
# to ``attractiveness_top_20``. Three reasons, in order of weight:
#
#  1. The gate cut is a MOMENTUM ranking, not an alpha ranking. It is
#     ``tech_score`` top-N (RSI / MACD / MA50 / MA200 / 20d momentum, equally
#     weighted) plus the 10 most-oversold-by-RSI — no fundamentals at all. On
#     the 2026-07-24 cycle its median name ranked 598 of 897 on attractiveness,
#     with 39 of 60 in the BOTTOM HALF of the universe. Attractiveness is the
#     6-pillar composite (quality/value/momentum/growth/stewardship/
#     defensiveness) in which momentum is one input of six.
#  2. §43: a hard gate must not double as alpha selection. Measured cost of the
#     conflation: 21d scanner recall 3.9% (150 TP vs 3,732 FN), filter lift
#     −0.68% — the survivor set both misses ~96% of winners and underperforms
#     the universe mean.
#  3. The gate cut has no persistent core — 42% week-over-week retention and
#     ZERO names present across all 9 measured cycles, against 76%/18 for a
#     count-matched rank cut (EXPERIMENTS.md 2026-07-27). It is a weekly
#     re-draw, not a watchlist.
#
# The attractiveness re-ranking this restores was built under config-I1400/I1407
# but spliced into the multi-agent graph, so it was orphaned when that layer was
# retired as champion (config#1580) — see alpha-engine-config-I4980.
PREDICTOR_UNIVERSE_CUT = "attractiveness_top_20"

# ── The gate cut's name (alpha-engine-config-I7578, Brian ruling 2026-08-17) ──
# Two 60-wide cuts exist and only one is the funnel — and until the I7578
# rename, the one that is NOT the funnel was the one whose name said "scanner".
# It was read backwards three times, each caught by hand rather than by a
# check. Measured 2026-08-17: the two 60s overlap on 12 of 60, and the champion
# 20 overlaps the gate cut on 3 of 20. They are near-disjoint sets and only one
# advances.
#
# The funnel INVARIANT was already enforced (I6630, ``assert_cut_invariants``).
# What was unguarded is the NAMING: nothing stopped a reader, an agent, or a new
# consumer from resolving "the scanner's top 60" to the gate cut and scoping to
# a set that feeds nothing. The word "baseline" is load-bearing — reading this
# name as the funnel now requires ignoring it.
#
# alpha-engine-config-I7818: ``scanner_gate_baseline_60`` itself read as a
# ``tech_score`` gate, which it stopped being at the 2026-07-22 config#1186
# cutover — see the module docstring's cut table. ``scanner_champion_60`` is
# the primary name; it states the actual basis (the scanner slot's champion
# arm) rather than a mechanism the key hasn't described in a month.
CHAMPION_CUT = "scanner_champion_60"
"""Primary name for the scanner's live candidate cut (alpha-engine-config-I7818).
States what the cut actually is — the champion arm's ranking — rather than
"gate" or "baseline", neither of which has been true since config#1186."""

GATE_BASELINE_CUT = "scanner_gate_baseline_60"
"""Deprecated alias for :data:`CHAMPION_CUT`, emitted for one window
(alpha-engine-config-I7818 follow-up tracks its removal). Same rationale as
the alias mechanism below: a consumer pinned on this name keeps working
through the window rather than reading a missing key as an empty cut."""

GATE_LEGACY_CUT = "scanner_candidates"
"""RETIRED as an emitted alias (alpha-engine-config-I7818) — the I7578
deprecation window is closed and this key is no longer written to new
artifacts; do not resurrect it as a third live name for this cut. Kept as a
named constant, not a literal, ONLY because artifacts written before I7578
carry the arm under this spelling and ``scoring/leaderboard_producers.py``
reads across that boundary for historical-series continuity
(``_load_cut_specs``'s ``aliases`` table). Any new code resolving the
LIVE cut must use :data:`CHAMPION_CUT`, never this one."""

TECH_SCORE_CUT_PREFIX = "tech_score_top_"
"""The displaced-incumbent baseline cut (alpha-engine-config-I7809).

Distinct from ``scanner_top_{N}``, which ranks by ``tech_score`` WITHIN the
champion's 60. This one ranks over every momentum-path-eligible row in the
scanned universe, so it answers "what is the universe's tech_score top 60?" —
the question Brian asked on 2026-08-20 and which no emitted cut answered.
"""

# Attractiveness-rank cuts emitted every cycle. 60 is count-matched to the
# scanner's ``momentum_top_n`` so scanner-cut-vs-rank-cut is an apples-to-apples
# comparison; 25 matches the size of the retired tracked population so the
# historical series stays interpretable across the producer change; 20 is the
# live champion width, count-matched to Think Tank's ``CHALLENGER_TOP_N`` so all
# three champion/challenger arms are directly comparable at equal size (an arm's
# win must not be confounded between selection rule and breadth — the
# count-matched framing is what made the churn finding legible at all).
ATTRACTIVENESS_FEED_TOP_N = 60
"""Width of the decision set fed forward to evidence ingestion and the
challenger arms — ``rag-corpus-policy.md`` §2.1 names this constant.

The fleet SSoT is ``nousergon_lib.decision_set.ATTRACTIVENESS_FEED_TOP_N``.
This module keeps a local definition rather than importing it because this
repo's ``nousergon-lib`` pin is v0.124.3 and adopting the lib constant means a
34-version jump that has nothing to do with this invariant; unification is
tracked and lands with the next pin bump. The value is asserted against the
consumer's in the cross-repo contract test rather than trusted to stay equal
by inspection.
"""

_RANK_CUTS = (20, 25, ATTRACTIVENESS_FEED_TOP_N)

_FEEDS_BY_RANK_CUT: dict[int, list[str]] = {
    20: ["predictor_universe"],
    25: [],
    ATTRACTIVENESS_FEED_TOP_N: ["sector_teams", "rag_corpus_scope", "thinktank_window"],
}
"""Which consumer reads which attractiveness cut, recorded IN the artifact.

``sector_teams`` on the 60 is Brian's ruling 2026-08-20
(``alpha-engine-config-I7823``) — and it is conditional on this cut holding the
champion pointer, which :func:`live_cut_champion` resolves at read time. The
list states the ARRANGEMENT; the pointer states which arm is serving it today.
"""

FEED_CUT_NAME = f"attractiveness_top_{ATTRACTIVENESS_FEED_TOP_N}"
"""The cut downstream evidence ingestion scopes to. The champion cut must be
its head — see the funnel invariant in :func:`build_universe_membership`."""

# Width of the incumbent (``tech_score``) challenger arm — held equal to the
# champion width so champion-vs-incumbent is a selection-rule comparison with
# breadth held constant.
_INCUMBENT_CUT_N = 20

# ── Momentum-horizon challenger cut (alpha-engine-config-I7538) ─────────────
# The champion attractiveness composite's momentum pillar puts 40% of its
# weight at horizons of one month or less — the short-term-REVERSAL window —
# and carries no 12-1 skip-month term. This arm re-ranks the SAME universe
# with `scoring.factor_scoring.challenger_composite_defs()`, which differs
# from the champion in `momentum_score` ONLY.
#
# Emitted at BOTH champion widths (20 and 60) so champion-vs-challenger is a
# selection-rule comparison with breadth held constant — an arm's win must
# never be confounded between the rule and its width
# (champion-challenger-policy.md §4).
CHALLENGER_CUT_PREFIX = "attractiveness_mom121_top_"
_CHALLENGER_CUT_NS = (20, ATTRACTIVENESS_FEED_TOP_N)

# ── Challenger cut registry (alpha-engine-config-I7574) ─────────────────────
# Each arm varies ONE thing about how the ~903-name universe is ranked, so the
# leaderboard can attribute a win to that thing. Two variables are in play and
# they are deliberately NOT combined into one arm:
#
#   mom121   momentum stays 1/6 of the composite; its COMPONENTS move to the
#            12-1 skip-month horizon. Isolates HORIZON.
#   momzero  components stay the champion's; the momentum pillar's WEIGHT goes
#            to 0 and the remaining five split 1.0 evenly. Isolates EXPOSURE.
#
# An arm that changed both could not tell you which one earned the result —
# and the two have genuinely different implications. If mom121 wins, momentum
# belongs in the score and was simply measured over the wrong window. If
# momzero wins, momentum does not belong in a one-year attractiveness score at
# all, whatever window it is measured over. Those are different beliefs about
# the strategy, not two strengths of one belief.
#
# momzero is cheap in a way mom121 is not: it needs no second factor-composite
# computation, because ``attractiveness_from_factor_profiles`` already accepts
# a ``pillar_weights`` vector. Same profiles, different weights.
PILLAR_ORDER_FOR_WEIGHTS = (
    "quality",
    "value",
    "momentum",
    "growth",
    "stewardship",
    "defensiveness",
)

MOMZERO_CUT_PREFIX = "attractiveness_momzero_top_"

MOMZERO_PILLAR_WEIGHTS: dict[str, float] = {
    "quality": 0.2,
    "value": 0.2,
    "momentum": 0.0,
    "growth": 0.2,
    "stewardship": 0.2,
    "defensiveness": 0.2,
}
"""Zero, not merely reduced. A partial weight muddles the read — it produces a
ranking that is neither the champion's nor a clean no-momentum counterfactual,
so a small win or loss cannot be attributed. Zero answers the question being
asked ("does momentum belong here at all?") and, if it wins, the follow-up
experiment is a tuned weight somewhere in (0, 1/6) — which is only worth
running once the sign is known."""

_DEFAULT_BUCKET = "alpha-engine-research"


class UniverseMembershipError(RuntimeError):
    """Raised when the membership artifact cannot be built from real inputs.

    Deliberately a hard error (feedback_no_silent_fails): this artifact is
    load-bearing for the predictor's daily universe, so an empty or partial
    membership is a red run, never a degraded write.
    """


# ── Cut refresh cadence (alpha-engine-config-I6666) ──────────────────────────
# How often the CUT is re-derived, as opposed to how often this artifact is
# WRITTEN. The two were the same thing until now: every invocation re-cut
# everything, so when config-I6494-A put the Scanner on the weekday preopen SF
# the predictor's universe silently began turning over daily — 32 distinct
# names held a top-20 slot across the six sessions 2026-07-29 → 2026-08-07, and
# only 12 of the original 20 survived, against a 21-day prediction horizon.
#
# Brian's ruling 2026-08-08: stay daily for now, but make the switch to weekly
# a one-line flip rather than a producer migration. Hence a declared setting
# with an env override, so the Scanner Lambda can move without a deploy.
#
# The artifact is written EVERY run under both settings (see
# ``resolve_cut_refresh``) — a carry-forward still writes. That is deliberate:
# it keeps ``universe_membership_latest`` honest at cadence ``weekday_sf``
# (alpha-engine-config-I6651), so a missed Scanner run stays visible as a
# missing artifact instead of being indistinguishable from a held cut.
CADENCE_DAILY = "daily"
CADENCE_WEEKLY = "weekly"
CUT_REFRESH_CADENCES = frozenset({CADENCE_DAILY, CADENCE_WEEKLY})
DEFAULT_CUT_REFRESH_CADENCE = CADENCE_DAILY


def cut_refresh_cadence() -> str:
    """The configured cut-refresh cadence.

    ``SCANNER_CUT_REFRESH_CADENCE`` overrides the default so the Scanner Lambda
    flips without a deploy. An unrecognised value RAISES rather than falling
    back: a typo silently selecting daily would be indistinguishable from the
    setting working, and the whole point of this knob is that its position is
    knowable.
    """
    value = (os.environ.get("SCANNER_CUT_REFRESH_CADENCE") or "").strip().lower()
    if not value:
        return DEFAULT_CUT_REFRESH_CADENCE
    if value not in CUT_REFRESH_CADENCES:
        raise UniverseMembershipError(
            f"SCANNER_CUT_REFRESH_CADENCE={value!r} is not a recognised cadence — "
            f"expected one of {sorted(CUT_REFRESH_CADENCES)} "
            f"(alpha-engine-config-I6666)"
        )
    return value


def _iso_week(date_str: str) -> tuple[int, int]:
    """``(iso_year, iso_week)`` for a ``YYYY-MM-DD`` string."""
    try:
        y, w, _ = date.fromisoformat(str(date_str)).isocalendar()
    except (TypeError, ValueError) as exc:
        raise UniverseMembershipError(f"cannot resolve the ISO week of {date_str!r}: {exc}") from exc
    return (y, w)


def should_recut(run_date: str, prior: dict | None, cadence: str | None = None) -> bool:
    """Whether this run re-derives the cuts, or carries the prior ones forward.

    ``daily`` always re-cuts. ``weekly`` re-cuts on the first run of each ISO
    week, and whenever there is no usable prior cut to carry.

    ISO week rather than a named weekday deliberately: a holiday-shortened week
    still gets exactly one re-cut, with no trading-calendar special-casing here
    and no way for a skipped Saturday to strand the cut for a fortnight.
    """
    cadence = cadence or cut_refresh_cadence()
    if cadence == CADENCE_DAILY:
        return True
    effective = (prior or {}).get("cut_effective_date")
    if not effective or not (prior or {}).get("cuts"):
        # No prior cut to carry — a first run, or an artifact predating this
        # field. Re-cut rather than invent one.
        return True
    return _iso_week(run_date) > _iso_week(effective)


# ── The champion cut pointer (alpha-engine-config-I7823) ────────────────────
# Brian's ruling 2026-08-20: `attractiveness_top_60` and `tech_score_top_60`
# run as a count-matched champion/challenger PAIR, the better performer is
# promoted weekly, and the SECTOR TEAMS read whichever is champion.
#
# The pointer is an S3 object rather than an env var or a constant, because a
# weekly promotion engine has to be able to move it without a deploy, and
# because "which arm was live on date D" then has an artifact instead of being
# reconstructed from a deploy log. Same reasoning as
# `config/factor_attractiveness_weights.json` (I7580).
CUT_CHAMPION_POINTER_KEY = "config/scanner_cut_champion.json"

PROMOTABLE_CUTS: tuple[str, ...] = (
    "attractiveness_top_60",
)
"""The arms eligible to hold the feed. Count-matched at 60 by construction.

A closed set, deliberately: the pointer is writable by an automated promotion
engine, and an unvalidated pointer is an arbitrary-cut-selection primitive one
bad write away from feeding the sector teams something nobody chose.

**Brian ruling 2026-08-21 (alpha-engine-config-I8060): `tech_score_top_60` is
OBSERVE-ONLY until it has weeks of measured performance.** It was made
promotable on 2026-08-20 (I7823) and first emitted the same day, so it has
never had a scored cohort; arming an automatic pointer write before any
evidence exists puts the whole gate on a floor nobody has watched hold. It
stays fully scored — see :data:`OBSERVE_ONLY_CUTS` — and returns here by
amending this tuple, which is the only edit required.
"""

OBSERVE_ONLY_CUTS: tuple[str, ...] = (
    f"{TECH_SCORE_CUT_PREFIX}{ATTRACTIVENESS_FEED_TOP_N}",
    f"{MOMZERO_CUT_PREFIX}{ATTRACTIVENESS_FEED_TOP_N}",
    f"{CHALLENGER_CUT_PREFIX}{ATTRACTIVENESS_FEED_TOP_N}",
)
"""Arms that are SCORED every cycle but cannot hold the feed.

champion-challenger-policy.md §3 makes measurement unconditional: promotion
changes which arm is consumed and changes nothing about what is measured. So
non-promotability is declared HERE, as a state, rather than left to be inferred
from absence from :data:`PROMOTABLE_CUTS` — ARCHITECTURE §140, a disposition
that is only an inference is not a state. `_load_cut_specs` reads this tuple, so
an arm added here is scored by construction and cannot become the registered-
but-unscored rumour §3 warns about.

Count-matched at :data:`ATTRACTIVENESS_FEED_TOP_N` with the promotable arm, so
promoting one later needs no re-baselining of its history.
"""

SLOT_ARMS: tuple[str, ...] = PROMOTABLE_CUTS + OBSERVE_ONLY_CUTS
"""Every arm of the universe-cut slot, promotable or not. The scoring surface
resolves its arm list from this so the board and the registry cannot drift."""

DEFAULT_CUT_CHAMPION = "attractiveness_top_60"
"""The standing champion, per Brian's ruling 2026-08-20. Serves whenever the
pointer is absent or unreadable — the 6-pillar attractiveness composite, which
is what fed the predictor and the evidence layers already.

The pillar WEIGHTS this cut ranks by are not a property of this constant: they
are read per-run from ``config/factor_attractiveness_weights.json`` via
``universe_board._load_pillar_weights``. This docstring said "momentum-free"
between 2026-08-17 and 2026-08-21, when that file carried momentum 0.0; Brian's
ruling 2026-08-21 (alpha-engine-config-I7988) restored equal weight and moved
the momentum-zero composite to the ``attractiveness_momzero_top_60`` shadow arm.
Naming a weighting here at all was the mistake — the weighting is configuration
and this is a cut name, so the two go stale on different clocks."""


def live_cut_champion(*, bucket: str | None = None, s3_client: Any = None) -> str:
    """The cut the sector teams read this cycle.

    Absence returns :data:`DEFAULT_CUT_CHAMPION` — a first run, or a fleet that
    has never promoted, is not an error. A pointer that EXISTS but names a cut
    outside :data:`PROMOTABLE_CUTS` is a different thing entirely and RAISES:
    something wrote a value nobody validated, and quietly serving the default
    instead would mean the promotion engine believes one arm is live while the
    funnel serves another — the exact class of drift the 2026-07-22 cutover
    produced (alpha-engine-config-I7808).
    """
    s3 = _client(s3_client)
    b = _bucket(bucket)
    try:
        body = s3.get_object(Bucket=b, Key=CUT_CHAMPION_POINTER_KEY)["Body"].read()
    except Exception as exc:  # noqa: BLE001 — absence is a legitimate state
        if "NoSuchKey" in str(exc) or "404" in str(exc):
            return DEFAULT_CUT_CHAMPION
        raise
    doc = json.loads(body)
    champion = doc.get("champion")
    if champion not in PROMOTABLE_CUTS:
        raise UniverseMembershipError(
            f"{CUT_CHAMPION_POINTER_KEY} names champion {champion!r}, which is not "
            f"one of {list(PROMOTABLE_CUTS)}. Refusing to resolve a feed from an "
            "unvalidated pointer."
        )
    return champion


def resolve_feed_cut(*, bucket: str | None = None, s3_client: Any = None) -> tuple[list[str], dict]:
    """``(tickers, provenance)`` for the sector teams' input set.

    Reads the LATEST membership artifact, not a dated one. Under the weekly
    cadence the scanner does not run every day, so a consumer keyed on
    ``{today}`` would find nothing and fail loud on four mornings out of five —
    the failure mode that makes the weekly cadence unshippable on its own
    (alpha-engine-config-I7823, step 1 of the merge order).

    Raises rather than degrading: the sector teams' input set is load-bearing,
    and an empty one is indistinguishable from "the scanner selected nobody".
    """
    champion = live_cut_champion(bucket=bucket, s3_client=s3_client)
    membership = read_latest_membership(bucket=bucket, s3_client=s3_client)
    if not membership:
        raise UniverseMembershipError(
            "no universe_membership/latest.json — the sector-team feed resolves "
            "from the membership artifact and cannot fall back to the raw universe"
        )
    cut = (membership.get("cuts") or {}).get(champion) or {}
    tickers = list(cut.get("tickers") or [])
    if not tickers:
        raise UniverseMembershipError(
            f"universe_membership/latest.json (run_date={membership.get('run_date')}) "
            f"carries no tickers under the champion cut {champion!r} — refusing to "
            "feed the sector teams an empty set"
        )
    return tickers, {
        "cut": champion,
        "run_date": membership.get("run_date"),
        "cut_effective_date": membership.get("cut_effective_date"),
        "basis": cut.get("basis"),
        "size": len(tickers),
    }


# ── The funnel as a READ contract (alpha-engine-config-I7842) ───────────────
# ``build_universe_membership`` already records, in the artifact, which cut
# each downstream consumer advances to. Until this section existed, only the
# sector-team feed had a reader (:func:`resolve_feed_cut`); every other named
# consumer re-derived its own set. Think Tank was the live instance: it sorted
# the universe board by ``attractiveness_score`` in Python, so it agreed with
# the funnel only for as long as the champion happened to rank on
# attractiveness, and would have kept covering the attractiveness 60 through a
# promotion to ``tech_score_top_60`` with nothing raising
# (alpha-engine-config-I7808, the same shape).
#
# The declaration is read, never re-stated on the consumer side: a consumer
# that hardcodes ``attractiveness_top_60`` is the duplicated-truth defect one
# level down from the one this replaces.

FUNNEL_CONSUMER_THINKTANK = "thinktank_coverage_window"
"""Key under ``funnel.advances_to`` naming Think Tank's coverage window."""

FUNNEL_CONSUMER_RAG = "rag_corpus_scope"
FUNNEL_CONSUMER_PREDICTOR = "predictor_universe"

TECH_SCORE_RANKS_FIELD = "tech_score_ranks"
"""Top-level field holding the FULL-UNIVERSE ``tech_score`` rank table.

Separate from ``scanner_ranks``, which is retained unchanged: that one ranks
names *within* the champion's own 60 and is what the incumbent-arm comparison
reads. Two tables because they answer two questions — "where does this name sit
in the tech_score universe?" and "how did the incumbent order the cut it was
actually given?" — and collapsing them would silently answer one with the other.
"""

_RANK_TABLE_BY_BASIS: dict[str, tuple[str, str, str]] = {
    "attractiveness_rank": ("ranks", "attractiveness_rank", "attractiveness_score"),
    "tech_score_rank": (TECH_SCORE_RANKS_FIELD, "tech_score_rank", "tech_score"),
}
"""``basis`` -> (membership field holding a FULL-UNIVERSE rank table, rank key, score key).

The DEFAULT index, used for artifacts written before ``rank_tables`` existed.
A current artifact carries its own ``rank_tables`` index and
:func:`rank_table_for_cut` prefers it: which field holds which basis is a fact
about the artifact in hand, and reading it out of a constant in the producer's
source is the same duplicated-truth defect the funnel declaration replaced.

``tech_score_rank`` was deliberately ABSENT here until
alpha-engine-config-I7843: the only tech table emitted was ``scanner_ranks``,
60 names ranked within the champion's cut, which cannot answer "who is rank 150
in this basis" — and answering it with the attractiveness table instead would
mean the champion pointer names one arm while the consumer ranks by another
(alpha-engine-config-I7808). The producer now emits the full table, so the
refusal is no longer the right answer for this basis; :func:`rank_table_for_cut`
still refuses for any basis that has none.
"""

MIN_PROMOTABLE_RANK_COVERAGE = 200
"""Widest rank position a consumer of a PROMOTABLE cut may ask about.

Think Tank's ``exit_rank`` (``config/thinktank.sample.yaml``) is the binding
consumer today. Asserted at the producer against every promotable cut's basis
so a basis that cannot serve the funnel's own consumers is a red Scanner run
rather than a promotion that fails on the morning it is made.
"""


def declared_cut_for(membership: dict, consumer: str) -> str:
    """The cut name ``membership`` declares for ``consumer``, or raise.

    Absence is a hard error, not a default. A defaulted or empty window is
    indistinguishable from "the scanner selected nobody" — the reasoning
    :func:`resolve_feed_cut` already gives for the sector-team feed, and it
    applies identically to every other consumer of this artifact.
    """
    advances = ((membership or {}).get("funnel") or {}).get("advances_to") or {}
    if consumer not in advances:
        raise UniverseMembershipError(
            f"universe_membership (run_date={(membership or {}).get('run_date')}) "
            f"does not declare funnel.advances_to.{consumer} — the consumer's set "
            f"is resolved FROM this declaration and has no fallback. Declared "
            f"consumers: {sorted(advances)}"
        )
    named = advances[consumer]
    cuts = (membership or {}).get("cuts") or {}
    if not named or named not in cuts:
        raise UniverseMembershipError(
            f"universe_membership (run_date={(membership or {}).get('run_date')}): "
            f"funnel.advances_to.{consumer}={named!r} names a cut that is not in "
            f"``cuts`` ({sorted(cuts)}). The declaration and the cuts are written "
            f"by the same producer in the same pass, so a dangling name is a "
            f"producer defect, never a consumer's problem to route around."
        )
    return str(named)


def serving_cut_for(
    membership: dict,
    consumer: str,
    *,
    bucket: str | None = None,
    s3_client: Any = None,
    champion: str | None = None,
) -> tuple[str, str]:
    """``(serving_cut_name, declared_cut_name)`` for ``consumer``.

    The declaration states the ARRANGEMENT — which slot in the funnel this
    consumer reads. The champion pointer states which ARM is serving that slot
    today (the distinction ``_FEEDS_BY_RANK_CUT``'s docstring already draws).
    So when the declared cut is one of :data:`PROMOTABLE_CUTS`, the serving cut
    is :func:`live_cut_champion`'s answer, exactly as :func:`resolve_feed_cut`
    resolves the sector-team feed. Brian's ruling 2026-08-20
    (alpha-engine-config-I7823): the two count-matched 60s are promoted weekly
    and the consumers of that slot read whichever is champion.

    A declared cut OUTSIDE the promotable set is served verbatim — it is not a
    contested slot, and routing it through the pointer would let a promotion in
    one slot silently move another consumer's window.
    """
    declared = declared_cut_for(membership, consumer)
    if declared not in PROMOTABLE_CUTS:
        return declared, declared
    serving = champion or live_cut_champion(bucket=bucket, s3_client=s3_client)
    cuts = membership.get("cuts") or {}
    if serving not in cuts:
        raise UniverseMembershipError(
            f"universe_membership (run_date={membership.get('run_date')}): the live "
            f"champion {serving!r} holds the {declared!r} feed slot that "
            f"funnel.advances_to.{consumer} names, but no such cut is emitted "
            f"({sorted(cuts)}). Refusing to serve {consumer} the losing arm's "
            f"membership — that is alpha-engine-config-I7808 repeating."
        )
    return serving, declared


_RANK_TABLE_ELIGIBILITY: dict[str, str] = {
    "attractiveness_rank": (
        "every name in the SCANNED universe with a rankable attractiveness "
        "score (factors/profiles/{run_date}/by_ticker.json, cross-sectional "
        "percentile over that same population)"
    ),
    "tech_score_rank": (
        "every scanned row the MOMENTUM PATH admitted (scan_path == 'momentum': "
        "liquidity floor, price floor, tech_score_min, MA200 floor, momentum "
        "volatility ceiling). Legitimately narrower than the universe — ranking "
        "a name the incumbent rule rejected would invent an ordering it never "
        "expressed."
    ),
}
"""Human-readable eligibility statement per basis, emitted with each table.

A consumer that finds its ceiling unanswerable in some basis is entitled to
know WHY the table is narrow — a declared gate, or a broken read — without
opening this module (alpha-engine-config-I7843)."""


def _rank_tables_block(membership: dict, population: int) -> dict[str, dict]:
    """The self-describing ``rank_tables`` index for an assembled artifact.

    One entry per basis that actually has a full-universe table in this
    artifact. ``size`` and ``serves_rank_ceiling`` are recorded so a consumer
    can check whether a basis reaches its ceiling rather than discover it by
    being refused (alpha-engine-config-I7843).
    """
    block: dict[str, dict] = {}
    for basis, (field, rank_key, score_key) in _RANK_TABLE_BY_BASIS.items():
        table = membership.get(field) or {}
        if not table:
            continue
        block[basis] = {
            "field": field,
            "rank_key": rank_key,
            "score_key": score_key,
            "size": len(table),
            "population": population,
            "serves_rank_ceiling": len(table) >= min(MIN_PROMOTABLE_RANK_COVERAGE, population),
            "eligibility": _RANK_TABLE_ELIGIBILITY.get(basis, ""),
        }
    return block


def _rank_table_index(membership: dict) -> dict[str, tuple[str, str, str]]:
    """``basis -> (field, rank_key, score_key)`` for the artifact in hand.

    Read from the artifact's own ``rank_tables`` declaration when it carries
    one, so a producer that adds a basis needs no matching edit in every
    consumer; falls back to :data:`_RANK_TABLE_BY_BASIS` for artifacts written
    before ``rank_tables`` existed (alpha-engine-config-I7843).

    A malformed declaration is IGNORED per-entry rather than raised on: the
    fallback is the honest older contract, and the caller's own coverage check
    is what decides whether the resolved table can answer its question.
    """
    declared = (membership or {}).get("rank_tables")
    if not isinstance(declared, dict) or not declared:
        return dict(_RANK_TABLE_BY_BASIS)
    index: dict[str, tuple[str, str, str]] = {}
    for basis, meta in declared.items():
        if not isinstance(meta, dict):
            continue
        field, rank_key, score_key = (meta.get("field"), meta.get("rank_key"), meta.get("score_key"))
        if field and rank_key and score_key:
            index[str(basis)] = (str(field), str(rank_key), str(score_key))
    return index or dict(_RANK_TABLE_BY_BASIS)


def rank_table_for_cut(
    membership: dict,
    cut_name: str,
    *,
    minimum_coverage: int | None = None,
) -> tuple[dict[str, int], str]:
    """``({ticker: rank}, basis)`` in the basis ``cut_name`` is ranked by.

    ``minimum_coverage`` is the widest rank position the caller will ask about
    (Think Tank's ``exit_rank``, say). A table shorter than that cannot answer
    the caller's question, and this raises rather than returning a table that
    silently makes every unranked name look absent — absence and "ranked worse
    than N" must never render identically (champion-challenger-policy §7.2).

    Raises for a basis with no full-universe table rather than substituting the
    attractiveness one: see :data:`_RANK_TABLE_BY_BASIS` and
    alpha-engine-config-I7843.
    """
    cut = (membership.get("cuts") or {}).get(cut_name) or {}
    basis = cut.get("basis")
    entry = _rank_table_index(membership).get(str(basis))
    if entry is None:
        raise UniverseMembershipError(
            f"universe_membership (run_date={membership.get('run_date')}): cut "
            f"{cut_name!r} is ranked by {basis!r}, for which this artifact emits "
            f"no full-universe rank table (it emits one only for "
            f"{sorted(_rank_table_index(membership))}). A consumer's rank ceiling cannot "
            f"be resolved in this basis, and resolving it in the attractiveness "
            f"basis instead would rank by an arm that is not the champion — "
            f"alpha-engine-config-I7808. Producer fix: "
            f"alpha-engine-config-I7843."
        )
    field, rank_key, _score_key = entry
    table = membership.get(field) or {}
    if not table:
        raise UniverseMembershipError(
            f"universe_membership (run_date={membership.get('run_date')}): "
            f"{field!r} is empty — cut {cut_name!r} declares basis {basis!r} but "
            f"the table that basis ranks in carries no names. Refusing rather "
            f"than resolving the ceiling out of another basis' table, which "
            f"would rank by an arm the champion pointer does not name "
            f"(alpha-engine-config-I7843)."
        )
    ranks = {str(t): int(v[rank_key]) for t, v in table.items() if rank_key in (v or {})}
    if minimum_coverage is not None:
        # Compared against the POPULATION, not against the ceiling alone: a
        # universe smaller than the ceiling is a legitimate state (the ceiling
        # simply does not bind), and refusing it would fail every small-universe
        # run. What must never pass is a table that ranks a fraction of a
        # universe it claims to rank — the tech basis today ranks 60 of 906, so
        # every name past 60 would be indistinguishable from a name the scanner
        # did not rank at all (alpha-engine-config-I7843).
        population = int(
            membership.get("universe_count") or ((membership.get("funnel") or {}).get("population")) or len(ranks)
        )
        required = min(int(minimum_coverage), population)
        if len(ranks) < required:
            raise UniverseMembershipError(
                f"universe_membership (run_date={membership.get('run_date')}): "
                f"{field!r} ranks {len(ranks)} of {population} name(s) in basis "
                f"{basis!r}, short of the {required} the consumer's rank ceiling "
                f"({minimum_coverage}) needs. A short table would make every name "
                f"past {len(ranks)} indistinguishable from a name the scanner did "
                f"not rank — refusing (alpha-engine-config-I7843)."
            )
    return ranks, str(basis)


def resolve_funnel_cut(
    consumer: str,
    *,
    bucket: str | None = None,
    s3_client: Any = None,
    membership: dict | None = None,
    minimum_rank_coverage: int | None = None,
) -> tuple[list[str], dict[str, int], dict]:
    """``(cut_tickers, {ticker: rank}, provenance)`` for a declared consumer.

    The one entry point a non-producer consumer needs: it reads the latest
    membership artifact, resolves the consumer's declared slot through the
    champion pointer, and returns both the membership (the cut) and the order
    (the rank table in the serving arm's own basis).

    ``membership`` is injectable so a caller that has already read the artifact
    does not read it twice; everything else is resolved here.

    Raises on every ambiguity. There is no degraded mode: an empty or defaulted
    window is indistinguishable from "the scanner selected nobody".
    """
    doc = membership if membership is not None else read_latest_membership(bucket=bucket, s3_client=s3_client)
    if not doc:
        raise UniverseMembershipError(
            f"no universe_membership/latest.json — {consumer} resolves its set "
            "from the membership artifact and cannot fall back to the raw universe"
        )
    serving, declared = serving_cut_for(doc, consumer, bucket=bucket, s3_client=s3_client)
    cut = (doc.get("cuts") or {}).get(serving) or {}
    tickers = list(cut.get("tickers") or [])
    if not tickers:
        raise UniverseMembershipError(
            f"universe_membership/latest.json (run_date={doc.get('run_date')}) "
            f"carries no tickers under {serving!r}, the cut serving "
            f"funnel.advances_to.{consumer} — refusing to hand {consumer} an "
            "empty window"
        )
    ranks, basis = rank_table_for_cut(doc, serving, minimum_coverage=minimum_rank_coverage)
    provenance = {
        "consumer": consumer,
        "cut": serving,
        "declared_cut": declared,
        "basis": basis,
        "size": len(tickers),
        "declared_size": cut.get("size"),
        "run_date": doc.get("run_date"),
        "cut_effective_date": doc.get("cut_effective_date"),
        "cut_refresh_cadence": doc.get("cut_refresh_cadence"),
        "rank_table_size": len(ranks),
        "schema_version": doc.get("schema_version"),
    }
    return tickers, ranks, provenance


def read_latest_membership(*, bucket: str | None = None, s3_client: Any = None) -> dict | None:
    """The current ``universe_membership/latest.json``, or ``None`` if absent.

    Absence is a legitimate first-run state and returns ``None``; any other
    error propagates. A read failure must not be silently read as "no prior
    cut", which under ``weekly`` would re-cut mid-week and quietly restore the
    daily churn this exists to stop.
    """
    s3 = _client(s3_client)
    b = _bucket(bucket)
    try:
        body = s3.get_object(Bucket=b, Key="universe_membership/latest.json")["Body"].read()
    except Exception as exc:
        if _is_missing_key(exc):
            return None
        raise
    return json.loads(body)


def _is_missing_key(exc: Exception) -> bool:
    """True only for a genuine "object does not exist" S3 error.

    Narrow on purpose. Every other failure — credentials, throttling, a denied
    bucket — must propagate, because under ``weekly`` a swallowed read would
    look like "no prior cut", re-cut mid-week, and quietly restore the daily
    churn this module exists to bound.
    """
    if type(exc).__name__ == "NoSuchKey":
        return True
    code = getattr(exc, "response", {}).get("Error", {}).get("Code")
    return code in {"NoSuchKey", "404"}


def carry_forward_cuts(fresh: dict, prior: dict) -> dict:
    """``fresh`` with the prior run's cuts held, and today's ranks kept.

    Ranks are deliberately NOT carried: they are today's ~900-name view, and
    the gap between fresh ranks and a held cut is precisely what an operator
    needs in order to see how stale the cut has become. ``cut_effective_date``
    is what says which is which.

    ``predictor_universe_cut`` is carried from the prior artifact rather than
    re-read from :data:`PREDICTOR_UNIVERSE_CUT`, so changing that constant
    cannot take effect mid-week without a re-cut.
    """
    held = dict(fresh)
    held["cuts"] = prior["cuts"]
    held["predictor_universe_cut"] = prior.get("predictor_universe_cut", fresh.get("predictor_universe_cut"))
    held["cut_effective_date"] = prior["cut_effective_date"]
    return held


def _bucket(bucket: str | None) -> str:
    return bucket or os.environ.get("S3_BUCKET") or _DEFAULT_BUCKET


def _client(s3_client: Any):
    if s3_client is not None:
        return s3_client
    import boto3

    return boto3.client("s3")


def _ranked(scores: dict[str, float], rank_key: str, score_key: str) -> dict[str, dict]:
    """``{ticker: {rank_key: int, score_key: float}}``, rank 1 = highest score.

    The one ranking primitive in this module. Ties broken by ticker so the
    table is deterministic across runs (a reproducible rank matters — consumers
    diff these week over week), and the KEY NAMES are a parameter rather than
    hardcoded: a table of ``tech_score`` values carrying ``attractiveness_rank``
    keys is a lie a reader has no way to catch, and until
    alpha-engine-config-I7843 the tech cut was derived from exactly that.
    """
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return {ticker: {rank_key: i + 1, score_key: score} for i, (ticker, score) in enumerate(ordered)}


def _rank_table(attractiveness: dict[str, float]) -> dict[str, dict]:
    """``{ticker: {attractiveness_rank, attractiveness_score}}``, rank 1 = most
    attractive."""
    return _ranked(attractiveness, "attractiveness_rank", "attractiveness_score")


def _tech_score_universe_rank_table(gate_eligible_tech_scores: dict[str, float]) -> dict[str, dict]:
    """``{ticker: {tech_score_rank, tech_score}}`` over every MOMENTUM-PATH row.

    The full-universe table in the ``tech_score_rank`` basis
    (alpha-engine-config-I7843). Distinct from :func:`_tech_score_rank_table`,
    which ranks the champion's own 60 and answers a different question.

    Legitimately narrower than the attractiveness table: ``scan_path ==
    "momentum"`` (see :func:`momentum_path_tech_scores`) is the incumbent rule's
    own eligibility gate, and ranking a name it rejected would invent an
    ordering the incumbent never expressed. The WIDTH is therefore declared in
    the artifact (``rank_tables["tech_score_rank"].size``) rather than left for
    a consumer to discover by finding its ceiling unanswerable.
    """
    return _ranked(gate_eligible_tech_scores, "tech_score_rank", "tech_score")


def _top_n_by(ranks: dict[str, dict], rank_key: str, n: int) -> list[str]:
    """The N highest-ranked tickers, returned SORTED (set semantics — this is a
    membership list, and rank order is already recoverable from the table)."""
    top = sorted(ranks.items(), key=lambda kv: kv[1][rank_key])[:n]
    return sorted(t for t, _ in top)


def _top_n(ranks: dict[str, dict], n: int) -> list[str]:
    """The N most attractive tickers, SORTED."""
    return _top_n_by(ranks, "attractiveness_rank", n)


def _tech_score_rank_table(scanner_tickers: list[str], tech_scores: dict[str, float]) -> dict[str, dict]:
    """``{ticker: {tech_score_rank, tech_score}}`` over the SCANNER CUT only,
    rank 1 = highest ``tech_score``. Ties broken by ticker, matching
    ``_rank_table``'s determinism contract.

    Scoped to the cut rather than the full universe deliberately: ``tech_score``
    is only meaningful as the incumbent's *selection* rule, and the incumbent
    only ever selected from names that cleared its own gate. Ranking all 900
    would invent an ordering the incumbent never expressed.

    Names in the cut with no ``tech_score`` are omitted — the caller decides
    whether a partial table is acceptable (it is: the cut membership itself is
    unaffected, only the derived top-N narrows).
    """
    scored = {t: tech_scores[t] for t in dict.fromkeys(scanner_tickers) if isinstance(tech_scores.get(t), (int, float))}
    ordered = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))
    return {ticker: {"tech_score_rank": i + 1, "tech_score": score} for i, (ticker, score) in enumerate(ordered)}


def assert_gate_cut_feeds_nothing_live(membership: dict, run_date: str) -> None:
    """The gate cut must not be named as the source of any live consumer.

    This is the check whose ABSENCE produced alpha-engine-config-I6630: the RAG
    corpus was scoped to the gate cut while the champion was drawn from the
    attractiveness rank, they overlapped on 2 of 20, and nothing asserted the
    two were related. The funnel invariant added there catches a champion that
    escapes the feed cut; it does not catch a consumer that was pointed at the
    wrong 60 in the first place, because that consumer's own cut is internally
    consistent.

    Enforced on the ARTIFACT rather than by grepping consumers, because the
    artifact is the contract: every live consumer resolves its set through one
    of the fields checked here.
    """
    live_fields = ("predictor_universe_cut", "feed_cut")
    gate_names = {CHAMPION_CUT, GATE_BASELINE_CUT}
    for field in live_fields:
        named = membership.get(field)
        if named in gate_names:
            raise UniverseMembershipError(
                f"universe membership {run_date}: {field}={named!r} — the "
                f"scanner GATE cut is a recorded baseline and feeds nothing "
                f"live. It is a tech_score momentum ranking, near-disjoint "
                f"from the attractiveness rank the champion is drawn from "
                f"(12 of 60 overlap, measured 2026-08-17). Pointing a live "
                f"consumer at it is alpha-engine-config-I6630 repeating."
            )
    funnel = membership.get("funnel") or {}
    for consumer, named in (funnel.get("advances_to") or {}).items():
        if named in gate_names:
            raise UniverseMembershipError(
                f"universe membership {run_date}: funnel.advances_to."
                f"{consumer}={named!r} — the gate cut feeds nothing live."
            )


MAX_UNRANKABLE_FRACTION = 0.05
"""Ceiling on the share of the scanned universe with no rankable score.

1 of 903 on 2026-08-20 (``VMRK``, no factor-profile row at all). A handful of
names the factor pipeline could not score is a normal state; a fifth of the
universe unrankable means the profiles read is broken upstream, and the
membership artifact must not publish a rank table over the remains of it as if
nothing had happened.
"""


def assert_population_parity(membership: dict, run_date: str) -> None:
    """The ranked population must be a subset of the SCANNED universe.

    This is the guard whose absence produced alpha-engine-config-I7844. Two
    artifacts are written by one Scanner invocation from two different sources:
    the universe board iterates ``candidates.json::scanner_eval_log`` (the
    scanned universe), while this artifact's ranks came from every key in
    ``factors/profiles/{date}/by_ticker.json``. That file legitimately carries
    names outside the scanned universe — Metron-supplemental rows and
    fundamental-only rows — so the two populations drifted apart with nothing
    raising. Measured 2026-08-20: ``EQR``/``AVB``/``HYOAS``/``XLRE`` were
    ranked 98/420/421/422 with NO technical data at all (``momentum_n: 0``,
    ``low_vol_n: 0``, ``sector: "Unknown"``, three of them carrying identical
    neutral default pillars), and ``EQR`` sat inside Think Tank's
    ``rank_ceiling: 150``.

    Asserted against the EVAL LOG rather than by reading the board, for two
    reasons. The board's population IS the eval log's by construction
    (``scanner_orchestrator.write_universe_board_for_scanner_run`` builds its
    rows from ``scanner_eval_log``, verified equal on the live 2026-08-20
    artifacts), so the eval log is the same fact one hop earlier. And the board
    write is fail-soft dashboard-only while this artifact is load-bearing —
    making the predictor's universe depend on a best-effort artifact being
    readable is the coupling :func:`attractiveness_for_run` already refuses.

    The reverse direction — scanned names with no rank — is a legitimate,
    BOUNDED state (a name with no factor profile cannot be ranked and must not
    be fabricated a score), so it is recorded in
    ``population_reconciliation.unrankable`` and only raises past
    :data:`MAX_UNRANKABLE_FRACTION`.
    """
    recon = membership.get("population_reconciliation") or {}
    scanned = recon.get("scanned_universe_size")
    if not scanned:
        # No eval log was supplied (historical backfill). Recorded as such in
        # the reconciliation block rather than asserted against a population
        # this run does not have.
        return
    outside = recon.get("ranked_outside_scanned_universe") or []
    if outside:
        raise UniverseMembershipError(
            f"universe membership {run_date}: {len(outside)} ranked name(s) are "
            f"absent from the scanned universe ({outside[:20]}). ``ranks`` and "
            f"the universe board are two views of ONE Scanner run and must "
            f"cover the same names; a name ranked here with no board row is "
            f"invisible to every consumer that joins the two, and it was "
            f"ranked on whatever partial pillar data the factor profiles "
            f"happened to carry (alpha-engine-config-I7844)."
        )
    unrankable = recon.get("unrankable") or []
    if len(unrankable) > MAX_UNRANKABLE_FRACTION * int(scanned):
        raise UniverseMembershipError(
            f"universe membership {run_date}: {len(unrankable)} of {scanned} "
            f"scanned name(s) have no rankable attractiveness score, past the "
            f"{MAX_UNRANKABLE_FRACTION:.0%} allowance. A rank table over the "
            f"remains of a broken factor-profile read must not be published as "
            f"a universe ranking (alpha-engine-config-I7844). Unrankable: "
            f"{unrankable[:20]}"
        )


def assert_rank_tables_cover_promotable_cuts(membership: dict, run_date: str) -> None:
    """Every promotable cut's basis must have a full-universe rank table.

    The producer half of alpha-engine-config-I7843. A cut in
    :data:`PROMOTABLE_CUTS` can be handed the funnel by the champion pointer at
    any time, and the consumer of that slot resolves its rank ceiling in the
    SERVING arm's basis. A basis with no table — or a table too short to reach
    :data:`MIN_PROMOTABLE_RANK_COVERAGE` — makes the promotion fail on the
    morning it is made, in a consumer, with the cut already live. Checked here
    instead, on the run that emits the arm.

    EXISTENCE is what raises. Whether the table is WIDE ENOUGH for a given
    consumer's ceiling is recorded (``rank_tables[basis].serves_rank_ceiling``)
    and enforced by :func:`rank_table_for_cut` at read time, not raised on here:
    a table's width depends on how many names cleared that arm's own
    eligibility gate this cycle, and an observe-only arm narrowing must never
    red a Scanner run whose load-bearing output — the predictor's universe — is
    fine. A basis with NO table is the opposite: a producer defect, decided by
    the code and identical on every run.
    """
    cuts = membership.get("cuts") or {}
    index = _rank_table_index(membership)

    # An OBSERVE-ONLY arm gets the same check, RECORDED rather than raised
    # (alpha-engine-config-I8060). Its basis losing its table costs the arm its
    # rank-IC on the leaderboard and nothing live, so redding a Scanner run over
    # it would be the wrong trade — but a measurement quietly disappearing must
    # still be visible, or "non-promotable" decays into "unmeasured" without
    # anyone deciding it (champion-challenger-policy.md §3).
    for cut_name in OBSERVE_ONLY_CUTS:
        cut = cuts.get(cut_name)
        if cut is None:
            continue
        basis = str(cut.get("basis"))
        entry = index.get(basis)
        if not entry or not membership.get(entry[0]):
            logger.warning(
                "[universe_membership] %s: observe-only arm %r is ranked by %r, "
                "for which this artifact emits no full-universe rank table "
                "(it emits one for %s). The arm is still scored on "
                "topn_alpha_vs_population; its rank-IC will read MISSING until "
                "the table returns (alpha-engine-config-I8060).",
                run_date, cut_name, basis, sorted(index),
            )

    for cut_name in PROMOTABLE_CUTS:
        cut = cuts.get(cut_name)
        if cut is None:
            # An arm can be legitimately absent (no eval log, unavailable
            # shadow profiles) — that is recorded as a miss for the arm, and it
            # is not promotable while it is missing.
            continue
        basis = str(cut.get("basis"))
        entry = index.get(basis)
        table = membership.get(entry[0]) if entry else None
        if not entry or not table:
            raise UniverseMembershipError(
                f"universe membership {run_date}: promotable cut {cut_name!r} is "
                f"ranked by {basis!r}, for which this artifact emits no "
                f"full-universe rank table (it emits one for {sorted(index)}). "
                f"The champion pointer can hand this arm the funnel at any "
                f"time, and the consumer of that slot resolves its rank ceiling "
                f"in the SERVING arm's basis — so an arm with no table is a "
                f"promotion that fails in a consumer on the morning it is made "
                f"(alpha-engine-config-I7843)."
            )


def assert_cut_invariants(membership: dict, run_date: str) -> None:
    """Raise ``UniverseMembershipError`` if the artifact's cuts violate an invariant.

    Extracted from :func:`build_universe_membership` so it can also run against
    a CARRIED-FORWARD artifact (alpha-engine-config-I6666). Under the weekly
    cadence the held cut is what the predictor consumes for the rest of the
    week, so a count-match or funnel violation in it is exactly as load-bearing
    as one in a freshly derived cut — guarding only the fresh path would leave
    the long-lived artifact unchecked.
    """
    _cuts = membership.get("cuts") or {}

    # Count-match is the whole point of N=20 (alpha-engine-config-I4983): an
    # arm's win must not be confounded between selection rule and breadth. When
    # the incumbent arm is emitted, its size MUST equal the champion cut's —
    # a partial eval log that yields n=10 against a n=20 champion is a red run,
    # never a silently-skewed comparison. (Absence of the incumbent arm when
    # no tech_scores were supplied is an honest degrade for historical
    # backfills; that path does not enter this check.)
    champion = _cuts.get(PREDICTOR_UNIVERSE_CUT)
    incumbent_cut = _cuts.get(f"scanner_top_{_INCUMBENT_CUT_N}")
    if champion is not None and incumbent_cut is not None:
        if champion["size"] != incumbent_cut["size"]:
            raise UniverseMembershipError(
                f"universe membership {run_date}: count-match broken — "
                f"{PREDICTOR_UNIVERSE_CUT}.size={champion['size']} vs "
                f"scanner_top_{_INCUMBENT_CUT_N}.size={incumbent_cut['size']} "
                f"(both must equal {_INCUMBENT_CUT_N}; a short tech_score table "
                f"must not silently narrow the incumbent arm)"
            )

    # The funnel invariant (alpha-engine-config-I6630). The champion cut is the
    # HEAD of the feed cut, and every downstream consumer is entitled to assume
    # it: the RAG corpus scopes to the feed cut so the scored names get
    # evidence, and Think Tank's gap-fill window is the same 60.
    #
    # This holds by construction while both are cuts of one ranking — which is
    # exactly why it went unchecked, and exactly how it broke. The corpus was
    # scoped to ``scanner_candidates`` (a tech_score momentum GATE, also 60
    # wide) against a champion drawn from the attractiveness RANK; measured
    # 2026-08-07 they overlapped on 2 of 20. Nothing failed, because nothing
    # asserted the two were related.
    #
    # Asserted at the producer so a champion moved outside the feed cut is a
    # RED SCANNER RUN, not a consumer's problem to discover weeks later. If a
    # future champion legitimately sits outside the attractiveness rank family
    # (a Think Tank promotion, say), the feed cut must move in the same change
    # — that is the coupling this enforces, not an obstacle to it.
    champion_cut = _cuts.get(PREDICTOR_UNIVERSE_CUT)
    feed_cut = _cuts.get(FEED_CUT_NAME)
    if champion_cut is not None and feed_cut is not None:
        escaped = sorted(set(champion_cut["tickers"]) - set(feed_cut["tickers"]))
        if escaped:
            raise UniverseMembershipError(
                f"universe membership {run_date}: funnel invariant broken — "
                f"{len(escaped)} of {champion_cut['size']} ticker(s) in the "
                f"champion cut {PREDICTOR_UNIVERSE_CUT!r} are absent from the "
                f"feed cut {FEED_CUT_NAME!r} ({escaped}). The champion must be "
                f"the head of the feed set: the corpus fills evidence for the "
                f"feed cut, so a champion outside it is scored on cold context "
                f"(alpha-engine-config-I6630)."
            )

    # Momentum-horizon challenger arm (alpha-engine-config-I7538): the same
    # two guards the incumbent arm gets, for the same reasons.
    #
    # COUNT-MATCH — the arm exists to answer "which momentum horizon ranks
    # better", and an arm compared at a different width answers a question
    # about breadth instead (§4).
    #
    # VACUITY — if the challenger resolves to the SAME membership as the
    # champion, every assertion still passes and the leaderboard reports a
    # well-formed comparison of an arm against itself (§4, alpha-engine-config
    # -I6429). The two definitions differ only in the momentum pillar, so an
    # identical top-20 is the signature of the override having silently not
    # applied — precisely the failure this must not render as a tie.
    for prefix, n in [(p, n) for p in (CHALLENGER_CUT_PREFIX, MOMZERO_CUT_PREFIX) for n in _CHALLENGER_CUT_NS]:
        challenger_cut = _cuts.get(f"{prefix}{n}")
        if challenger_cut is None:
            continue
        if challenger_cut["size"] != n:
            raise UniverseMembershipError(
                f"universe membership {run_date}: count-match broken — "
                f"{prefix}{n}.size={challenger_cut['size']}, "
                f"must equal {n}. A short challenger table must not silently "
                f"narrow the arm and turn a horizon comparison into a breadth one."
            )
        counterpart = _cuts.get(f"attractiveness_top_{n}")
        if counterpart is not None and set(challenger_cut["tickers"]) == set(counterpart["tickers"]):
            raise UniverseMembershipError(
                f"universe membership {run_date}: vacuous challenger — "
                f"{prefix}{n} resolves to the same {n} names as "
                f"attractiveness_top_{n}. The two composites differ in the "
                f"momentum pillar, so identical membership means the challenger "
                f"definition did not apply. Scoring this would compare the "
                f"champion against itself and report it as a tie "
                f"(champion-challenger-policy.md §4)."
            )


def compute_turnover(current: dict, prior: dict | None) -> dict | None:
    """Per-cut membership delta of ``current`` against the PRIOR write, or None.

    Producer-side rather than consumer-side (alpha-engine-config-I6785) for two
    reasons. First, the number has to be durable and readable without loading N
    artifacts — a monitor asking "did the cut move?" should not have to
    reconstruct a series. Second, the consumer-side derivation is exactly what
    the clobber defect proved cannot be trusted: the dashboard computes churn
    from dated keys that a later same-day writer had already replaced.

    "Prior" means the immediately-preceding WRITE, not the preceding calendar
    day. On a day with both a preopen and an exercise run, the exercise run's
    turnover is measured against that morning's cut, which is the honest
    comparison — and ``prior_run_date`` + ``prior_generated_at`` say so
    explicitly rather than leaving the reader to assume a day boundary.

    Returns None (never a block of zeros) when there is no usable prior. Zeros
    would read as "nothing changed"; the truth is "nothing to compare against",
    and a surface that cannot tell those apart invents a 100%-retention point
    at the start of every series.

    Only cuts present in BOTH artifacts are compared. A cut that appears or
    disappears between writes is a schema change, and reporting it as 100%
    churn would misattribute a producer edit to the market.
    """
    if not prior:
        return None
    prior_generated = prior.get("generated_at")
    if prior_generated and prior_generated == current.get("generated_at"):
        # The prior pointer still holds THIS artifact — a re-write of the same
        # run. Comparing it to itself would publish a fabricated 100% retention.
        return None

    cur_cuts = current.get("cuts") or {}
    prior_cuts = prior.get("cuts") or {}
    per_cut: dict[str, dict] = {}
    for name, block in cur_cuts.items():
        prior_block = prior_cuts.get(name)
        if not prior_block:
            continue
        now = {str(t).upper() for t in (block.get("tickers") or [])}
        was = {str(t).upper() for t in (prior_block.get("tickers") or [])}
        if not was:
            continue
        retained = len(now & was)
        per_cut[name] = {
            "size": len(now),
            "retained": retained,
            "added": len(now - was),
            "dropped": len(was - now),
            "retention_pct": round(100.0 * retained / len(was), 2),
        }
    if not per_cut:
        return None
    return {
        "prior_run_date": prior.get("run_date"),
        "prior_generated_at": prior_generated,
        "per_cut": per_cut,
    }


def _live_champion_name() -> str:
    """The scanner arm the live cut is ranked by, from the register.

    Read rather than hardcoded so a future promotion updates this artifact's
    ``ranked_by`` automatically — the drift this whole change exists to close
    was a hardcoded ranking name surviving a cutover
    (alpha-engine-config-I7808). Falls back to a literal ``"unknown"`` rather
    than to a guess: an unreadable register must not be able to make the
    artifact ASSERT a ranking.
    """
    try:
        from data.scanner_specs import live_champion_spec

        return live_champion_spec().name
    except Exception:  # noqa: BLE001 — naming metadata, never load-bearing
        return "unknown"


def build_universe_membership(
    run_date: str,
    scanner_tickers: list[str],
    attractiveness: dict[str, float],
    *,
    tech_scores: dict[str, float] | None = None,
    gate_eligible_tech_scores: dict[str, float] | None = None,
    challenger_attractiveness: dict[str, float] | None = None,
    momzero_attractiveness: dict[str, float] | None = None,
    scanned_universe: list[str] | None = None,
    generated_at: str | None = None,
    backfilled_from: str | None = None,
    prior: dict | None = None,
) -> dict:
    """Assemble the membership artifact from already-resolved inputs.

    Pure — no S3, no clock beyond the ``generated_at`` default — so the schema
    contract is unit-testable without fixtures or network.

    Parameters
    ----------
    run_date : ISO date of the cycle this membership describes.
    scanner_tickers : the scanner's gate cut, verbatim from
        ``candidates.json::scanner_tickers``.
    attractiveness : ``{ticker: attractiveness_score}`` for the full scanned
        universe. Names with a null/absent score must be filtered by the caller;
        an unrankable name cannot be in a rank cut.
    tech_scores : ``{ticker: tech_score}`` from
        ``candidates.json::scanner_eval_log``. When supplied, emits the
        ``scanner_top_20`` incumbent-challenger cut plus the ``scanner_ranks``
        table, and FAILS LOUD if that arm cannot be count-matched to the
        champion (partial eval logs that would emit n≠20). Optional so
        historical backfills, whose inputs may predate the eval-log field,
        still build — they omit the incumbent arm rather than fabricating one.
    gate_eligible_tech_scores : ``{ticker: tech_score}`` restricted to
        ``scan_path == "momentum"`` rows (:func:`momentum_path_tech_scores`).
        When supplied, emits the ``tech_score_top_60`` displaced-incumbent
        baseline. Kept separate from ``tech_scores`` rather than derived from
        it because the two answer different questions and applying the wrong
        one silently produces a cut over names the incumbent rule rejected.
    scanned_universe : every ticker the scanner EVALUATED this run
        (``candidates.json::scanner_eval_log::ticker``, via
        :func:`scanned_universe_from_eval_log`). This is the universe board's
        population by construction, so supplying it lets the producer assert
        that its own rank table and the board cover the same names
        (alpha-engine-config-I7844). ``None`` — the historical-backfill path,
        which has no eval log — skips the parity assertion and records the
        absence in ``population_reconciliation`` rather than asserting against
        a population the run does not have.
    prior : the previous membership artifact, when one was readable. Used only
        to compute the ``turnover`` block; passing None yields ``turnover:
        null`` rather than a fabricated zero-churn record. Kept a parameter
        rather than an S3 read inside this function so the schema contract
        stays pure and unit-testable without fixtures or network.
    backfilled_from : provenance string when this artifact is RECONSTRUCTED from
        historical inputs rather than emitted by the live run. Consumers must be
        able to tell a reconstruction from a first-class write — a backfill can
        only be as good as the artifacts that survived, and silently presenting
        one as the other is how a gap becomes invisible.

    Raises
    ------
    UniverseMembershipError
        If either input is empty. Both are required for a meaningful artifact,
        and a membership file that silently claims "nobody is in any cut" is the
        exact failure mode I4818 was: consumers cannot tell it from a real empty.
    """
    if not scanner_tickers:
        raise UniverseMembershipError(
            f"universe membership {run_date}: scanner_tickers is empty — the "
            "scanner cut is the predictor's universe source and cannot be empty"
        )
    if not attractiveness:
        raise UniverseMembershipError(
            f"universe membership {run_date}: no rankable attractiveness scores — rank cuts would all be empty"
        )

    ranks = _rank_table(attractiveness)
    _gate_block = {
        # NOT "scanner_gate", and not "tech_score_rank" either. The live cut is
        # ranked by the scanner slot's CHAMPION arm, which has been the momentum
        # sleeve since the 2026-07-22 config#1186 cutover; this basis said
        # "scanner_gate" while the docstring above said "a tech_score momentum
        # ranking", and both were read as tech_score by consumers and by
        # humans (alpha-engine-config-I7808). The champion's identity is
        # resolved from the register rather than hardcoded, so a future
        # promotion cannot leave this string behind the way the last one did.
        "basis": "scanner_champion_rank",
        "ranked_by": _live_champion_name(),
        "size": len(set(scanner_tickers)),
        "tickers": sorted(set(scanner_tickers)),
        "source": f"candidates/{run_date}/candidates.json::scanner_tickers",
        "feeds": [],
        "role": (
            "the scanner slot's champion-ARM artifact — a candidate-generation "
            "experiment, scored on the scanner leaderboard. Feeds NOTHING as of "
            "2026-08-20: the sector teams moved to the champion CUT resolved by "
            "universe_membership.resolve_feed_cut, and no other consumer reads "
            "it (alpha-engine-config-I7823). It fed the sector teams before "
            "that only because _resolve_agent_input_set happened to read it, "
            "which is how a cutover in another module silently replaced the "
            "researched set with a disjoint one for four weeks."
        ),
    }
    cuts: dict[str, dict] = {
        CHAMPION_CUT: dict(_gate_block),
        # Deprecated alias (alpha-engine-config-I7818, successor to I7578).
        # Emitted so a consumer pinned on the old name keeps working through
        # the deprecation window rather than reading a missing key as an empty
        # cut — silently trading a rename for a zero-size universe is the
        # worse failure. Known live reader:
        # crucible-dashboard/loaders/universe_churn.py (migrating to
        # CHAMPION_CUT in the same change — see I7818).
        #
        # ``scanner_candidates`` (the I7578 alias) is NOT re-emitted here: its
        # own deprecation window is closed and I7818 retires it outright.
        GATE_BASELINE_CUT: {
            **_gate_block,
            "deprecated_alias_for": CHAMPION_CUT,
            "removal_tracked_by": "alpha-engine-config-I7818 follow-up",
        },
    }
    for n in _RANK_CUTS:
        tickers = _top_n(ranks, n)
        cuts[f"attractiveness_top_{n}"] = {
            "basis": "attractiveness_rank",
            "size": len(tickers),
            "tickers": tickers,
            "source": f"scanner/universe/{run_date}/universe.json::attractiveness_score",
            # Named per width rather than as one blanket list: these three cuts
            # have genuinely different consumers, and a shared value would make
            # "who reads the 25?" unanswerable from the artifact.
            "feeds": _FEEDS_BY_RANK_CUT.get(n, []),
        }

    # Incumbent challenger arm (alpha-engine-config-I4983). ``scanner_candidates``
    # is stored alphabetically — set semantics — so the incumbent's own ordering
    # is NOT recoverable from it. Without this table there is no way to ask "what
    # would the incumbent rule have picked at N?", which is precisely the
    # comparison the champion flip has to be judged against.
    scanner_ranks = _tech_score_rank_table(scanner_tickers, tech_scores) if tech_scores else {}
    if scanner_ranks:
        top = sorted(scanner_ranks.items(), key=lambda kv: kv[1]["tech_score_rank"])
        incumbent = sorted(t for t, _ in top[:_INCUMBENT_CUT_N])
        cuts[f"scanner_top_{_INCUMBENT_CUT_N}"] = {
            "basis": "tech_score_rank",
            "size": len(incumbent),
            "tickers": incumbent,
            "source": f"candidates/{run_date}/candidates.json::scanner_eval_log::tech_score",
        }

    # Displaced-incumbent baseline (alpha-engine-config-I7809). The universe's
    # tech_score top-N over the rows the momentum path admitted — what the live
    # cut WAS before the 2026-07-22 cutover. Emitted unconditionally when the
    # projection is non-empty rather than gated on a config flag: its whole
    # purpose is that the cutover stays measurable, and a baseline that can be
    # switched off is a baseline that is off on the cycle you needed it.
    #
    # The full table is now KEPT, not discarded after the top-60 slice
    # (alpha-engine-config-I7843): this cut is promotable, and the consumer of
    # the slot it can be promoted into resolves its rank ceiling in this basis.
    tech_score_ranks = _tech_score_universe_rank_table(gate_eligible_tech_scores or {})
    if gate_eligible_tech_scores:
        for n in (ATTRACTIVENESS_FEED_TOP_N,):
            tickers = _top_n_by(tech_score_ranks, "tech_score_rank", n)
            cuts[f"{TECH_SCORE_CUT_PREFIX}{n}"] = {
                "basis": "tech_score_rank",
                "size": len(tickers),
                "tickers": tickers,
                "source": (f"candidates/{run_date}/candidates.json::scanner_eval_log::tech_score@scan_path=momentum"),
                "feeds": [],
                "role": (
                    "recorded baseline — the incumbent ranking the 2026-07-22 "
                    "config#1186 cutover displaced, at the champion's width. "
                    "Feeds nothing live."
                ),
                # Annotated, NOT raised on. Identity here means the live cut
                # fell back to the incumbent ordering — which is the DOCUMENTED
                # degrade when factor loadings are unavailable
                # (SCANNER_CONTRACT.md §5), so reding the Scanner run for it
                # would turn a graceful degradation into an outage. It still
                # has to be visible: a cycle where the champion silently did
                # not apply is a cycle whose leaderboard row is about a
                # different arm than its label claims.
                "equals_live_cut": set(tickers) == set(scanner_tickers),
            }

    # Momentum-horizon challenger arm (alpha-engine-config-I7538). Ranked over
    # the SAME universe by the SAME machinery — only the momentum pillar's
    # component definition differs. Omitted entirely rather than faked when the
    # challenger profiles are unavailable: an absent arm is a recorded miss
    # (champion-challenger-policy.md §3), whereas a cut silently falling back
    # to champion scores would enter the leaderboard as a challenger that is
    # actually the champion — the §4 vacuity case, and it would read as a
    # legitimate tie forever.
    if challenger_attractiveness:
        challenger_ranks = _rank_table(challenger_attractiveness)
        for n in _CHALLENGER_CUT_NS:
            tickers = _top_n(challenger_ranks, n)
            cuts[f"{CHALLENGER_CUT_PREFIX}{n}"] = {
                "basis": "attractiveness_rank_mom121",
                "size": len(tickers),
                "tickers": tickers,
                "source": (f"scanner/factor_profiles_shadow/mom121/{run_date}/profiles.json::attractiveness_score"),
            }

    # Zero-momentum arm (alpha-engine-config-I7574). Same champion factor
    # profiles, same pillar scores — only the WEIGHT vector differs, so this
    # arm and the champion cannot disagree about anything except how much the
    # momentum pillar counted. Same absent-is-a-miss rule as above.
    if momzero_attractiveness:
        momzero_ranks = _rank_table(momzero_attractiveness)
        for n in _CHALLENGER_CUT_NS:
            tickers = _top_n(momzero_ranks, n)
            cuts[f"{MOMZERO_CUT_PREFIX}{n}"] = {
                "basis": "attractiveness_rank_momzero",
                "size": len(tickers),
                "tickers": tickers,
                "source": (f"factors/profiles/{run_date}/by_ticker.json::attractiveness_score@momentum_weight=0"),
            }

    assert_cut_invariants({"cuts": cuts}, run_date)

    # ── Population reconciliation (alpha-engine-config-I7844) ───────────────
    # The scanned universe is the board's population by construction. Stating
    # both sides and their difference IN the artifact is what makes the two
    # halves of one Scanner run checkable by a reader who has only one of them.
    scanned = sorted({str(t).upper() for t in (scanned_universe or [])})
    ranked_set = set(ranks)
    reconciliation: dict[str, Any] = {
        "source": f"candidates/{run_date}/candidates.json::scanner_eval_log::ticker",
        "scanned_universe_size": len(scanned) or None,
        "ranked": len(ranks),
        "unrankable": sorted(set(scanned) - ranked_set) if scanned else [],
        "ranked_outside_scanned_universe": sorted(ranked_set - set(scanned)) if scanned else [],
        "unrankable_reason": (
            "in the scanned universe with no rankable attractiveness score — no "
            "factor-profile row, or every pillar leg undefined. Carried on the "
            "universe board with attractiveness_score=null and counted there in "
            "attractiveness_coverage.excluded_tickers; absent from ``ranks``, "
            "because a fabricated score would occupy a rank position."
        ),
    }
    if not scanned:
        reconciliation["note"] = (
            "no scanner_eval_log was supplied (historical backfill) — the ranked "
            "population could not be reconciled against the scanned universe on "
            "this run, and was NOT asserted."
        )

    membership = {
        "schema_version": SCHEMA_VERSION,
        "producer": PRODUCER,
        "run_date": run_date,
        "generated_at": generated_at or datetime.now(UTC).isoformat(timespec="seconds"),
        "universe_count": len(ranks),
        "predictor_universe_cut": PREDICTOR_UNIVERSE_CUT,
        "feed_cut": FEED_CUT_NAME,
        # alpha-engine-config-I7578. The chain, stated in the artifact rather
        # than only in this module, so a reader can answer "what advances?"
        # without reading the producer's source. Every name here is a key in
        # ``cuts`` except ``population``, which is the rank table's width.
        "funnel": {
            "population": len(ranks),
            "feed_cut": {"name": FEED_CUT_NAME, "size": len(cuts.get(FEED_CUT_NAME, {}).get("tickers", []))},
            "champion_cut": {
                "name": PREDICTOR_UNIVERSE_CUT,
                "size": len(cuts.get(PREDICTOR_UNIVERSE_CUT, {}).get("tickers", [])),
            },
            "advances_to": {
                "predictor_universe": PREDICTOR_UNIVERSE_CUT,
                "rag_corpus_scope": FEED_CUT_NAME,
                "thinktank_coverage_window": FEED_CUT_NAME,
            },
            "feeds_nothing_live": [CHAMPION_CUT, GATE_BASELINE_CUT],
        },
        # alpha-engine-config-I6666 — a freshly built artifact is by definition
        # re-cut today; ``carry_forward_cuts`` overwrites this when the run
        # holds the prior cut instead. ``cut_effective_date == run_date`` is
        # therefore the test for "this cut is today's".
        "cut_effective_date": run_date,
        "population_reconciliation": reconciliation,
        "cuts": cuts,
        "ranks": ranks,
    }
    if scanner_ranks:
        membership["scanner_ranks"] = scanner_ranks
    if tech_score_ranks:
        membership[TECH_SCORE_RANKS_FIELD] = tech_score_ranks
    # The basis -> full-universe-table INDEX, declared in the artifact rather
    # than left in a constant on the consumer side (alpha-engine-config-I7843).
    # A consumer resolving its serving arm's rank ceiling asks this block which
    # field to read and how wide it is; adding a basis is then a producer-side,
    # versioned change with no matching edit in every reader — the same reason
    # ``funnel.advances_to`` is declared here rather than restated downstream.
    membership["rank_tables"] = _rank_tables_block(membership, len(ranks))
    if backfilled_from:
        membership["backfilled_from"] = backfilled_from
    # Runs on the ASSEMBLED artifact, unlike the cut invariants above: what
    # these check are the routing fields and the rank tables, which do not
    # exist until the dict is built.
    assert_gate_cut_feeds_nothing_live(membership, run_date)
    assert_population_parity(membership, run_date)
    assert_rank_tables_cover_promotable_cuts(membership, run_date)
    membership["turnover"] = compute_turnover(membership, prior)
    return membership


def run_stamp(generated_at: str) -> str:
    """``2026-08-10T12:49:30+00:00`` → ``20260810T124930Z``.

    Derived from the artifact's own ``generated_at`` rather than from the wall
    clock at write time, so the immutable key and the payload can never
    disagree about when the cut was computed. Normalised to UTC first: two runs
    stamped in different offsets must not sort into the wrong order in an
    ``s3 ls``, which is how an operator reads this prefix.

    Falls back to the raw string with unsafe characters stripped if the value
    is unparseable — an odd key is recoverable, a dropped write is not.
    """
    try:
        parsed = datetime.fromisoformat(generated_at)
    except (TypeError, ValueError):
        return "".join(c for c in str(generated_at) if c.isalnum())
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def write_universe_membership_to_s3(
    membership: dict,
    run_date: str,
    *,
    bucket: str | None = None,
    s3_client: Any = None,
) -> str:
    """Write the immutable ``runs/`` copy plus the two pointer keys. Returns the
    dated pointer key (unchanged return contract).

    Write ORDER is load-bearing: the immutable copy lands FIRST. If the process
    dies between writes, the surviving state is "the run is recorded but the
    pointers still name the previous one" — recoverable. The reverse order
    yields pointers naming a cut with no durable copy, which is the very state
    I6785 was filed about.
    """
    s3 = _client(s3_client)
    b = _bucket(bucket)
    body = json.dumps(membership, separators=(",", ":"), default=str).encode("utf-8")
    stamp = run_stamp(membership.get("generated_at") or "")
    run_key = f"universe_membership/{run_date}/runs/{stamp}.json"
    dated_key = f"universe_membership/{run_date}/membership.json"
    for key in (run_key, dated_key, "universe_membership/latest.json"):
        s3.put_object(Bucket=b, Key=key, Body=body, ContentType="application/json")
    turnover = membership.get("turnover") or {}
    logger.info(
        "[universe_membership] wrote %d cuts over %d ranked names → s3://%s/%s (+dated, +latest); turnover vs %s: %s",
        len(membership.get("cuts", {})),
        membership.get("universe_count", 0),
        b,
        run_key,
        turnover.get("prior_run_date") or "n/a",
        {k: v["retention_pct"] for k, v in (turnover.get("per_cut") or {}).items()} or "none",
    )
    return dated_key


def attractiveness_from_board(board: dict) -> dict[str, float]:
    """``{ticker: attractiveness_score}`` from a universe-board artifact, skipping
    names with no score (unrankable — never coerced to 0, which would rank them
    as the least attractive names rather than as absent).

    Used by the historical backfill (``scripts/backfill_universe_membership.py``),
    which has dated boards but no longer has the factor profiles those runs read.
    The live path uses :func:`attractiveness_for_run` instead — see its docstring
    for why the board is the wrong live input.
    """
    out: dict[str, float] = {}
    for stock in (board or {}).get("stocks") or []:
        ticker = stock.get("ticker")
        score = stock.get("attractiveness_score")
        if ticker and score is not None:
            out[str(ticker)] = float(score)
    return out


def _restrict_to_scanned_universe(
    profiles: dict,
    scanned_universe: list[str] | None,
    *,
    arm: str,
) -> dict:
    """``profiles`` restricted to the names the scanner actually evaluated.

    Applied BEFORE the cross-sectional scoring chokepoint, never after
    (alpha-engine-config-I7844). Attractiveness is a percentile over the
    population being scored, so filtering afterwards leaves every surviving
    score computed against a population that included names the scanner never
    saw. Measured on the live 2026-08-20 artifacts: scoring all 906 profile rows
    and scoring the 902 that the scanner evaluated gave DIFFERENT scores for
    860 of the 902 common names, and the two orderings first diverged at rank
    26 — while filtering first reproduces the universe board's scores exactly,
    which is the byte-identity this module's docstring has always claimed and
    did not have.

    ``None`` leaves the profiles untouched: the historical backfill has no eval
    log, and inventing one would be worse than an unreconciled artifact that
    says so.
    """
    if not scanned_universe:
        return profiles
    keep = {str(t).upper() for t in scanned_universe}
    restricted = {t: v for t, v in profiles.items() if str(t).upper() in keep}
    dropped = sorted(set(profiles) - set(restricted))
    if dropped:
        logger.info(
            "[universe_membership] %s: %d factor-profile row(s) are outside the "
            "scanned universe and are NOT ranked (%s%s) — the profiles file "
            "legitimately carries Metron-supplemental and fundamental-only names "
            "the scanner never evaluated (alpha-engine-config-I7844)",
            arm,
            len(dropped),
            dropped[:20],
            "..." if len(dropped) > 20 else "",
        )
    if not restricted:
        raise UniverseMembershipError(
            f"{arm}: no factor-profile row survives restriction to the "
            f"{len(keep)}-name scanned universe — the profiles and the scanner "
            f"eval log describe disjoint populations, which is a broken read, "
            f"never an empty universe (alpha-engine-config-I7844)"
        )
    return restricted


def attractiveness_for_run(
    run_date: str,
    *,
    scanned_universe: list[str] | None = None,
    bucket: str | None = None,
    s3_client: Any = None,
) -> dict[str, float]:
    """``{ticker: attractiveness_score}`` for ``run_date``, computed from the
    run's factor profiles via the SSOT cross-sectional chokepoint, over the
    SCANNED universe.

    Deliberately NOT sourced from the universe board, for two reasons:

      1. **Failure independence.** The board write is best-effort (dashboard-only
         visibility) and its caller swallows failures; this artifact is
         load-bearing. Sourcing membership from the board would make a
         non-fatal board hiccup silently degrade the predictor's universe.
      2. **The board's gate block is not trustworthy** — 2026-07-02 reported zero
         gate-passing names while the scanner cut was the normal 60
         (alpha-engine-config-I4820). Membership must never inherit that.

    Factor profiles are written earlier in the same Scanner invocation
    (``compute_and_write_factor_profiles``), so this read is same-run data, and
    it goes through the identical pillar-weighting chokepoint the board uses.

    ``scanned_universe`` is what makes the ranks here byte-identical to the
    console board's — a property this docstring asserted for weeks while it was
    false. The profiles file is NOT the scanned universe: it legitimately
    carries Metron-supplemental and fundamental-only rows the scanner never
    evaluated, and scoring them shifted every percentile in the table
    (alpha-engine-config-I7844, corrected 2026-08-20). Omit it only where no
    eval log exists — the historical backfill.
    """
    from scoring.universe_board import (
        _read_factor_profiles,
        attractiveness_from_factor_profiles,
    )

    profiles = _read_factor_profiles(run_date, bucket, s3_client) or {}
    if not profiles:
        raise UniverseMembershipError(
            f"universe membership {run_date}: no factor profiles readable — attractiveness rank cuts cannot be built"
        )
    profiles = _restrict_to_scanned_universe(profiles, scanned_universe, arm=f"champion attractiveness {run_date}")
    scored = attractiveness_from_factor_profiles(profiles, bucket=bucket, s3_client=s3_client)
    return {
        ticker: float(v["attractiveness_score"])
        for ticker, v in scored.items()
        if v.get("attractiveness_score") is not None
    }


def tech_scores_from_eval_log(eval_log: list[dict] | None) -> dict[str, float]:
    """``{ticker: tech_score}`` from ``candidates.json::scanner_eval_log``.

    The eval log is the AUTHORITATIVE per-ticker scanner verdict (see
    ``data.scanner_orchestrator.build_scanner_eval_rows_for_board``) and already
    carries ``tech_score`` for every scanned name, so the incumbent's ranking
    needs no new plumbing through the orchestrator — only this projection.

    Rows without a numeric ``tech_score`` are skipped rather than coerced: a
    missing score must not rank as the worst possible score.
    """
    out: dict[str, float] = {}
    for row in eval_log or []:
        ticker = row.get("ticker")
        score = row.get("tech_score")
        if ticker and isinstance(score, (int, float)):
            out[str(ticker).upper()] = float(score)
    return out


def scanned_universe_from_eval_log(eval_log: list[dict] | None) -> list[str]:
    """Every ticker the scanner EVALUATED this run, from the eval log.

    This is the universe board's population by construction:
    ``data.scanner_orchestrator.write_universe_board_for_scanner_run`` builds
    the board's rows from this same list (``build_scanner_eval_rows_for_board``),
    and ``scoring.universe_board.build_universe_board`` emits one row per row it
    is given. Verified equal on the live 2026-08-20 artifacts (903 = 903, same
    set). Projecting it here is therefore how this producer checks itself
    against the board without reading a fail-soft artifact
    (alpha-engine-config-I7844).

    Distinct from every other eval-log projection in this module: the others
    select rows by a SCORE or a gate, this one is the population itself, so a
    row with no ``tech_score`` and a row that failed every gate both count.
    """
    out: list[str] = []
    seen: set[str] = set()
    for row in eval_log or []:
        ticker = row.get("ticker")
        if not ticker:
            continue
        t = str(ticker).upper()
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def momentum_path_tech_scores(eval_log: list[dict] | None) -> dict[str, float]:
    """``{ticker: tech_score}`` restricted to rows the MOMENTUM PATH admitted.

    ``scan_path == "momentum"`` is set by ``data.scanner.run_quant_filter``
    only for rows that cleared the liquidity floor, the price floor,
    ``tech_score_min``, the MA200 floor and the momentum-path volatility
    ceiling. Ranking this set by ``tech_score`` reproduces the pre-2026-07-22
    live cut exactly, which is what makes ``tech_score_top_N`` a baseline
    rather than an approximation of one.

    Distinct from :func:`tech_scores_from_eval_log`, which projects EVERY
    scored row: that one exists to rank names already inside the champion's
    cut, where the gates are a given. Ranking the universe needs the gates
    applied, and using the wider projection here would admit names the
    incumbent rule would have rejected outright.
    """
    out: dict[str, float] = {}
    for row in eval_log or []:
        if row.get("scan_path") != "momentum":
            continue
        ticker = row.get("ticker")
        score = row.get("tech_score")
        if ticker and isinstance(score, (int, float)):
            out[str(ticker).upper()] = float(score)
    return out


def champion_momentum_weight(bucket: str | None = None, s3_client: Any = None) -> float:
    """The momentum pillar's weight in the LIVE champion composite.

    Read from the same chokepoint the cut producer uses
    (``universe_board._load_pillar_weights`` → the optional private
    ``config/factor_attractiveness_weights.json``, equal-weight when absent),
    so this can never disagree with the weights the champion cut was actually
    ranked by. Returns 0.0 on any read failure — the conservative direction,
    because a zero reading only suppresses observe-only arms.
    """
    from scoring.universe_board import _load_pillar_weights

    try:
        return float(_load_pillar_weights(bucket, s3_client).get("momentum", 0.0))
    except Exception as exc:  # noqa: BLE001 — gating an observe-only arm
        logger.info(
            "[universe_membership] champion momentum weight unreadable (%s) — "
            "treating as 0 and suppressing the momentum arms",
            exc,
        )
        return 0.0


def momentum_arms_applicable(bucket: str | None = None, s3_client: Any = None) -> bool:
    """Whether either momentum challenger arm can express a real experiment.

    **Both arms vary something that the champion's momentum weight multiplies.**
    ``mom121`` changes the momentum pillar's COMPONENTS; ``momzero`` changes its
    WEIGHT to zero. When the live champion already weights momentum at 0:

      * ``momzero`` is the champion, by definition — same profiles, same weights;
      * ``mom121`` is *also* the champion, because a re-composed pillar
        multiplied by zero contributes exactly what the old pillar did.

    Both arms would then emit membership identical to ``attractiveness_top_N``
    and trip the §4 vacuity guard in :func:`assert_cut_invariants`, turning a
    correct configuration into a RED Scanner run — and membership is
    load-bearing for the predictor's universe.

    That guard is right to fire on an identical cut *when momentum carries
    weight*, because identity then means the override silently did not apply.
    It is wrong when the weight is zero, because identity is the arithmetically
    correct answer. So the applicability test belongs here, upstream of the
    guard, rather than as an exception inside it.

    Brian ruling 2026-08-17: the live champion weights momentum at 0 so the
    top-60 captures ~1-year candidates rather than short-term movers
    (``config/factor_attractiveness_weights.json``). Under that ruling both arms
    are inapplicable, not failed — they are omitted and the leaderboard records
    no arm, rather than recording a tie against the champion. Restore a non-zero
    momentum weight and both arms resume automatically with no code change.
    """
    return champion_momentum_weight(bucket, s3_client) > 1e-9


def momzero_attractiveness_for_run(
    run_date: str,
    *,
    scanned_universe: list[str] | None = None,
    bucket: str | None = None,
    s3_client: Any = None,
) -> dict[str, float] | None:
    """``{ticker: attractiveness_score}`` with the momentum pillar weighted 0.

    ``scanned_universe`` is threaded through for the same reason the champion
    gets it, and it is load-bearing for the COMPARISON rather than only for
    correctness: this arm and the champion must rank the same population, or
    the leaderboard attributes a population difference to the weight vector
    (champion-challenger-policy.md §4).

    Reads the CHAMPION's factor profiles — not a shadow copy — because this arm
    changes only the weight vector, not the pillar scores. Sharing the input
    artifact is what guarantees the two arms cannot differ for any other
    reason; recomputing profiles here would reintroduce a data-vintage
    difference the experiment is trying to exclude.

    Fail-soft: an observe-only arm must never red a Scanner run. Returns None
    (not an empty dict, not a champion fallback) so the cuts are omitted and
    recorded as a miss for this arm alone.
    """
    from scoring.universe_board import (
        _read_factor_profiles,
        attractiveness_from_factor_profiles,
    )

    if not momentum_arms_applicable(bucket, s3_client):
        logger.info(
            "[universe_membership] momzero arm inapplicable — the live champion "
            "already weights momentum at 0, so this arm IS the champion; cuts "
            "omitted (not a miss, not a tie)"
        )
        return None

    try:
        profiles = _read_factor_profiles(run_date, bucket, s3_client) or {}
        if not profiles:
            return None
        profiles = _restrict_to_scanned_universe(profiles, scanned_universe, arm=f"momzero arm {run_date}")
        scored = attractiveness_from_factor_profiles(profiles, pillar_weights=MOMZERO_PILLAR_WEIGHTS)
    except Exception as exc:  # noqa: BLE001 — observe-only arm
        logger.info(
            "[universe_membership] momzero arm not computed (%s) — cuts omitted, "
            "recorded as a miss for that arm (I7573)",
            exc,
        )
        return None
    return {
        ticker: rec["attractiveness_score"]
        for ticker, rec in scored.items()
        if isinstance(rec, dict) and rec.get("attractiveness_score") is not None
    }


def challenger_attractiveness_for_run(
    run_date: str,
    *,
    scanned_universe: list[str] | None = None,
    bucket: str | None = None,
    s3_client: Any = None,
) -> dict[str, float] | None:
    """``{ticker: attractiveness_score}`` under the mom121 challenger composite,
    or ``None`` when the shadow profiles are unreadable.

    Fail-soft, unlike :func:`attractiveness_for_run`. That one is load-bearing —
    without it the predictor has no universe. This one feeds an OBSERVE-only
    arm, so its absence must never red a Scanner run whose champion output is
    fine. Returning ``None`` (rather than an empty dict or a champion fallback)
    is what makes `build_universe_membership` omit the arm's cuts entirely,
    which the leaderboard records as a miss for that arm
    (champion-challenger-policy.md §3) instead of scoring the champion twice.
    """
    import boto3

    from scoring.factor_scoring import CHALLENGER_PROFILE_PREFIX
    from scoring.universe_board import attractiveness_from_factor_profiles

    if not momentum_arms_applicable(bucket, s3_client):
        logger.info(
            "[universe_membership] mom121 arm inapplicable — the live champion "
            "weights momentum at 0, so a re-composed momentum pillar contributes "
            "nothing and this arm resolves to the champion; cuts omitted"
        )
        return None

    key = f"{CHALLENGER_PROFILE_PREFIX}/{run_date}/by_ticker.json"
    try:
        s3 = s3_client or boto3.client("s3")
        obj = s3.get_object(Bucket=_bucket(bucket), Key=key)
        profiles = json.loads(obj["Body"].read())
    except Exception as exc:  # noqa: BLE001 — observe-only arm
        logger.info(
            "[universe_membership] mom121 challenger profiles unreadable at %s (%s) "
            "— challenger cut omitted, recorded as a miss for that arm",
            key,
            exc,
        )
        return None
    if not profiles:
        return None
    # Same population as the champion — an arm compared over a different set of
    # names answers a question about the population, not about the horizon.
    profiles = _restrict_to_scanned_universe(profiles, scanned_universe, arm=f"mom121 arm {run_date}")
    scored = attractiveness_from_factor_profiles(profiles)
    return {
        ticker: rec["attractiveness_score"]
        for ticker, rec in scored.items()
        if isinstance(rec, dict) and rec.get("attractiveness_score") is not None
    }


def compute_and_write_universe_membership(
    run_date: str,
    scanner_tickers: list[str],
    *,
    scanner_eval_log: list[dict] | None = None,
    bucket: str | None = None,
    s3_client: Any = None,
) -> str:
    """Scanner entry point — build from the run's candidate cut + factor profiles
    and write. Returns the dated S3 key.

    ``scanner_eval_log`` is the run's ``candidates.json::scanner_eval_log``; when
    passed, the incumbent ``scanner_top_20`` arm and ``scanner_ranks`` are
    emitted and count-matched to the champion. An omitted log degrades to "no
    incumbent arm recorded"; a present-but-too-thin log fails the Scanner run
    rather than writing a breadth-confounded comparison.

    Raises ``UniverseMembershipError`` on any empty input. The caller must NOT
    wrap this in a fail-soft except: a missing membership artifact leaves the
    predictor resolving a stale universe, which is the defect this artifact
    exists to prevent.
    """
    cadence = cut_refresh_cadence()
    # Read unconditionally (alpha-engine-config-I6785). It was previously read
    # only under ``weekly``, because carry-forward was the only consumer; the
    # ``turnover`` block needs it on every cadence, and under ``daily`` it is
    # the ONLY thing that records that the cut moved. ``read_latest_membership``
    # returns None on genuine absence and raises on anything else, so this does
    # not weaken the weekly path's read guarantee.
    prior = read_latest_membership(bucket=bucket, s3_client=s3_client)

    # The scanned universe — the board's population, one hop earlier. Every arm
    # is ranked over it, so no arm can differ from another by which names it
    # scored (alpha-engine-config-I7844).
    scanned_universe = scanned_universe_from_eval_log(scanner_eval_log)

    membership = build_universe_membership(
        run_date,
        scanner_tickers,
        attractiveness_for_run(run_date, scanned_universe=scanned_universe, bucket=bucket, s3_client=s3_client),
        tech_scores=tech_scores_from_eval_log(scanner_eval_log),
        gate_eligible_tech_scores=momentum_path_tech_scores(scanner_eval_log),
        challenger_attractiveness=challenger_attractiveness_for_run(
            run_date, scanned_universe=scanned_universe, bucket=bucket, s3_client=s3_client
        ),
        momzero_attractiveness=momzero_attractiveness_for_run(
            run_date, scanned_universe=scanned_universe, bucket=bucket, s3_client=s3_client
        ),
        scanned_universe=scanned_universe,
        prior=prior,
    )
    membership["cut_refresh_cadence"] = cadence

    if not should_recut(run_date, prior, cadence):
        membership = carry_forward_cuts(membership, prior)
        # The guards are re-run against the HELD cuts, not only against freshly
        # derived ones: a carried-forward artifact is what the predictor
        # consumes for the rest of the week, so a count-match or funnel
        # violation in it is exactly as load-bearing.
        assert_cut_invariants(membership, run_date)
        logger.info(
            "[universe_membership] %s cadence — carrying the cut from %s (ranks refreshed for %s)",
            cadence,
            membership["cut_effective_date"],
            run_date,
        )
    else:
        logger.info("[universe_membership] %s cadence — re-cut for %s", cadence, run_date)

    # Recomputed against the cuts as WRITTEN, after any carry-forward. Under
    # ``weekly`` the held branch replaces the freshly-derived cuts wholesale, so
    # a turnover block computed before that branch would describe a cut this
    # artifact does not contain. On a carried cut the honest answer is 100%
    # retention with zero added — "the cut was held" is a real reading, and one
    # a monitor should be able to see without inferring it from the cadence.
    membership["turnover"] = compute_turnover(membership, prior)

    return write_universe_membership_to_s3(membership, run_date, bucket=bucket, s3_client=s3_client)
