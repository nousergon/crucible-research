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

  ``scanner_gate_baseline_60``
                             FEEDS NOTHING LIVE. A recorded baseline: the
                             scanner's own gate cut, ``candidates.json::
                             scanner_tickers`` verbatim, basis=``scanner_gate``.
                             A ``tech_score`` momentum ranking, NOT an alpha
                             ranking — near-disjoint from the attractiveness
                             rank the champion is drawn from (12 of 60 overlap,
                             measured 2026-08-17). Retained as the incumbent
                             challenger arm and the churn baseline.
                             Was ``scanner_candidates``; that key is still
                             emitted as a deprecated alias
                             (alpha-engine-config-I7578).
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
    "universe_count": int,             # names with a rankable attractiveness score
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
    "scanner_ranks": {                  # SCANNER CUT only, rank 1 = highest
      "AAPL": {"tech_score_rank": int, "tech_score": float},   # tech_score.
      ...                               # Absent when no eval log was supplied.
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
of the ranking method.
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
# Two 60-wide cuts exist and only one is the funnel — and until this rename, the
# one that is NOT the funnel was the one whose name said "scanner". It was read
# backwards three times, each caught by hand rather than by a check. Measured
# 2026-08-17: the two 60s overlap on 12 of 60, and the champion 20 overlaps the
# gate cut on 3 of 20. They are near-disjoint sets and only one advances.
#
# The funnel INVARIANT was already enforced (I6630, ``assert_cut_invariants``).
# What was unguarded is the NAMING: nothing stopped a reader, an agent, or a new
# consumer from resolving "the scanner's top 60" to the gate cut and scoping to
# a set that feeds nothing. The word "baseline" is load-bearing — reading this
# name as the funnel now requires ignoring it.
GATE_BASELINE_CUT = "scanner_gate_baseline_60"
GATE_LEGACY_CUT = "scanner_candidates"
"""Deprecated alias, emitted alongside the new name for one window."""

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
    "quality", "value", "momentum", "growth", "stewardship", "defensiveness",
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


def _rank_table(attractiveness: dict[str, float]) -> dict[str, dict]:
    """``{ticker: {attractiveness_rank, attractiveness_score}}``, rank 1 = most
    attractive. Ties broken by ticker so the table is deterministic across runs
    (a reproducible rank matters — consumers diff these week over week)."""
    ordered = sorted(attractiveness.items(), key=lambda kv: (-kv[1], kv[0]))
    return {
        ticker: {"attractiveness_rank": i + 1, "attractiveness_score": score}
        for i, (ticker, score) in enumerate(ordered)
    }


def _top_n(ranks: dict[str, dict], n: int) -> list[str]:
    """The N most attractive tickers, returned SORTED (set semantics — this is a
    membership list, and rank order is already recoverable from ``ranks``)."""
    top = sorted(ranks.items(), key=lambda kv: kv[1]["attractiveness_rank"])[:n]
    return sorted(t for t, _ in top)


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
    gate_names = {GATE_BASELINE_CUT, GATE_LEGACY_CUT}
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
    for prefix, n in [
        (p, n) for p in (CHALLENGER_CUT_PREFIX, MOMZERO_CUT_PREFIX) for n in _CHALLENGER_CUT_NS
    ]:
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


def build_universe_membership(
    run_date: str,
    scanner_tickers: list[str],
    attractiveness: dict[str, float],
    *,
    tech_scores: dict[str, float] | None = None,
    challenger_attractiveness: dict[str, float] | None = None,
    momzero_attractiveness: dict[str, float] | None = None,
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
        "basis": "scanner_gate",
        "size": len(set(scanner_tickers)),
        "tickers": sorted(set(scanner_tickers)),
        "source": f"candidates/{run_date}/candidates.json::scanner_tickers",
        "feeds": [],
        "role": (
            "recorded baseline and incumbent challenger arm. Feeds NOTHING "
            "live — not the predictor universe, not the RAG corpus scope, not "
            "Think Tank's coverage window. See the funnel block."
        ),
    }
    cuts: dict[str, dict] = {
        GATE_BASELINE_CUT: dict(_gate_block),
        # Deprecated alias (alpha-engine-config-I7578). Emitted so a consumer
        # pinned on the old name keeps working through the deprecation window
        # rather than reading a missing key as an empty cut — silently trading
        # a rename for a zero-size universe is the worse failure. Known live
        # reader: crucible-dashboard/loaders/universe_churn.py.
        GATE_LEGACY_CUT: {
            **_gate_block,
            "deprecated_alias_for": GATE_BASELINE_CUT,
            "removal_tracked_by": "alpha-engine-config-I7578 follow-up",
        },
    }
    for n in _RANK_CUTS:
        tickers = _top_n(ranks, n)
        cuts[f"attractiveness_top_{n}"] = {
            "basis": "attractiveness_rank",
            "size": len(tickers),
            "tickers": tickers,
            "source": f"scanner/universe/{run_date}/universe.json::attractiveness_score",
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
                "source": (
                    f"scanner/factor_profiles_shadow/mom121/{run_date}/"
                    "profiles.json::attractiveness_score"
                ),
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
                "source": (
                    f"factors/profiles/{run_date}/by_ticker.json"
                    "::attractiveness_score@momentum_weight=0"
                ),
            }

    assert_cut_invariants({"cuts": cuts}, run_date)

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
            "feed_cut": {"name": FEED_CUT_NAME,
                         "size": len(cuts.get(FEED_CUT_NAME, {}).get("tickers", []))},
            "champion_cut": {"name": PREDICTOR_UNIVERSE_CUT,
                             "size": len(cuts.get(PREDICTOR_UNIVERSE_CUT, {}).get("tickers", []))},
            "advances_to": {
                "predictor_universe": PREDICTOR_UNIVERSE_CUT,
                "rag_corpus_scope": FEED_CUT_NAME,
                "thinktank_coverage_window": FEED_CUT_NAME,
            },
            "feeds_nothing_live": [GATE_BASELINE_CUT, GATE_LEGACY_CUT],
        },
        # alpha-engine-config-I6666 — a freshly built artifact is by definition
        # re-cut today; ``carry_forward_cuts`` overwrites this when the run
        # holds the prior cut instead. ``cut_effective_date == run_date`` is
        # therefore the test for "this cut is today's".
        "cut_effective_date": run_date,
        "cuts": cuts,
        "ranks": ranks,
    }
    if scanner_ranks:
        membership["scanner_ranks"] = scanner_ranks
    if backfilled_from:
        membership["backfilled_from"] = backfilled_from
    # Runs on the ASSEMBLED artifact, unlike the cut invariants above: what it
    # checks are the routing fields, which do not exist until the dict is built.
    assert_gate_cut_feeds_nothing_live(membership, run_date)
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


def attractiveness_for_run(run_date: str, *, bucket: str | None = None, s3_client: Any = None) -> dict[str, float]:
    """``{ticker: attractiveness_score}`` for ``run_date``, computed from the
    run's factor profiles via the SSOT cross-sectional chokepoint.

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
    it goes through the identical pillar-weighting chokepoint the board uses —
    the ranks here are byte-identical to the console board's.
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


def champion_momentum_weight(
    bucket: str | None = None, s3_client: Any = None
) -> float:
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


def momentum_arms_applicable(
    bucket: str | None = None, s3_client: Any = None
) -> bool:
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
    run_date: str, *, bucket: str | None = None, s3_client: Any = None
) -> dict[str, float] | None:
    """``{ticker: attractiveness_score}`` with the momentum pillar weighted 0.

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
        scored = attractiveness_from_factor_profiles(
            profiles, pillar_weights=MOMZERO_PILLAR_WEIGHTS
        )
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
    run_date: str, *, bucket: str | None = None, s3_client: Any = None
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

    membership = build_universe_membership(
        run_date,
        scanner_tickers,
        attractiveness_for_run(run_date, bucket=bucket, s3_client=s3_client),
        tech_scores=tech_scores_from_eval_log(scanner_eval_log),
        challenger_attractiveness=challenger_attractiveness_for_run(
            run_date, bucket=bucket, s3_client=s3_client
        ),
        momzero_attractiveness=momzero_attractiveness_for_run(
            run_date, bucket=bucket, s3_client=s3_client
        ),
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
