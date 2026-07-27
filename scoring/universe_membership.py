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

  ``scanner_candidates``     the scanner's own gate cut — the tickers in
                             ``candidates.json::scanner_tickers`` verbatim.
                             basis=``scanner_gate``. This is the cut the
                             predictor's universe resolves from.
  ``attractiveness_top_25``  top 25 by cross-sectional attractiveness rank.
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
  ``s3://{bucket}/universe_membership/{run_date}/membership.json``
  ``s3://{bucket}/universe_membership/latest.json`` (sidecar)

Schema::

  {
    "schema_version": 1,
    "producer": "crucible-research/scoring/universe_membership.py",
    "run_date": "YYYY-MM-DD",
    "generated_at": "ISO-8601 UTC",
    "universe_count": int,             # names with a rankable attractiveness score
    "predictor_universe_cut": "scanner_candidates",   # names the cut the predictor resolves from
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
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
PRODUCER = "crucible-research/scoring/universe_membership.py"

# The cut the predictor resolves its daily scoring universe from. Named in the
# artifact (``predictor_universe_cut``) rather than hardcoded on the consumer
# side, so changing which cut drives inference is a producer-side, versioned,
# reviewable decision — not a silent constant edit in another repo.
PREDICTOR_UNIVERSE_CUT = "scanner_candidates"

# Attractiveness-rank cuts emitted every cycle. 60 is count-matched to the
# scanner's ``momentum_top_n`` so scanner-cut-vs-rank-cut is an apples-to-apples
# comparison; 25 matches the size of the retired tracked population so the
# historical series stays interpretable across the producer change.
_RANK_CUTS = (25, 60)

_DEFAULT_BUCKET = "alpha-engine-research"


class UniverseMembershipError(RuntimeError):
    """Raised when the membership artifact cannot be built from real inputs.

    Deliberately a hard error (feedback_no_silent_fails): this artifact is
    load-bearing for the predictor's daily universe, so an empty or partial
    membership is a red run, never a degraded write.
    """


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


def build_universe_membership(
    run_date: str,
    scanner_tickers: list[str],
    attractiveness: dict[str, float],
    *,
    generated_at: str | None = None,
    backfilled_from: str | None = None,
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
    cuts: dict[str, dict] = {
        "scanner_candidates": {
            "basis": "scanner_gate",
            "size": len(set(scanner_tickers)),
            "tickers": sorted(set(scanner_tickers)),
            "source": f"candidates/{run_date}/candidates.json::scanner_tickers",
        }
    }
    for n in _RANK_CUTS:
        tickers = _top_n(ranks, n)
        cuts[f"attractiveness_top_{n}"] = {
            "basis": "attractiveness_rank",
            "size": len(tickers),
            "tickers": tickers,
            "source": f"scanner/universe/{run_date}/universe.json::attractiveness_score",
        }

    membership = {
        "schema_version": SCHEMA_VERSION,
        "producer": PRODUCER,
        "run_date": run_date,
        "generated_at": generated_at or datetime.now(UTC).isoformat(timespec="seconds"),
        "universe_count": len(ranks),
        "predictor_universe_cut": PREDICTOR_UNIVERSE_CUT,
        "cuts": cuts,
        "ranks": ranks,
    }
    if backfilled_from:
        membership["backfilled_from"] = backfilled_from
    return membership


def write_universe_membership_to_s3(
    membership: dict,
    run_date: str,
    *,
    bucket: str | None = None,
    s3_client: Any = None,
) -> str:
    """Write to the dated key + the ``latest.json`` sidecar. Returns the dated key."""
    s3 = _client(s3_client)
    b = _bucket(bucket)
    body = json.dumps(membership, separators=(",", ":"), default=str).encode("utf-8")
    dated_key = f"universe_membership/{run_date}/membership.json"
    for key in (dated_key, "universe_membership/latest.json"):
        s3.put_object(Bucket=b, Key=key, Body=body, ContentType="application/json")
    logger.info(
        "[universe_membership] wrote %d cuts over %d ranked names → s3://%s/%s (+latest)",
        len(membership.get("cuts", {})),
        membership.get("universe_count", 0),
        b,
        dated_key,
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


def compute_and_write_universe_membership(
    run_date: str,
    scanner_tickers: list[str],
    *,
    bucket: str | None = None,
    s3_client: Any = None,
) -> str:
    """Scanner entry point — build from the run's candidate cut + factor profiles
    and write. Returns the dated S3 key.

    Raises ``UniverseMembershipError`` on any empty input. The caller must NOT
    wrap this in a fail-soft except: a missing membership artifact leaves the
    predictor resolving a stale universe, which is the defect this artifact
    exists to prevent.
    """
    membership = build_universe_membership(
        run_date,
        scanner_tickers,
        attractiveness_for_run(run_date, bucket=bucket, s3_client=s3_client),
    )
    return write_universe_membership_to_s3(membership, run_date, bucket=bucket, s3_client=s3_client)
