"""Shadow producers for the FILLING champion arms (alpha-engine-config-I9307).

A *filling* arm is one that does not itself write ``signals.json`` — it
synthesizes its ENTER picks **downstream, inside the executor**
(``crucible-executor/executor/champion.py::apply_champion_selection``, hence
``registry.FILLING_CHAMPION_ARMS``). Two arms are in that shape today:
``scanner_predictor_direct`` and ``scanner_top20_predictor``.

WHY THIS MODULE EXISTS
----------------------
The live ``signals/{date}/signals.json`` producer is ``signals_envelope``,
which is **empty-by-contract** (``registry.EMPTY_BUY_CANDIDATES_BY_CONTRACT_PRODUCERS``):
it never proposes entries and its ``buy_candidates`` is always ``[]``. The
producer leaderboard nevertheless scored the champion arm by reading ENTER
picks out of that artifact, so from the moment the envelope became the live
producer (epic config-I2515 Phase B) the champion contributed **zero** cohort
dates — measured 2026-08-29: 903 tickers, every one ``HOLD``, on every date
from 2026-07-18 onward. Seven weeks of a dead arm rendering as a *thin* one.

The fix is not to make the envelope emit ENTER (it is empty-by-contract on
purpose and ``assert_producer_champion_coherence`` depends on that — config#5713).
The fix is that a filling arm writes its picks to the SAME comparable artifact
every other arm writes: ``signals_shadow/{arm}/{date}/signals.json``, on the
same weekly pass, from the same durable inputs, so champion and challengers are
scored **on the same basis, over the same cohort, from the same source**
(champion-challenger-policy.md §4).

UNCONDITIONAL BY CONSTRUCTION (§3)
----------------------------------
These builders run for BOTH filling arms on every weekly pass, regardless of
which arm the pointer currently serves. That is the load-bearing reason the
shadow is written here and NOT captured inside the executor at the point of
synthesis: the executor only ever synthesizes for the arm that is *currently
champion*, so an executor-side capture would keep exactly one arm measurable
and would go dark on the incumbent the moment the pointer moved — the same
defect, mirrored. §3: *promotion changes which arm's output is consumed; it
changes nothing about what is measured.*

It is also why the shadow is taken at the SELECTION RULE, not at the executor's
final plan: the plan is post-universe-filter, post-coverage-filter and
post-optimizer, so scoring it would measure the whole executor stack rather
than the selection rule under test (§4: hold everything constant except the
thing being tested).

ONE WRITER, NOT A PER-ARM BRANCH
--------------------------------
Both arms share :func:`build_filling_shadow`; they differ only in a *pool
loader*. The per-arm branch is what produced three divergent arm registers
(alpha-engine-config-I9277, -I9299), and this module deliberately does not
repeat it. The runner (``producers/runner.py``) is the single write site for
every challenger shadow, including these — so these arms inherit its FAIL-HARD
shadow-gap detection (config#1683) with no new plumbing.

FIDELITY TO WHAT THE EXECUTOR ACTUALLY SERVES
---------------------------------------------
The ranking rules here reproduce ``crucible-executor``'s handlers exactly:

* ``scanner_predictor_direct`` — the research-free predictor cohort for the
  date, sorted by ``predicted_alpha`` descending, top-N. (The executor also
  cross-sectionally centres the alphas before injecting them; centring is a
  constant shift, so the ORDER and therefore the selected set are
  byte-identical — the executor's own comment records this. The centring is an
  optimizer-anchor concern, not a selection concern, and is deliberately not
  reproduced here.)
* ``scanner_top20_predictor`` — the members of the cut named by
  ``universe_membership/{date}::predictor_universe_cut``, sorted by the
  predictor's own ``predicted_alpha`` descending, top-N.

Both then map rank onto a score band with the SAME monotone transform the
executor uses (:func:`rank_to_score`). Because the transform is monotone the
score band cannot change the ranking, only its rendering.

**This is a second implementation of a rule crucible-executor also holds, and
that is a tracked debt, not an accident.** The rule is three lines and the
executor is trading-critical, so lifting it into ``nousergon_lib`` and having
the executor adopt it is a SERVING-path change, deliberately not made in the
same change as this measurement-only fix. ``tests/test_filling_arm_shadow.py``
pins the rule against a recorded live cohort so a divergence is caught here;
the lift is tracked separately.

NOT A LIVE-TRADING PATH. Everything written by this module goes to
``signals_shadow/``, which the executor and predictor never read.
"""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any

logger = logging.getLogger(__name__)

# The durable inputs, mirrored from crucible-executor/executor/champion.py.
# Kept as literals rather than a cross-repo import (the same convention the
# executor uses for CHALLENGER_SELECTION_LATEST_KEY): these are stable S3
# contracts, not shared code.
from data.fetchers.price_fetcher import fetch_sp500_sp400_with_sectors

RESEARCH_FREE_PARQUET_KEY = "predictor/research_free_backfill/predictor_outcomes_research_free.parquet"
MEMBERSHIP_KEY = "universe_membership/{date}/membership.json"
PREDICTIONS_KEY = "predictor/predictions/{date}.json"

# The width each filling arm actually serves. The executor uses
# ``n = len(buy_candidates) or champion_top_n_default``; because the live
# producer is empty-by-contract, ``buy_candidates`` is ALWAYS empty, so the
# served width is always the default. Reproducing the served width (rather than
# the leaderboard's ``top_n``) is what makes the shadow mean what the arm does.
CHAMPION_TOP_N_DEFAULT = 10
CHAMPION_SCORE_FLOOR = 60.0
CHAMPION_SCORE_CEILING = 95.0

ARCHITECTURE_VERSION = "filling_arm_shadow_v1"


class FillingShadowError(RuntimeError):
    """A filling arm's shadow could not be built from its declared inputs.

    RAISES rather than emitting an empty cohort. An arm that writes a
    well-formed shadow containing no picks is indistinguishable, on the
    leaderboard, from an arm that legitimately selected nothing — which is the
    exact conflation champion-challenger-policy.md §3 forbids and the exact
    shape of the defect this module exists to close.
    """


def rank_to_score(rank_fraction: float, floor: float, ceiling: float) -> float:
    """Map a within-pool rank fraction in [0, 1] (0 = best) onto
    ``[floor, ceiling]``, best rank → ceiling, worst rank → floor.

    Byte-identical to ``crucible-executor/executor/champion.py::_rank_to_score``.
    Monotone, so it fixes the RENDERING of the arm's ranking and can never
    change the ranking itself.
    """
    if ceiling <= floor:
        raise ValueError(f"score ceiling ({ceiling}) must exceed score floor ({floor})")
    rank_fraction = min(max(rank_fraction, 0.0), 1.0)
    return ceiling - rank_fraction * (ceiling - floor)


def rank_by_alpha(rows: Sequence[tuple[str, float]]) -> list[tuple[str, float]]:
    """``(ticker, alpha)`` sorted by alpha DESCENDING, ties broken by ticker.

    The tie-break is explicit so the shadow is reproducible from the same
    inputs on any machine — an unstable order would make the recorded track
    record unverifiable, which is the only asset this loop has (§3.1).
    """
    return sorted(rows, key=lambda r: (-r[1], r[0]))


def build_shadow_payload(
    arm: str,
    run_date: str,
    ranked: Sequence[tuple[str, float]],
    *,
    pool_size: int,
    pool_source: str,
    top_n: int = CHAMPION_TOP_N_DEFAULT,
    score_floor: float = CHAMPION_SCORE_FLOOR,
    score_ceiling: float = CHAMPION_SCORE_CEILING,
    sector_map: Mapping[str, str] | None = None,
) -> dict:
    """The conforming ``signals_shadow`` document for one filling arm/date.

    Shape is the one ``scoring/leaderboard_producers._enter_ranked_and_scores``
    already reads and ``no_agent_quant`` / ``single_agent_quant`` already write:
    a top-level ``signals`` dict keyed by ticker, each entry carrying
    ``signal == "ENTER"`` and a numeric ``score``. Schema:
    ``contracts/arm_shadow_signals.schema.json``.

    ``sector_map`` (``{ticker: sector}``, the constituents artifact via
    ``fetch_sp500_sp400_with_sectors`` — the SAME source ``no_agent`` and
    ``single_agent`` resolve from) puts a ``sector`` on every row, exactly as
    those producers do (``sector_map.get(ticker, "Unknown")``). Measured
    2026-09-05 (weekly SF watch-rerun-2026-09-04-1/2/3): rows written without
    it were refused by ``scoring.promotion_guards.assert_promotable``'s
    UNRESOLVED-SECTOR rule on every run — the executor sizes against sector
    caps, so an actionable row with no sector is unservable — and a refused
    ChallengerShadow degrades the whole weekly run. The guard stays the
    refusal point; this module only supplies what the guard reads.

    RAISES on an empty ranking — see :class:`FillingShadowError`.
    """
    sectors = sector_map or {}
    if not ranked:
        raise FillingShadowError(
            f"{arm}: pool {pool_source!r} for {run_date} yielded ZERO ranked names "
            "— refusing to write a well-formed shadow carrying no picks, which "
            "would render on the leaderboard as an arm that legitimately "
            "selected nothing (champion-challenger-policy.md §3)."
        )

    top = list(ranked[:top_n])
    signals: dict[str, dict] = {}
    for rank, (ticker, alpha) in enumerate(top):
        rank_fraction = rank / max(pool_size - 1, 1)
        signals[ticker] = {
            "ticker": ticker,
            "signal": "ENTER",
            "score": rank_to_score(rank_fraction, score_floor, score_ceiling),
            "rating": "BUY",
            "conviction": "medium",
            "sector": sectors.get(ticker, "Unknown"),
            "predicted_alpha": alpha,
            # §7.5 — the artifact names the arm that produced it rather than
            # carrying a literal that goes stale when the pointer moves.
            "champion_arm": arm,
        }

    return {
        "schema_version": 1,
        "date": run_date,
        "producer": arm,
        "architecture_version": ARCHITECTURE_VERSION,
        "market_regime": "neutral",
        "sector_ratings": {},
        "sector_modifiers": {},
        "signals": signals,
        "population": [t for t, _ in ranked],
        "universe": [],
        # The arm's picks, in the same list shape the challengers emit.
        "buy_candidates": [dict(v) for v in signals.values()],
        "quarantined": [],
        # Provenance for the comparison: which pool, how wide, and how many of
        # it were taken. An arm scored on a short pool must not read as an arm
        # that chose badly.
        "arm_pool": {
            "pool_source": pool_source,
            "pool_size": pool_size,
            "n_ranked": len(ranked),
            "n_selected": len(top),
            "top_n": top_n,
        },
    }


# ── Pool loaders — one per arm, the ONLY thing the two arms differ by ────────


def _s3(archive_manager) -> tuple[Any, str]:
    """``(client, bucket)`` from the archive manager, which already owns both."""
    client = getattr(archive_manager, "s3", None) or getattr(archive_manager, "s3_client", None)
    bucket = getattr(archive_manager, "bucket", None) or getattr(archive_manager, "bucket_name", None)
    if client is None or not bucket:
        raise FillingShadowError(
            "archive manager exposes no usable (s3 client, bucket) pair — "
            f"got client={client!r} bucket={bucket!r}"
        )
    return client, bucket


def _get_json(s3: Any, bucket: str, key: str) -> dict | None:
    from botocore.exceptions import ClientError

    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return None
        raise
    return json.loads(resp["Body"].read())


def load_research_free_pool(
    s3: Any, bucket: str, run_date: str,
) -> tuple[list[tuple[str, float]], str]:
    """``scanner_predictor_direct``'s pool for ``run_date``.

    The research-free parquet is a HISTORY, not a latest-cohort snapshot —
    measured 2026-08-29: 2080 rows across 35 distinct ``prediction_date``
    values, covering every recent weekly cohort date. That is what makes this
    arm's shadow BACKFILLABLE, and it is why the champion's cohort does not
    have to be rebuilt from zero over the next 21 sessions.

    RAISES when the date is absent — a filling arm silently skipping a cohort
    date is the miss/broken conflation §3 forbids.
    """
    import pandas as pd
    from botocore.exceptions import ClientError

    try:
        body = s3.get_object(Bucket=bucket, Key=RESEARCH_FREE_PARQUET_KEY)["Body"].read()
    except ClientError as e:
        raise FillingShadowError(
            f"research-free parquet s3://{bucket}/{RESEARCH_FREE_PARQUET_KEY} "
            f"unreadable: {e}"
        ) from e

    df = pd.read_parquet(io.BytesIO(body))
    required = {"ticker", "prediction_date", "predicted_alpha"}
    missing = required - set(df.columns)
    if missing:
        raise FillingShadowError(
            f"research-free parquet missing column(s) {sorted(missing)} — got {sorted(df.columns)}"
        )

    cohort = df[df["prediction_date"].astype(str) == run_date]
    if cohort.empty:
        available = sorted({str(d) for d in df["prediction_date"].unique()})
        raise FillingShadowError(
            f"research-free parquet carries no prediction_date == {run_date}; "
            f"it holds {len(available)} date(s), latest {available[-1] if available else 'none'}"
        )

    rows = [
        (str(t), float(a))
        for t, a in zip(cohort["ticker"], cohort["predicted_alpha"], strict=True)
        if a == a  # NaN-safe
    ]
    return rank_by_alpha(rows), "research_free_parquet"


def load_predictor_cut_pool(
    s3: Any, bucket: str, run_date: str,
) -> tuple[list[tuple[str, float]], str]:
    """``scanner_top20_predictor``'s pool for ``run_date``.

    The members of whichever cut ``universe_membership/{date}`` names in its own
    ``predictor_universe_cut`` field, scored by the predictor's own
    ``predicted_alpha`` from ``predictor/predictions/{date}.json``.

    The cut NAME is never hardcoded (§7.5): when crucible-research moves
    ``PREDICTOR_UNIVERSE_CUT`` this arm follows it with no edit here, exactly as
    the executor's ``_resolve_predictor_cut_pool`` does.
    """
    membership = _get_json(s3, bucket, MEMBERSHIP_KEY.format(date=run_date))
    if not membership:
        raise FillingShadowError(
            f"no universe_membership for {run_date} — cannot resolve this arm's pool"
        )
    cut_name = membership.get("predictor_universe_cut")
    if not cut_name:
        raise FillingShadowError(
            f"universe_membership/{run_date} names no predictor_universe_cut"
        )
    cut = (membership.get("cuts") or {}).get(cut_name) or {}
    pool = [str(t) for t in (cut.get("tickers") or [])]
    if not pool:
        raise FillingShadowError(
            f"cut {cut_name!r} in universe_membership/{run_date} carries no tickers"
        )

    preds_doc = _get_json(s3, bucket, PREDICTIONS_KEY.format(date=run_date))
    if not preds_doc:
        raise FillingShadowError(
            f"no predictor/predictions/{run_date}.json — this arm's ranking IS "
            "the predictor's output, so there is nothing to rank without it"
        )
    preds = _predictions_by_ticker(preds_doc)

    rows: list[tuple[str, float]] = []
    for ticker in pool:
        alpha = preds.get(ticker)
        if alpha is None:
            continue
        rows.append((ticker, float(alpha)))

    if not rows:
        raise FillingShadowError(
            f"none of the {len(pool)} name(s) in cut {cut_name!r} carries a usable "
            f"predicted_alpha in predictor/predictions/{run_date}.json — refusing "
            "to synthesize zero candidates while reporting a healthy arm"
        )
    if len(rows) < len(pool):
        logger.warning(
            "[filling_arms] scanner_top20_predictor %s: %d of %d cut member(s) "
            "carry no usable predicted_alpha — SHORT pool, recorded on arm_pool",
            run_date, len(pool) - len(rows), len(pool),
        )
    return rank_by_alpha(rows), f"predictor_cut:{cut_name}"


def _predictions_by_ticker(doc: Any) -> dict[str, float]:
    """``{ticker: predicted_alpha}`` from the predictions artifact.

    Tolerates the two live shapes (a top-level list of rows, or a dict keyed by
    ticker) because both appear in the historical series; anything else RAISES
    rather than yielding an empty map that would read as "the predictor had no
    opinion".
    """
    rows: Sequence[Mapping[str, Any]]
    if isinstance(doc, list):
        rows = doc
    elif isinstance(doc, Mapping):
        for field in ("predictions", "rows", "records"):
            val = doc.get(field)
            if isinstance(val, list):
                rows = val
                break
            if isinstance(val, Mapping):
                return {
                    str(t): float(v["predicted_alpha"])
                    for t, v in val.items()
                    if isinstance(v, Mapping) and v.get("predicted_alpha") is not None
                }
        else:
            if all(isinstance(v, Mapping) for v in doc.values()):
                return {
                    str(t): float(v["predicted_alpha"])
                    for t, v in doc.items()
                    if isinstance(v, Mapping) and v.get("predicted_alpha") is not None
                }
            raise FillingShadowError(
                f"unrecognised predictions document shape: keys={sorted(doc)[:10]}"
            )
    else:
        raise FillingShadowError(f"unrecognised predictions document type {type(doc).__name__}")

    return {
        str(r["ticker"]): float(r["predicted_alpha"])
        for r in rows
        if isinstance(r, Mapping) and r.get("ticker") and r.get("predicted_alpha") is not None
    }


# ── The ONE builder, parameterised by pool loader ─────────────────────────────

POOL_LOADERS = {
    "scanner_predictor_direct": load_research_free_pool,
    "scanner_top20_predictor": load_predictor_cut_pool,
}


def build_filling_shadow(arm: str, run_date: str, archive_manager, **_ctx) -> dict:
    """Build one filling arm's shadow payload for ``run_date``.

    Matches the ``ProducerSpec.build`` signature
    (``(run_date, archive_manager, **ctx) -> payload``) once bound to an arm,
    so ``producers.runner`` writes it through the SAME
    ``write_shadow_signals_json`` path and the SAME FAIL-HARD gap detection as
    every other challenger. There is no second writer.
    """
    loader = POOL_LOADERS.get(arm)
    if loader is None:
        raise FillingShadowError(
            f"no pool loader registered for filling arm {arm!r} — known: {sorted(POOL_LOADERS)}"
        )
    s3, bucket = _s3(archive_manager)
    ranked, pool_source = loader(s3, bucket, run_date)
    # Sector from the constituents artifact (S3, hard-fails on a read error),
    # the same call no_agent/single_agent make; a runner that already holds
    # the map may pass it through ctx instead of re-reading it.
    sector_map = _ctx.get("sector_map")
    if not isinstance(sector_map, Mapping):
        _, sector_map = fetch_sp500_sp400_with_sectors()
    payload = build_shadow_payload(
        arm, run_date, ranked, pool_size=len(ranked), pool_source=pool_source,
        sector_map=sector_map,
    )
    logger.info(
        "[filling_arms] %s %s: %d selected from pool=%s (%d ranked)",
        arm, run_date, len(payload["signals"]), pool_source, len(ranked),
    )
    return payload


def run_scanner_predictor_direct_producer(run_date: str, archive_manager, **ctx) -> dict:
    """``ProducerSpec.build`` for ``scanner_predictor_direct``."""
    return build_filling_shadow("scanner_predictor_direct", run_date, archive_manager, **ctx)


def run_scanner_top20_predictor_producer(run_date: str, archive_manager, **ctx) -> dict:
    """``ProducerSpec.build`` for ``scanner_top20_predictor``."""
    return build_filling_shadow("scanner_top20_predictor", run_date, archive_manager, **ctx)
