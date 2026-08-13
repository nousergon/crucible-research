"""
Factor scoring — Phase 1c of the institutional factor substrate (260513 plan),
extended Phase 3b of the attractiveness-pillars-260520 arc with Growth +
Stewardship composites.

Combines raw factor data (already populated in the production feature store
at s3://alpha-engine-research/features/{date}/) into 6 composite factor
scores per ticker, each percentile-ranked WITHIN sector to avoid
cross-sector noise (an OW Tech sector shouldn't lose all its names just
because Tech mean-vol is high vs. Healthcare).

Composite factors (mirroring AQR Style Premia + GS QIS conventions, with
Phase 3b additions covering the Growth + Stewardship pillars from the
attractiveness-pillars-260520 arc):

- ``quality_score``       — Quality (QMJ-style): ROE + (1 - debt/equity) +
                            gross margin + current ratio
- ``momentum_score``      — Cross-sectional momentum: 12-1m + 6m + 1m
                            + distance-from-52w-high
- ``low_vol_score``       — Inverse realized vol: (1 - 20d realized vol
                            z-score) + (1 - vol_ratio_10_60 z-score)
- ``value_score``         — Inverse multiples: (1 - PE) + (1 - PB) + FCF yield
- ``growth_score``        — Sustainable growth (Phase 3b): 3y revenue CAGR +
                            3y EPS CAGR + sustainable growth rate
                            (``roe × (1 - payout_ratio)``, derived) + 5y
                            CAPEX growth. Backed by alpha-engine-data Phase 3a
                            fundamental fields (revenue_growth_3y /
                            eps_growth_3y / payout_ratio / capex_growth_5y).
- ``stewardship_score``   — Capital allocation discipline (Phase 3b, extended
                            config#2428 with an institutional-accumulation
                            component): (1 - payout_ratio) + 5y CAPEX growth
                            + net institutional accumulation (13F QoQ fund
                            count delta, sourced from the ``inst_ownership``
                            derived table via ``data.substrate.reader
                            .read_inst_ownership`` — the same reader
                            ``qual_tools.get_institutional_activity`` uses).
                            Still a thin-quant signal by design — the
                            Stewardship pillar's qualitative side (Qual
                            Analyst's pillar rubric) carries most of the
                            discriminative weight. Insider ownership % is
                            deferred (Finnhub metric=all does not expose it;
                            would need a separate /stock/insider-transactions
                            integration to earn its way into the composite).

All composites returned on a 0-100 within-sector percentile scale so they
compose with the existing 0-100 quant/qual sub-scores in
scoring/composite.py.

Tolerant-reader contract: when Phase 3a fundamental fields are absent
(pre-merge / first SF firing after merge), the underlying
``_within_sector_pct_rank`` returns all-NaN for those columns and the
partial-coverage handling in ``compute_factor_composites`` keeps the
existing 4 composites stable while emitting NaN for the new 2. Downstream
consumers (``score_aggregator``, factor blend, dashboard) treat NaN as
"no data" rather than 0 — same convention as the existing 4 composites.

Produced once per Saturday SF run (and on demand by ad-hoc backtester
runs). Cached to s3://alpha-engine-research/factors/profiles/{date}/by_ticker.json
so downstream consumers (composite scoring extension in Phase 3, quant
@tool in Phase 2, backtester attribution in Phase 5) read from a single
canonical artifact without re-deriving from raw factor parquets.
"""

from __future__ import annotations

import io
import json
import logging
import os
from datetime import date as _date

import pandas as pd

from data.fetchers.feature_store_reader import (
    _FACTOR_TECHNICAL_COLS as _READER_FACTOR_TECHNICAL_COLS,
)

logger = logging.getLogger(__name__)


# ── Composite definitions ───────────────────────────────────────────────────
# Each composite is a weighted sum of within-sector-percentile-ranked raw
# factors. Weights sum to 1.0 per composite. Higher composite score = more
# desirable on that factor axis (e.g. high quality_score = more profitable +
# lower leverage; high low_vol_score = LOWER realized vol).
#
# `invert=True` means the raw factor is INVERSELY desirable (e.g. higher PE
# = less desirable, so we invert the percentile rank before combining).
# config#1039: composite definitions are experiment beliefs
# (HARNESS_EXPERIMENT_CLASSIFICATION.md §1) — the experiment package's
# scoring.yaml `factor_composites:` overrides them; these literals are the
# public BASELINE the repo runs with when no package entry is present.
_BASELINE_COMPOSITE_DEFS: dict[str, list[tuple[str, float, bool]]] = {
    "quality_score": [
        # (raw_factor_column, weight, invert_rank)
        ("roe", 0.30, False),
        ("debt_to_equity", 0.25, True),  # less debt = better
        ("gross_margin", 0.25, False),
        ("current_ratio", 0.20, False),
    ],
    "momentum_score": [
        ("momentum_20d", 0.30, False),
        ("return_60d", 0.25, False),
        ("return_120d", 0.20, False),
        ("dist_from_52w_high", 0.15, False),  # closer to high (less negative) = better
        ("momentum_5d", 0.10, False),
    ],
    "low_vol_score": [
        ("realized_vol_20d", 0.50, True),  # lower vol = higher score
        ("vol_ratio_10_60", 0.30, True),  # vol stable / declining = higher score
        ("atr_14_pct", 0.20, True),
    ],
    "value_score": [
        ("pe_ratio", 0.40, True),
        ("pb_ratio", 0.30, True),
        ("fcf_yield", 0.30, False),
    ],
    # ── Phase 3b of attractiveness-pillars-260520 — Growth + Stewardship
    # composites. Backed by alpha-engine-data Phase 3a fundamental fields
    # plus one derived field (sustainable_growth_rate, computed inline
    # before the percentile-rank step). When the Phase 3a fields are
    # absent (pre-merge / first SF firing after merge), all components
    # rank-NaN and the composite emits NaN — same partial-coverage
    # handling as the existing 4 composites.
    "growth_score": [
        ("revenue_growth_3y", 0.30, False),
        ("eps_growth_3y", 0.30, False),
        # Derived column: roe × (1 - payout_ratio). Higher retention at
        # high ROE = compounder territory.
        ("sustainable_growth_rate", 0.25, False),
        ("capex_growth_5y", 0.15, False),  # reinvestment intensity proxy
    ],
    # config#2428: rebalanced 2-component -> 3-component split adding an
    # institutional-accumulation signal (net 13F fund count delta,
    # gated/scaled by the previously-orphaned institutional_min_funds /
    # institutional_boost research params — see
    # ``institutional_accumulation_score`` below). Weights: payout_ratio
    # and capex_growth_5y each give up 0.15 (0.50 -> 0.35) to fund the new
    # 0.30 institutional component — smart money accumulating/distributing
    # is itself a stewardship signal (are large, informed holders
    # endorsing management's capital allocation with their own capital),
    # but it's noisier and lower-conviction than the two existing
    # fundamental components, hence the smaller share.
    "stewardship_score": [
        # Lower payout = more retention capacity for compounders. Yes,
        # "good stewardship" is context-dependent (high payout is fine
        # for a low-ROIC mature business returning cash; low payout is
        # fine for a high-ROIC compounder), but cross-sectional ranking
        # within sector partially neutralizes this — mature
        # high-payout sectors (Utilities, REITs) rank against each other,
        # growth sectors rank against each other.
        ("payout_ratio", 0.35, True),
        ("capex_growth_5y", 0.35, False),  # sustained reinvestment
        # Derived column: gated/scaled net institutional fund accumulation
        # (n_funds_increasing - n_funds_decreasing) for the most recent
        # 13F quarter. NaN when the ticker has no inst_ownership row
        # (no 13F coverage yet / not held by any tracked fund) — same
        # NaN-propagates-and-reallocates convention as every other
        # component here.
        ("institutional_accumulation_score", 0.30, False),
    ],
}


def _resolve_composite_defs() -> dict[str, list[tuple[str, float, bool]]]:
    """Package override → baseline fallback for the composite definitions.

    The package expresses each component as [column, weight, invert]; tuples
    are restored here so downstream consumption is shape-identical.
    """
    from config import FACTOR_COMPOSITES_CFG

    if not FACTOR_COMPOSITES_CFG:
        return _BASELINE_COMPOSITE_DEFS
    return {
        composite: [tuple(component) for component in components]
        for composite, components in FACTOR_COMPOSITES_CFG.items()
    }


_COMPOSITE_DEFS = _resolve_composite_defs()

# Derived raw factor columns — computed by ``_add_derived_factors`` before
# the percentile-rank step. Each entry is (output_col, fn) where fn takes
# the merged dataframe and returns the new Series. Listed separately so the
# raw → composite path is fully transparent: only ``_COMPOSITE_DEFS``
# components ever land in the final composites; this dict is the audit
# trail for any column that didn't come straight off the parquets.
_DERIVED_FACTOR_DEFS: dict[str, str] = {
    # roe × (1 - payout_ratio). When either input is NaN, the product
    # propagates NaN — partial-coverage handling continues to work
    # downstream.
    "sustainable_growth_rate": "roe * (1 - payout_ratio)",
    # config#2428: net institutional accumulation (n_funds_increasing -
    # n_funds_decreasing), gated by institutional_min_funds and scaled by
    # institutional_boost (both from config.get_research_params() —
    # previously orphaned once the CIK-buggy fetcher that used to read
    # them was deprecated). NaN when inst_ownership has no row for the
    # ticker. See _add_institutional_accumulation_factor.
    "institutional_accumulation_score": (
        "gate(n_funds_increasing - n_funds_decreasing, institutional_min_funds) * institutional_boost"
    ),
}


def _add_derived_factors(merged: pd.DataFrame) -> pd.DataFrame:
    """Compute derived raw factor columns referenced by ``_COMPOSITE_DEFS``.

    Computes ``sustainable_growth_rate`` = roe × (1 - payout_ratio) and
    ``institutional_accumulation_score`` (config#2428 — see
    ``_add_institutional_accumulation_factor``). NaN inputs propagate to
    NaN outputs so the partial-coverage handling in
    ``compute_factor_composites`` continues to work — a ticker with roe
    present but payout_ratio missing contributes NaN to
    sustainable_growth_rate and the weight reallocates pro-rata to the
    other Growth pillar components (same contract for Stewardship +
    institutional_accumulation_score).

    Mutates ``merged`` in place AND returns it for convenience. Idempotent —
    safe to call twice (overwrites the derived columns).
    """
    if "roe" in merged.columns and "payout_ratio" in merged.columns:
        # 1 - payout_ratio = retention rate. NaN propagates through both
        # the subtraction and the multiplication.
        merged["sustainable_growth_rate"] = merged["roe"] * (1.0 - merged["payout_ratio"])
    else:
        # Phase 3a hasn't flowed through yet — emit NaN explicitly so the
        # rank step degrades gracefully. (Pandas treats a missing column
        # via direct indexing as a KeyError; the explicit NaN insertion
        # mirrors the rank-step contract.)
        merged["sustainable_growth_rate"] = float("nan")

    merged = _add_institutional_accumulation_factor(merged)
    return merged


def _add_institutional_accumulation_factor(merged: pd.DataFrame) -> pd.DataFrame:
    """Compute ``institutional_accumulation_score`` (config#2428).

    Net 13F fund accumulation = ``n_funds_increasing - n_funds_decreasing``
    for the most recent quarter, gated by ``institutional_min_funds`` (a
    ticker needs at least that many funds *moving* — increasing OR
    decreasing — before the signal is considered meaningful; below that
    threshold it's too thin a sample to trust, so we emit 0/neutral
    rather than a noisy raw delta) and scaled by ``institutional_boost``
    (both research params — orphaned since the CIK-buggy fetcher that
    used to consume them was deprecated 2026-07-13; this finally rewires
    them).

    NaN when ``n_funds_increasing`` / ``n_funds_decreasing`` aren't present
    on ``merged`` at all (inst_ownership wasn't joined in — e.g. a caller
    invoking ``compute_factor_composites`` without an inst_ownership_df,
    same tolerant-reader contract as the Phase 3a fields). Per-row NaN
    (ticker present in the join but with null fund counts — i.e. no 13F
    coverage for that name this quarter) also propagates to NaN, which is
    the correct "no data" signal — NOT the same as "0 net accumulation".

    Mutates ``merged`` in place AND returns it. Idempotent.
    """
    if "n_funds_increasing" not in merged.columns or "n_funds_decreasing" not in merged.columns:
        merged["institutional_accumulation_score"] = float("nan")
        return merged

    try:
        from config import get_research_params

        params = get_research_params()
        min_funds = params["institutional_min_funds"]
        boost = params["institutional_boost"]
    except Exception as e:  # noqa: BLE001 - config read must never break scoring
        logger.debug(
            "institutional_accumulation_score: research params unavailable (%s), using defaults min_funds=3 boost=3.0",
            e,
        )
        min_funds = 3
        boost = 3.0

    n_inc = merged["n_funds_increasing"]
    n_dec = merged["n_funds_decreasing"]
    net = n_inc - n_dec  # NaN propagates when either side is NaN

    # Gate: total funds *moving* (either direction) below min_funds is too
    # thin a sample — treat as neutral (0) rather than a noisy raw delta.
    # Rows with NaN net (no inst_ownership row at all) stay NaN through the
    # np.where below since the comparison against NaN is False either way
    # but we explicitly re-mask afterwards to be unambiguous.
    n_moving = n_inc.fillna(0) + n_dec.fillna(0)
    gated = net.where(n_moving >= min_funds, other=0.0)
    gated = gated.mask(net.isna(), other=float("nan"))

    merged["institutional_accumulation_score"] = gated * boost
    return merged


def _within_sector_pct_rank(
    df: pd.DataFrame,
    factor_col: str,
    sector_col: str,
    invert: bool = False,
) -> pd.Series:
    """Compute percentile rank (0-100) of `factor_col` within each sector.

    NaN inputs propagate (return NaN — composite weight reallocates to
    other available factors per ticker). If `invert=True`, percentile is
    inverted (e.g. for PE ratio: highest PE → lowest score).

    Pandas `rank(pct=True)` handles ties by averaging ranks (the standard
    Spearman-tie convention).
    """
    if factor_col not in df.columns:
        return pd.Series([float("nan")] * len(df), index=df.index)
    ranks = df.groupby(sector_col)[factor_col].rank(pct=True, na_option="keep")
    pct = ranks * 100.0
    if invert:
        pct = 100.0 - pct
    return pct


def compute_factor_composites(
    technical_df: pd.DataFrame,
    fundamental_df: pd.DataFrame,
    sector_map: dict[str, str],
    inst_ownership_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute the 6 factor composites per ticker.

    Args:
        technical_df: feature store technical.parquet (per-ticker price-derived).
            Must include `ticker` column + the raw factor columns referenced
            in _COMPOSITE_DEFS for momentum / low_vol composites.
        fundamental_df: feature store fundamental.parquet (per-ticker
            Finnhub-sourced). Must include `ticker` column + the raw factor
            columns referenced in _COMPOSITE_DEFS for quality / value.
        sector_map: {ticker: sector_name} mapping. Tickers without a sector
            mapping default to ``"Unknown"`` and are ranked together.
        inst_ownership_df: optional 13F institutional-ownership snapshot
            (config#2428), same shape as
            ``data.substrate.reader.read_inst_ownership()`` — one row per
            ticker for the most recent quarter with ``n_funds_increasing``
            / ``n_funds_decreasing`` columns. Omitted or empty →
            ``institutional_accumulation_score`` emits NaN for every
            ticker (tolerant-reader contract, same as the Phase 3a
            fields when fundamental_df lacks them) and stewardship_score
            falls back to its 2 remaining components.

    Returns:
        DataFrame with columns:
            ticker, sector,
            quality_score, momentum_score, low_vol_score, value_score,
            growth_score, stewardship_score,
            quality_n, momentum_n, low_vol_n, value_n, growth_n, stewardship_n
            (the *_n columns count how many raw factors actually
            contributed per composite — see partial-data handling below.)
    """
    # Merge tech + fundamental on ticker — outer join so a ticker missing
    # from one source still produces the composites for which we DO have
    # data (partial coverage is normal: a name without fundamentals can
    # still get momentum + low_vol).
    merged = technical_df.merge(
        fundamental_df.drop(columns=[c for c in fundamental_df.columns if c == "date"], errors="ignore"),
        on="ticker",
        how="outer",
        suffixes=("", "_fund"),
    )

    # config#2428: outer-join in the 13F institutional-ownership snapshot
    # (same partial-coverage posture as fundamental_df above — a ticker
    # with no 13F row still gets every OTHER composite fully populated;
    # only institutional_accumulation_score / stewardship_score degrade).
    if inst_ownership_df is not None and len(inst_ownership_df) > 0 and "ticker" in inst_ownership_df.columns:
        inst_cols = ["ticker", "n_funds_increasing", "n_funds_decreasing"]
        inst_cols = [c for c in inst_cols if c in inst_ownership_df.columns]
        merged = merged.merge(
            inst_ownership_df[inst_cols],
            on="ticker",
            how="outer",
            suffixes=("", "_inst"),
        )

    merged["sector"] = merged["ticker"].map(lambda t: sector_map.get(t, "Unknown"))

    # Phase 3b — add derived columns referenced by the new composites
    # BEFORE the percentile-rank step. Currently:
    #   sustainable_growth_rate = roe × (1 - payout_ratio)
    merged = _add_derived_factors(merged)

    for composite, components in _COMPOSITE_DEFS.items():
        # Compute within-sector percentile rank for each component
        component_ranks: list[tuple[str, float, pd.Series]] = []
        for factor_col, weight, invert in components:
            ranks = _within_sector_pct_rank(merged, factor_col, "sector", invert=invert)
            component_ranks.append((factor_col, weight, ranks))

        # Per-ticker weighted average of available component ranks. If a
        # component is NaN for a ticker, its weight reallocates pro-rata
        # to the components that ARE available — partial-coverage tickers
        # get a defensible composite from whatever data they have.
        composite_vals: list[float] = []
        composite_n: list[int] = []
        for i in range(len(merged)):
            num = 0.0
            denom = 0.0
            count = 0
            for _, weight, ranks in component_ranks:
                v = ranks.iloc[i]
                if pd.notna(v):
                    num += weight * v
                    denom += weight
                    count += 1
            composite_vals.append(num / denom if denom > 0 else float("nan"))
            composite_n.append(count)

        merged[composite] = composite_vals
        merged[f"{composite[:-6]}_n"] = composite_n  # quality_n, momentum_n, etc.

    keep_cols = ["ticker", "sector"] + list(_COMPOSITE_DEFS.keys()) + [f"{c[:-6]}_n" for c in _COMPOSITE_DEFS.keys()]
    return merged[keep_cols].copy()


def write_factor_profiles_to_s3(
    profiles_df: pd.DataFrame,
    run_date: str,
    bucket: str | None = None,
    provenance: dict | None = None,
) -> str:
    """Write factor profiles to S3 as `{date}/by_ticker.json`.

    Schema: ``{ticker: {sector, quality_score, momentum_score,
    low_vol_score, value_score, *_n}}``. Consumers (composite scoring,
    quant @tool, backtester attribution) read this single canonical
    artifact rather than re-deriving from raw parquets.

    ``provenance`` records which dated feature snapshot each group resolved
    to. It is written to a SEPARATE ``{date}/provenance.json`` rather than
    folded into ``by_ticker.json`` — that payload is a flat ticker→record
    mapping every consumer iterates, so a reserved non-ticker key would be
    a schema change to a live contract for no gain. Once a cut can be built
    from carried-forward inputs, "which snapshot produced this ranking" has
    to be recoverable from an artifact rather than inferred from the
    calendar (principles §1, transparency).

    Returns the S3 key written.
    """
    import boto3

    bucket = bucket or os.environ.get("S3_BUCKET", "alpha-engine-research")
    key = f"factors/profiles/{run_date}/by_ticker.json"

    # Convert to {ticker: {field: value}} dict, dropping NaN scores
    payload: dict[str, dict] = {}
    for _, row in profiles_df.iterrows():
        ticker = row["ticker"]
        record = {"sector": row["sector"]}
        for col in profiles_df.columns:
            if col in ("ticker", "sector"):
                continue
            v = row[col]
            if pd.notna(v):
                record[col] = float(v) if isinstance(v, (int, float)) else int(v)
        payload[ticker] = record

    body = json.dumps(payload, indent=2)
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")

    if provenance is not None:
        s3.put_object(
            Bucket=bucket,
            Key=f"factors/profiles/{run_date}/provenance.json",
            Body=json.dumps(provenance, indent=2),
            ContentType="application/json",
        )

    # Also write `latest.json` sidecar for cache-warm convenience
    latest_key = "factors/profiles/latest.json"
    s3.put_object(Bucket=bucket, Key=latest_key, Body=body, ContentType="application/json")

    logger.info(
        "Factor profiles written to s3://%s/%s (%d tickers, %d composite columns)",
        bucket,
        key,
        len(payload),
        len(_COMPOSITE_DEFS),
    )
    return key


def read_factor_profiles_from_s3(
    run_date: str | None = None,
    bucket: str | None = None,
) -> dict[str, dict] | None:
    """Read factor profiles from S3.

    If `run_date` is None, reads `factors/profiles/latest.json` sidecar
    (cheap; no S3 list call). Returns None on any read failure (consumers
    should treat absence as "no factor data available, skip factor blend").
    """
    import boto3
    from botocore.exceptions import ClientError

    bucket = bucket or os.environ.get("S3_BUCKET", "alpha-engine-research")
    key = f"factors/profiles/{run_date}/by_ticker.json" if run_date else "factors/profiles/latest.json"
    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchKey":
            logger.warning("Factor profile read error from s3://%s/%s: %s", bucket, key, e)
        return None
    except Exception as e:
        logger.warning("Unexpected factor profile read error: %s", e)
        return None


# ── Per-group snapshot resolution (daily-scanner support) ────────────────────
# The two feature groups this module reads have DIFFERENT natural cadences,
# and the original exact-dated `features/{run_date}/` read collapsed that
# distinction into "Saturday or nothing":
#
#   technical    — recomputed every trading day by alpha-engine-data's
#                  builders/daily_append.py (weekday MorningEnrich).
#   fundamental  — collected WEEKLY by collectors/fundamentals.py in Saturday
#                  DataPhase1. PE / ROE / market cap move on earnings, so a
#                  daily refresh would recompute identical numbers at 5x the
#                  FMP/Finnhub call volume. Carry-forward is correct here;
#                  daily collection is not.
#
# Each group therefore resolves to the latest snapshot ON OR BEFORE run_date
# and carries its own staleness bound, counted in NYSE TRADING days
# (feedback_staleness_windows_trading_days_not_calendar_260717). Past the
# bound this RAISES: trading one stale surface (the whole point of the daily
# cutover) for another that is silently six weeks old would be a worse bug
# than the one being fixed.
_MAX_TECHNICAL_STALENESS_TD = 3
_MAX_FUNDAMENTAL_STALENESS_TD = 10

# The technical columns the composites above actually reference. Derived by
# hand from _BASELINE_COMPOSITE_DEFS ∩ alpha-engine-data's `technical`
# feature group — the composites reference 21 columns total, of which these
# 8 are technical, 11 are fundamental, and 2 (institutional_accumulation_score,
# sustainable_growth_rate) are derived elsewhere in this module.
#
# Narrower than the scanner's 12-column tech_score set on purpose: the two
# consumers overlap but differ, and this read spans ~900 symbols.
# The 8 technical columns this module's factor pillar reads. DECLARED IN THE
# READER (``data/fetchers/feature_store_reader.py``) and re-exported here
# under its historical name: the reader builds the union projection serving
# both this call site and the scanner's from one ``read_batch``
# (alpha-engine-config-I6855), and a union derived from a second copy of the
# list would drift silently the first time either side gained a column.
_FACTOR_TECHNICAL_COLS: tuple[str, ...] = _READER_FACTOR_TECHNICAL_COLS


def _technical_frame_from_arctic(tickers: list[str], run_date_str: str) -> pd.DataFrame:
    """Technical factor inputs from the DAILY ArcticDB surface.

    The S3 ``features/{date}/technical.parquet`` partition only exists on
    Saturdays, so resolving this group to "latest on or before run_date"
    would still hand a Thursday run a snapshot 4 trading days old — past its
    own staleness bound, i.e. the daily scanner would work Mon-Wed and fail
    Thu-Fri. Carry-forward is the right answer for fundamentals, which are
    genuinely collected weekly; it is the wrong answer for technicals, which
    alpha-engine-data recomputes every trading day and merely does not
    re-snapshot to S3.
    """
    from data.fetchers.feature_store_reader import read_latest_features

    rows = (
        read_latest_features(
            tickers,
            columns=_FACTOR_TECHNICAL_COLS,
            ref_date=_date.fromisoformat(run_date_str),
        )
        or {}
    )
    if not rows:
        raise FactorSnapshotStalenessError(
            f"factor group 'technical': ArcticDB returned no rows for {len(tickers)} tickers as of {run_date_str}"
        )
    return pd.DataFrame([{"ticker": t, **vals} for t, vals in rows.items()])


class FactorSnapshotStalenessError(RuntimeError):
    """A feature group's newest snapshot is older than its staleness bound."""


def _list_feature_snapshot_dates(s3, bucket: str) -> list[str]:
    """Sorted ``YYYY-MM-DD`` partition names under ``features/``.

    Paginated: the prefix accumulates one partition per weekly run and has
    long outlived a single 1000-key ``list_objects_v2`` page.
    """
    dates: list[str] = []
    token = None
    # Bounded, and both continuation conditions are compared by IDENTITY
    # rather than truthiness. A `while True` whose exit depends on the
    # truthiness of a remote response field is one unexpected payload away
    # from spinning forever inside a Lambda until it times out — and any
    # test double returning a permissive object satisfies "truthy" for both
    # IsTruncated and NextContinuationToken, which is exactly how this hung
    # a local suite before the guard existed. 1000 pages x 1000 keys is far
    # past any real size of this prefix.
    for _ in range(1000):
        kwargs = {"Bucket": bucket, "Prefix": "features/", "Delimiter": "/"}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for p in resp.get("CommonPrefixes") or []:
            part = p["Prefix"].rstrip("/").split("/")[-1]
            if len(part) == 10 and part[4] == "-" and part[7] == "-":
                dates.append(part)
        if resp.get("IsTruncated") is not True:
            break
        token = resp.get("NextContinuationToken")
        if not isinstance(token, str) or not token:
            break
    else:
        logger.warning(
            "features/ prefix listing hit the 1000-page cap — returning %d partitions, which may be incomplete",
            len(dates),
        )
    return sorted(dates)


def _resolve_group_snapshot(
    s3,
    bucket: str,
    group: str,
    run_date_str: str,
    max_staleness_td: int,
    *,
    candidate_dates: list[str],
) -> tuple[pd.DataFrame, str]:
    """Latest ``features/{date}/{group}.parquet`` on or before ``run_date_str``.

    Returns ``(df, resolved_date)``. Raises ``FactorSnapshotStalenessError``
    when nothing resolves, or when what resolves is beyond
    ``max_staleness_td`` NYSE trading days behind ``run_date_str``.
    """
    from nousergon_lib.trading_calendar import count_trading_days

    on_or_before = [d for d in candidate_dates if d <= run_date_str]
    for resolved in reversed(on_or_before):
        key = f"features/{resolved}/{group}.parquet"
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
        except Exception as exc:  # noqa: BLE001
            # EXPECTED, not swallowed: weekday partitions carry `technical`
            # and no `fundamental`, so resolution walks back past them. The
            # search is still bounded — exhausting every candidate raises
            # below, and resolving something too old raises on staleness, so
            # neither a missing group nor an ancient one can pass silently.
            logger.debug("factor group %r absent at %s (%s)", group, resolved, exc)
            continue
        df = pd.read_parquet(io.BytesIO(obj["Body"].read()), engine="pyarrow")
        stale_td = count_trading_days(_date.fromisoformat(resolved), _date.fromisoformat(run_date_str))
        if stale_td > max_staleness_td:
            raise FactorSnapshotStalenessError(
                f"factor group {group!r}: newest snapshot on or before "
                f"{run_date_str} is {resolved}, {stale_td} NYSE trading days "
                f"behind (bound {max_staleness_td}). Refusing to rank the "
                "universe on it. Producer: alpha-engine-data "
                f"{'builders/daily_append.py (weekday MorningEnrich)' if group == 'technical' else 'collectors/fundamentals.py (Saturday DataPhase1)'}."
            )
        logger.info(
            "Factor group %r resolved to %s (%d trading days behind %s, bound %d)",
            group,
            resolved,
            stale_td,
            run_date_str,
            max_staleness_td,
        )
        return df, resolved

    raise FactorSnapshotStalenessError(
        f"factor group {group!r}: no features/{{date}}/{group}.parquet exists "
        f"on or before {run_date_str} (searched {len(on_or_before)} partitions)"
    )


def compute_and_write_factor_profiles(
    run_date: str | _date,
    sector_map: dict[str, str],
    bucket: str | None = None,
) -> str:
    """Read raw factor parquets, compute composites, write profiles.

    Runs on ANY trading day, not only Saturday. Each feature group resolves
    independently to the latest snapshot on or before ``run_date`` — see
    ``_resolve_group_snapshot`` for why the two groups carry different
    staleness bounds.

    Reads:
      - s3://{bucket}/features/{<= run_date}/technical.parquet
      - s3://{bucket}/features/{<= run_date}/fundamental.parquet
      - s3://{bucket}/features/metron_supplemental/{run_date}/{technical,fundamental}.parquet
        (OPTIONAL — metron-ops#177: Metron-held/watchlisted tickers outside the
        S&P500+400 universe above, written by alpha-engine-data's
        compute_metron_supplemental_features. Absent on any run where the
        producer found nothing to add, or hasn't shipped yet — never blocks
        the main compute.)
      - data/inst_ownership latest.json + parquet via
        ``data.substrate.reader.read_inst_ownership`` (config#2428,
        OPTIONAL — same reader ``qual_tools.get_institutional_activity``
        uses; absent/empty just yields NaN institutional_accumulation_score,
        never blocks the main compute).

    Writes:
      - s3://{bucket}/factors/profiles/{run_date}/by_ticker.json
      - s3://{bucket}/factors/profiles/{run_date}/provenance.json
      - s3://{bucket}/factors/profiles/latest.json (sidecar)

    Returns the dated S3 key written.
    """
    import boto3

    bucket = bucket or os.environ.get("S3_BUCKET", "alpha-engine-research")
    run_date_str = run_date.isoformat() if isinstance(run_date, _date) else run_date

    s3 = boto3.client("s3")

    # One listing shared by both group resolutions — the prefix is identical
    # and it is the expensive part of this read path.
    candidate_dates = _list_feature_snapshot_dates(s3, bucket)

    def _read_metron_supplemental(parquet_name: str) -> pd.DataFrame | None:
        key = f"features/metron_supplemental/{run_date_str}/{parquet_name}.parquet"
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            return pd.read_parquet(io.BytesIO(obj["Body"].read()), engine="pyarrow")
        except Exception:  # noqa: BLE001 - genuinely optional artifact, never fabricate/raise
            return None

    if os.environ.get("SCANNER_FEATURE_SOURCE", "arcticdb").strip().lower() == "s3":
        technical_df, technical_date = _resolve_group_snapshot(
            s3,
            bucket,
            "technical",
            run_date_str,
            _MAX_TECHNICAL_STALENESS_TD,
            candidate_dates=candidate_dates,
        )
    else:
        technical_df = _technical_frame_from_arctic(sorted(sector_map), run_date_str)
        technical_date = run_date_str

    fundamental_df, fundamental_date = _resolve_group_snapshot(
        s3,
        bucket,
        "fundamental",
        run_date_str,
        _MAX_FUNDAMENTAL_STALENESS_TD,
        candidate_dates=candidate_dates,
    )

    supplemental_technical = _read_metron_supplemental("technical")
    if supplemental_technical is not None and not supplemental_technical.empty:
        technical_df = pd.concat([technical_df, supplemental_technical], ignore_index=True)
    supplemental_fundamental = _read_metron_supplemental("fundamental")
    if supplemental_fundamental is not None and not supplemental_fundamental.empty:
        fundamental_df = pd.concat([fundamental_df, supplemental_fundamental], ignore_index=True)

    # config#2428: 13F institutional-ownership snapshot for
    # institutional_accumulation_score. Same reader + bucket convention as
    # qual_tools.get_institutional_activity — genuinely optional artifact
    # (no 13F data yet this quarter, or reader import fails in an
    # environment without the substrate module wired up); never blocks
    # the main compute, same posture as the Metron-supplemental reads
    # above.
    try:
        from data.substrate.reader import read_inst_ownership

        inst_ownership_df = read_inst_ownership(s3_client=s3, bucket=bucket)
    except Exception as e:  # noqa: BLE001 - optional artifact, never block scoring
        logger.info("inst_ownership read unavailable for factor profiles: %s", e)
        inst_ownership_df = None

    profiles = compute_factor_composites(
        technical_df=technical_df,
        fundamental_df=fundamental_df,
        sector_map=sector_map,
        inst_ownership_df=inst_ownership_df,
    )

    return write_factor_profiles_to_s3(
        profiles,
        run_date_str,
        bucket=bucket,
        provenance={
            "run_date": run_date_str,
            "groups": {
                "technical": {
                    "resolved_date": technical_date,
                    "carried_forward": technical_date != run_date_str,
                    "source": (
                        "s3_weekly_snapshot"
                        if os.environ.get("SCANNER_FEATURE_SOURCE", "arcticdb").strip().lower() == "s3"
                        else "arcticdb_daily"
                    ),
                },
                "fundamental": {
                    "resolved_date": fundamental_date,
                    "carried_forward": fundamental_date != run_date_str,
                },
            },
        },
    )
