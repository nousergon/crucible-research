"""
Universe scoreboard — the full ~900-name S&P 500+400 scanner universe with
per-stock attractiveness + factor + raw-metric data, published as ONE typed S3
artifact for the dashboard's filterable universe board.

Motivation. The scanner evaluates ~903 tickers every Saturday and the factor
substrate (``scoring/factor_scoring.py`` → ``factors/profiles/{date}/``) already
computes 6 pillar scores for ALL of them — but downstream only the ~25-60 that
survive the quant filter + agent pipeline get a composite attractiveness score
or appear in ``signals.json``. The rejected ~840 retain factor + technical data
in the ``scanner_evaluations`` SQLite table, invisible outside the console DB.
This module lifts that full-universe data into a product contract so the
dashboard can show, sort, and FILTER all ~900 names by attractiveness, by each
pillar, by raw valuation/fundamental/technical metric ranges, by sector, by
country, and by scanner gate status.

Output (versioned — consumers pin on ``schema_version``):
  ``s3://{bucket}/scanner/universe/{run_date}/universe.json``
  ``s3://{bucket}/scanner/universe/latest.json`` (sidecar)

Schema::

  {
    "schema_version": 1,
    "as_of": "YYYY-MM-DD",
    "universe_count": int,
    "attractiveness_method": "equal_weight_available_pillars",
    "pillars": ["quality", "value", "momentum", "growth", "stewardship", "defensiveness"],
    "stocks": [
      {
        "ticker": "AAPL",
        "sector": "Information Technology",   # GICS, from factor profile / sector_map
        "country": "United States",            # domicile, from universe_classification (null if uncovered)
        "industry": "Consumer Electronics",    # null if uncovered
        "attractiveness_score": 0-100 | null,  # equal-weight mean of available pillar scores
        "pillars": {quality, value, momentum, growth, stewardship, defensiveness},  # 0-100 | null each
        "pillar_coverage": {quality: int, ...},   # # raw factors that contributed per pillar
        "focus_score": 0-100 | null,           # scanner's 4-factor regime-blended subscore
        "focus_stance": "momentum" | ... | null,
        "tech_score": 0-100 | null,            # scanner's pure-technical attractiveness
        "gate": {"quant_filter_pass": 0|1, "filter_fail_reason": str | null},
        "metrics": {  # DISPLAY-ready raw units (see _DISPLAY_METRICS denorm contract)
          "current_price", "market_cap", "avg_volume",
          "pe", "pb", "fcf_yield", "dividend_yield", "debt_to_equity", "current_ratio", "payout_ratio",
          "roe", "gross_margin", "revenue_growth_3y", "eps_growth_3y",
          "rsi_14", "momentum_20d", "return_60d", "return_120d",
          "realized_vol_20d", "atr_pct", "dist_from_52w_high", "price_vs_ma200", "beta"
        }
      },
      ...
    ]
  }

Failure posture. This is a SECONDARY observability artifact hung off the
research run's primary deliverable (``signals.json``). Per the no-silent-fails
exception for secondary observability, the archive_writer call site fail-SOFTs
with a WARN log (the board is dashboard visibility, not a trading-path
contract) — a board-write failure must NOT fail the research run. The builder
itself raises on genuinely broken inputs so the caller's WARN records a real
fault rather than silently emitting an empty board.
"""

from __future__ import annotations

import io
import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

UNIVERSE_BOARD_SCHEMA_VERSION = 1

# Pillar → factor-profile field. Single source of truth lives in
# scoring/composite.py::_PILLAR_TO_FACTOR_KEY; imported lazily in the builder so
# this module has no import-time dependency on composite's config loading.
_PILLAR_ORDER = ("quality", "value", "momentum", "growth", "stewardship", "defensiveness")

# DISPLAY metric contract: (feature_store_column, output_field, denorm_multiplier).
#
# The feature store (alpha-engine-data ``features/{date}/*.parquet``) stores
# several valuation ratios PREDICTOR-NORMALIZED (e.g. pe_ratio = trailing P/E ÷
# 30) — see ``features/SCHEMA.md`` §Fundamental, whose normalizations are pinned
# by that repo's ``test_schema_contract.py``. ``denorm_multiplier`` recovers the
# human-readable raw value for display (multiplier 1.0 = already clean units).
# This map MIRRORS the documented SCHEMA.md contract; if a future PR changes a
# normalization in the data repo, ``test_universe_board.py`` and the data repo's
# schema-contract test are the paired guards.
_FUNDAMENTAL_METRICS: tuple[tuple[str, str, float], ...] = (
    ("pe_ratio", "pe", 30.0),                 # PE = pe_ratio × 30
    ("pb_ratio", "pb", 5.0),                  # PB = pb_ratio × 5
    ("debt_to_equity", "debt_to_equity", 2.0),  # D/E = col × 2
    ("current_ratio", "current_ratio", 3.0),  # CR = col × 3
    ("fcf_yield", "fcf_yield", 1.0),          # decimal pct — clean
    ("dividend_yield", "dividend_yield", 1.0),
    ("payout_ratio", "payout_ratio", 1.0),
    ("roe", "roe", 1.0),                      # decimal pct — clean
    ("gross_margin", "gross_margin", 1.0),    # 0–1 fraction — clean
    ("revenue_growth_3y", "revenue_growth_3y", 1.0),  # CAGR — clean
    ("eps_growth_3y", "eps_growth_3y", 1.0),
    ("market_cap_raw", "market_cap", 1.0),    # raw dollars — clean
)
_TECHNICAL_METRICS: tuple[tuple[str, str, float], ...] = (
    ("rsi_14", "rsi_14", 1.0),                # 0–100 — clean
    ("momentum_20d", "momentum_20d", 1.0),    # decimal return — clean
    ("return_60d", "return_60d", 1.0),
    ("return_120d", "return_120d", 1.0),
    ("realized_vol_20d", "realized_vol_20d", 1.0),  # annualized decimal — clean
    ("atr_14_pct", "atr_pct", 1.0),           # decimal pct — clean
    ("dist_from_52w_high", "dist_from_52w_high", 1.0),
    ("price_vs_ma200", "price_vs_ma200", 1.0),
    ("beta_60d", "beta", 1.0),                # dimensionless — clean
    ("avg_volume_20d_raw", "avg_volume", 1.0),  # raw shares — clean
)

DEFAULT_BUCKET = "alpha-engine-research"


def _bucket(bucket: str | None) -> str:
    return bucket or os.environ.get("S3_BUCKET", DEFAULT_BUCKET)


def _attractiveness(pillar_scores: dict[str, Optional[float]]) -> Optional[float]:
    """Equal-weight mean of the AVAILABLE pillar scores (partial-coverage
    reallocation — a name with only 4 of 6 pillars still gets a defensible
    score from those 4, mirroring factor_scoring's per-composite handling).
    Returns None when no pillar is available."""
    vals = [v for v in pillar_scores.values() if v is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 2)


def _num(v: Any, multiplier: float = 1.0) -> Optional[float]:
    """Coerce to a finite float (applying ``multiplier``) or None. NaN / inf /
    non-numeric → None (a coverage gap, never a fabricated value)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):  # NaN / inf
        return None
    return round(f * multiplier, 6)


def build_universe_board(
    run_date: str,
    scanner_evals: list[dict],
    *,
    factor_profiles: dict | None = None,
    classification: dict | None = None,
    technical_df: "Any" = None,
    fundamental_df: "Any" = None,
    bucket: str | None = None,
    s3_client: Any = None,
) -> dict:
    """Assemble the full-universe scoreboard payload.

    Args:
        run_date: YYYY-MM-DD run stamp.
        scanner_evals: the per-ticker scanner-evaluation rows built in
            ``archive_writer`` (ticker, sector, tech_score, current_price,
            focus_score/stance, quant_filter_pass, filter_fail_reason, …). This
            is the authoritative universe membership + gate source.
        factor_profiles: ``{ticker: {sector, *_score, *_n}}`` (factors/profiles).
            Read from S3 when None.
        classification: ``{ticker: {sector, country, industry}}`` from the
            universe_classification artifact. Read from S3 when None; an empty
            map degrades country/industry to null (fail-soft, never guessed).
        technical_df / fundamental_df: feature-store parquets. Read from S3 when
            None.
        bucket / s3_client: S3 wiring (defaults: env bucket + a fresh boto3
            client) — injectable for tests.

    Returns the board dict (also the unit under the producer contract test).

    Raises when ``scanner_evals`` is empty (a research run that produced no
    universe is a real fault — the caller's WARN records it).
    """
    if not scanner_evals:
        raise ValueError(
            "universe_board: scanner_evals is empty — the research run produced "
            "no universe rows; refusing to emit an empty board (no-silent-fails)."
        )

    if factor_profiles is None:
        factor_profiles = _read_factor_profiles(run_date, bucket, s3_client) or {}
    if classification is None:
        classification = _read_classification(bucket, s3_client) or {}
    if technical_df is None:
        technical_df = _read_parquet("technical", run_date, bucket, s3_client)
    if fundamental_df is None:
        fundamental_df = _read_parquet("fundamental", run_date, bucket, s3_client)

    from scoring.composite import _PILLAR_TO_FACTOR_KEY

    tech_by_ticker = _index_parquet(technical_df)
    fund_by_ticker = _index_parquet(fundamental_df)

    stocks: list[dict] = []
    for row in scanner_evals:
        ticker = row.get("ticker")
        if not ticker:
            continue
        profile = factor_profiles.get(ticker, {})
        cls = classification.get(ticker, {})
        tech = tech_by_ticker.get(ticker, {})
        fund = fund_by_ticker.get(ticker, {})

        pillar_scores = {
            pillar: _num(profile.get(_PILLAR_TO_FACTOR_KEY[pillar]))
            for pillar in _PILLAR_ORDER
        }
        pillar_coverage = {
            pillar: int(profile[f"{_PILLAR_TO_FACTOR_KEY[pillar][:-6]}_n"])
            for pillar in _PILLAR_ORDER
            if f"{_PILLAR_TO_FACTOR_KEY[pillar][:-6]}_n" in profile
        }

        metrics: dict[str, Optional[float]] = {
            "current_price": _num(row.get("current_price")),
        }
        for col, field, mult in _FUNDAMENTAL_METRICS:
            metrics[field] = _num(fund.get(col), mult)
        for col, field, mult in _TECHNICAL_METRICS:
            # tech_score row carries some technicals too; prefer the parquet
            # (full set), fall back to the scanner-eval row for the few it has.
            val = tech.get(col)
            if val is None and col in ("rsi_14",):
                val = row.get(col)
            metrics[field] = _num(val, mult)

        stocks.append({
            "ticker": ticker,
            "sector": profile.get("sector") or row.get("sector") or cls.get("sector"),
            "country": cls.get("country"),
            "industry": cls.get("industry"),
            "attractiveness_score": _attractiveness(pillar_scores),
            "pillars": pillar_scores,
            "pillar_coverage": pillar_coverage,
            "focus_score": _num(row.get("focus_score")),
            "focus_stance": row.get("focus_stance"),
            "tech_score": _num(row.get("tech_score")),
            "gate": {
                "quant_filter_pass": int(row.get("quant_filter_pass", 0) or 0),
                "filter_fail_reason": row.get("filter_fail_reason"),
            },
            "metrics": metrics,
        })

    stocks.sort(
        key=lambda s: (s["attractiveness_score"] is None, -(s["attractiveness_score"] or 0))
    )

    return {
        "schema_version": UNIVERSE_BOARD_SCHEMA_VERSION,
        "as_of": run_date,
        "universe_count": len(stocks),
        "attractiveness_method": "equal_weight_available_pillars",
        "pillars": list(_PILLAR_ORDER),
        "stocks": stocks,
    }


# ── S3 I/O ──────────────────────────────────────────────────────────────────

def _client(s3_client: Any):
    if s3_client is not None:
        return s3_client
    import boto3
    return boto3.client("s3")


def _read_factor_profiles(run_date: str, bucket: str | None, s3_client: Any) -> dict | None:
    """Read ``factors/profiles/{run_date}/by_ticker.json`` (written earlier in
    the same run), falling back to the ``latest.json`` sidecar."""
    s3 = _client(s3_client)
    b = _bucket(bucket)
    for key in (f"factors/profiles/{run_date}/by_ticker.json", "factors/profiles/latest.json"):
        try:
            obj = s3.get_object(Bucket=b, Key=key)
            return json.loads(obj["Body"].read())
        except Exception:
            continue
    logger.warning("[universe_board] no factor profiles readable for %s — pillars will be null", run_date)
    return None


def _read_classification(bucket: str | None, s3_client: Any) -> dict | None:
    """Read ``market_data/universe_classification/latest.json`` → ``{ticker:
    {sector, country, industry}}``. Domicile is near-static so latest is fine.
    Absent artifact (pre-first-production) degrades country/industry to null."""
    s3 = _client(s3_client)
    try:
        obj = s3.get_object(Bucket=_bucket(bucket), Key="market_data/universe_classification/latest.json")
        return json.loads(obj["Body"].read()).get("data", {})
    except Exception:
        logger.warning("[universe_board] universe_classification not readable — country/industry will be null")
        return None


def _read_parquet(name: str, run_date: str, bucket: str | None, s3_client: Any):
    """Read ``features/{run_date}/{name}.parquet`` → DataFrame, or None on miss
    (board still builds; that metric group degrades to null)."""
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        return None
    s3 = _client(s3_client)
    try:
        obj = s3.get_object(Bucket=_bucket(bucket), Key=f"features/{run_date}/{name}.parquet")
        return pd.read_parquet(io.BytesIO(obj["Body"].read()), engine="pyarrow")
    except Exception:
        logger.warning("[universe_board] features/%s/%s.parquet not readable — those metrics null", run_date, name)
        return None


def _index_parquet(df: "Any") -> dict[str, dict]:
    """``{ticker: {col: value}}`` from a feature-store DataFrame (empty when
    df is None / has no ``ticker`` column)."""
    if df is None or not hasattr(df, "columns") or "ticker" not in df.columns:
        return {}
    out: dict[str, dict] = {}
    for record in df.to_dict(orient="records"):
        t = record.get("ticker")
        if t:
            out[str(t)] = record
    return out


def write_universe_board_to_s3(board: dict, run_date: str, *, bucket: str | None = None, s3_client: Any = None) -> str:
    """Write the board to the dated key + the ``latest.json`` sidecar. Returns
    the dated key."""
    s3 = _client(s3_client)
    b = _bucket(bucket)
    body = json.dumps(board, separators=(",", ":"), default=str).encode("utf-8")
    dated_key = f"scanner/universe/{run_date}/universe.json"
    for key in (dated_key, "scanner/universe/latest.json"):
        s3.put_object(Bucket=b, Key=key, Body=body, ContentType="application/json")
    logger.info(
        "[universe_board] wrote %d stocks → s3://%s/%s (+latest)",
        board.get("universe_count", 0), b, dated_key,
    )
    return dated_key


def compute_and_write_universe_board(
    run_date: str,
    scanner_evals: list[dict],
    *,
    bucket: str | None = None,
    s3_client: Any = None,
) -> str:
    """archive_writer entry point — build from S3-resident inputs + the in-memory
    scanner_evals and write the artifact. Returns the dated S3 key."""
    board = build_universe_board(run_date, scanner_evals, bucket=bucket, s3_client=s3_client)
    return write_universe_board_to_s3(board, run_date, bucket=bucket, s3_client=s3_client)
