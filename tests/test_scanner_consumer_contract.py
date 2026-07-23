"""Cross-repo scanner consumer-contract test.

Pins the Research scanner's contract with the alpha-engine-data feature
store: the ``indicators["avg_volume_20d"]`` slot consumed by
``data.scanner.run_quant_filter`` MUST be populated from a RAW-shares
column, not a normalized ratio.

Background — the 901/903 silent-failure bug. The feature store emitted
``avg_volume_20d`` as a normalized ratio (≈1.0) intended for predictor
input, but the scanner's ``MIN_AVG_VOLUME = 500_000`` gate compared it
to raw shares. Result: virtually every feature-store-covered ticker
silently failed the liquidity gate for months. The fix landed
``avg_volume_20d_raw`` in alpha-engine-data Phase 1; this test pins the
consumer flip in Phase 2.

This test fails LOUDLY if a future PR:
  - Reverts the consumer flip (reads ``avg_volume_20d`` instead of
    ``avg_volume_20d_raw`` from feature store rows).
  - Adds a new feature-store read site for the liquidity gate without
    sourcing it from the raw column.

See also:
  - ``alpha-engine-data/features/SCHEMA.md`` §2 (consumer contract).
  - ``~/Development/alpha-engine-docs/private/feature-store-schema-audit-260525.md``.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Fixture: synthetic feature-store rows for 100 tickers ──────────────────


def _synthetic_feature_store_rows(n_tickers: int = 100) -> dict[str, dict]:
    """Build feature-store rows matching the Phase 1 alpha-engine-data
    schema. Each row has BOTH columns:

      - ``avg_volume_20d``      : ~1.0 ratio (predictor input)
      - ``avg_volume_20d_raw``  : ~5_000_000 raw shares (scanner input)

    The 5M raw value is comfortably above ``MIN_AVG_VOLUME = 500_000``
    so every ticker should pass the liquidity gate under a correctly
    wired consumer.
    """
    rows: dict[str, dict] = {}
    for i in range(n_tickers):
        ticker = f"T{i:03d}"
        rows[ticker] = {
            "rsi_14": 55.0,
            "macd_cross": 0.0,
            "macd_above_zero": True,
            "macd_line_last": 0.5,
            "price_vs_ma50": 0.02,
            "price_vs_ma200": 0.05,
            "momentum_20d": 0.03,
            "momentum_5d": 0.01,
            # Both columns present, on different scales.
            "avg_volume_20d": 0.98,           # normalized ratio (~1.0)
            "avg_volume_20d_raw": 5_000_000,  # raw shares
            "atr_14_pct": 0.018,
            "dist_from_52w_high": -0.04,
            "dist_from_52w_low": 0.20,
        }
    return rows


def _synthetic_daily_closes(tickers: list[str], price: float = 50.0) -> dict[str, float]:
    """Above MIN_PRICE for all tickers."""
    return dict.fromkeys(tickers, price)


# ── Consumer-slot contract ────────────────────────────────────────────────


def test_orchestrator_reads_raw_column_into_avg_volume_slot():
    """``scanner_orchestrator`` MUST populate ``indicators["avg_volume_20d"]``
    from ``fs_row["avg_volume_20d_raw"]``, not from the normalized column.

    Sniff-test: median ``avg_volume_20d`` across enriched tickers must be
    in raw-shares scale (>= 1M). If the consumer is wired wrong, the
    median will collapse to ~1.0 and this assertion fails.
    """
    from data.scanner_orchestrator import _build_technical_scores_from_feature_store

    fs_rows = _synthetic_feature_store_rows(n_tickers=100)
    constituents = list(fs_rows.keys())
    sector_map = dict.fromkeys(constituents, "Technology")

    with (
        patch("data.fetchers.feature_store_reader.read_latest_features", return_value=fs_rows),
        patch("data.fetchers.feature_store_reader.read_latest_daily_closes",
              return_value=_synthetic_daily_closes(constituents)),
    ):
        technical_scores, n_enriched = _build_technical_scores_from_feature_store(
            constituents, sector_map,
        )

    assert n_enriched == 100
    avg_vols = [
        ts["avg_volume_20d"] for ts in technical_scores.values()
        if ts.get("avg_volume_20d") is not None
    ]
    assert len(avg_vols) == 100
    median = sorted(avg_vols)[len(avg_vols) // 2]
    assert median >= 1_000_000, (
        f"Scanner consumer slot avg_volume_20d median = {median:,.0f} — "
        "expected raw shares scale (>= 1M). The consumer is reading the "
        "normalized predictor column instead of avg_volume_20d_raw. See "
        "data/scanner_orchestrator.py."
    )


def test_scanner_liquidity_pass_count_is_high_on_raw_consumer():
    """Run the full scanner against the Phase-1-shaped feature store and
    assert ``liquidity_pass=1`` on substantially every ticker.

    Before the fix, ~all tickers failed (avg_volume_20d ≈ 1.0 < 500_000).
    After the fix, ~all tickers pass (avg_volume_20d_raw ≈ 5M > 500_000).
    """
    from data.scanner import run_quant_filter
    from data.scanner_orchestrator import _build_technical_scores_from_feature_store

    fs_rows = _synthetic_feature_store_rows(n_tickers=100)
    constituents = list(fs_rows.keys())
    sector_map = dict.fromkeys(constituents, "Technology")

    with (
        patch("data.fetchers.feature_store_reader.read_latest_features", return_value=fs_rows),
        patch("data.fetchers.feature_store_reader.read_latest_daily_closes",
              return_value=_synthetic_daily_closes(constituents)),
    ):
        technical_scores, _ = _build_technical_scores_from_feature_store(
            constituents, sector_map,
        )

    # price_data empty — exercises the feature-store-only path the scanner
    # Lambda + the standalone scanner use in production.
    run_quant_filter(
        tickers=constituents,
        price_data={},
        technical_scores=technical_scores,
        sector_map=sector_map,
    )
    eval_log = run_quant_filter._last_eval_log
    liquidity_pass_count = sum(1 for r in eval_log if r.get("liquidity_pass") == 1)

    # The hard floor is 60 (per the Phase 2 acceptance criterion in the
    # plan doc). Synthetic fixture is fully clean so we expect ≥ 95.
    assert liquidity_pass_count >= 95, (
        f"Only {liquidity_pass_count} / 100 tickers passed the scanner "
        "liquidity gate. The consumer is reading the normalized "
        "avg_volume_20d column (~1.0) instead of avg_volume_20d_raw "
        "(raw shares). Check data/scanner_orchestrator.py "
        "_build_technical_scores_from_feature_store and "
        "graph/research_graph.py::fetch_data_node."
    )


def test_scanner_fails_loud_when_raw_column_missing():
    """Transitional state: if the feature store has rows but no
    ``avg_volume_20d_raw`` field (e.g., before Phase 1's Saturday DataPhase1
    has populated S3), the scanner must fail the liquidity gate cleanly
    — NOT silently pass it.

    Per [[feedback_no_silent_fails]]: missing data => loud refuse, not
    quiet incorrect.
    """
    from data.scanner import run_quant_filter
    from data.scanner_orchestrator import _build_technical_scores_from_feature_store

    fs_rows = _synthetic_feature_store_rows(n_tickers=10)
    # Strip the raw column from every row — simulate pre-Phase-1 data.
    for row in fs_rows.values():
        row.pop("avg_volume_20d_raw")

    constituents = list(fs_rows.keys())
    sector_map = dict.fromkeys(constituents, "Technology")

    with (
        patch("data.fetchers.feature_store_reader.read_latest_features", return_value=fs_rows),
        patch("data.fetchers.feature_store_reader.read_latest_daily_closes",
              return_value=_synthetic_daily_closes(constituents)),
    ):
        technical_scores, _ = _build_technical_scores_from_feature_store(
            constituents, sector_map,
        )

    # The slot should be None for every ticker — not falling back to
    # the normalized column (which would silently fail under the old gate).
    for ticker, ts in technical_scores.items():
        assert ts["avg_volume_20d"] is None, (
            f"{ticker}: avg_volume_20d slot is {ts['avg_volume_20d']!r} "
            "when avg_volume_20d_raw is missing. Expected None — the "
            "consumer must not silently fall back to the normalized column."
        )

    run_quant_filter(
        tickers=constituents,
        price_data={},
        technical_scores=technical_scores,
        sector_map=sector_map,
    )
    eval_log = run_quant_filter._last_eval_log
    liquidity_fail_count = sum(
        1 for r in eval_log if r.get("liquidity_pass") == 0
    )
    assert liquidity_fail_count == 10, (
        f"Expected all 10 tickers to fail the liquidity gate (raw column "
        f"missing); got {liquidity_fail_count} fails. The scanner is "
        "silently passing tickers with None avg_volume_20d — fix the gate."
    )


def test_research_graph_consumer_slot_uses_raw_column():
    """Source-level invariant: the indicators dict built in
    ``fetch_data_node`` must source ``avg_volume_20d`` from
    ``fs_row.get("avg_volume_20d_raw")``.

    A grep-style check. Cheap, catches a revert PR even if the test above
    is somehow not exercising the exact code path.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    graph_path = os.path.join(repo_root, "graph", "research_graph.py")
    with open(graph_path, encoding="utf-8") as f:
        src = f.read()
    assert 'fs_row.get("avg_volume_20d_raw")' in src, (
        "graph/research_graph.py no longer reads avg_volume_20d_raw from "
        "the feature store. The Research scanner's liquidity gate will "
        "regress to silent zero-output. See SCHEMA.md."
    )
    # And the wrong-form must not appear in an indicators dict.
    bad_pattern = '"avg_volume_20d": fs_row.get("avg_volume_20d")'
    assert bad_pattern not in src, (
        "graph/research_graph.py contains a feature-store read that "
        "pulls the normalized avg_volume_20d into the scanner's "
        f"indicators slot ({bad_pattern!r}). This is the regression "
        "this contract test is meant to catch."
    )


def test_scanner_orchestrator_consumer_slot_uses_raw_column():
    """Same source-level invariant for the standalone scanner Lambda."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    orch_path = os.path.join(repo_root, "data", "scanner_orchestrator.py")
    with open(orch_path, encoding="utf-8") as f:
        src = f.read()
    assert 'fs_row.get("avg_volume_20d_raw")' in src, (
        "data/scanner_orchestrator.py no longer reads avg_volume_20d_raw "
        "from the feature store. The standalone scanner Lambda's "
        "liquidity gate will regress to silent zero-output."
    )
    bad_pattern = '"avg_volume_20d": fs_row.get("avg_volume_20d")'
    assert bad_pattern not in src, (
        "data/scanner_orchestrator.py contains the regression pattern: "
        f"{bad_pattern!r}."
    )
