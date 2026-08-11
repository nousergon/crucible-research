"""Source selection + staleness for the scanner's feature reader.

The live champion path must never reach the WEEKLY S3 snapshot by accident,
and must never rank on a stale ArcticDB read without saying so.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _restore_source_env():
    """Always restore SCANNER_FEATURE_SOURCE, even when a test fails.

    Without this a failing test leaves the override set and every later
    test in the session silently reads the wrong feature source.
    """
    import os

    old = os.environ.get("SCANNER_FEATURE_SOURCE")
    yield
    if old is None:
        os.environ.pop("SCANNER_FEATURE_SOURCE", None)
    else:
        os.environ["SCANNER_FEATURE_SOURCE"] = old


def _reload_with_source(value: str | None):
    """Return the reader module with SCANNER_FEATURE_SOURCE applied.

    NO importlib.reload: the module reads the env var at call time, so the
    setting can be varied by patching os.environ alone. An earlier version
    of this helper did reload, which swapped the module object out from
    under patches applied in OTHER test files and hard-aborted the suite at
    test_scanner_handler.py while passing when run alone.
    """
    import os

    import data.fetchers.feature_store_reader as mod

    if value is None:
        os.environ.pop("SCANNER_FEATURE_SOURCE", None)
    else:
        os.environ["SCANNER_FEATURE_SOURCE"] = value
    return mod


def _arctic_frame(last_date: str) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(last_date)])
    return pd.DataFrame({"rsi_14": [55.0], "momentum_20d": [0.03]}, index=idx)


def _batch_result(df):
    res = MagicMock()
    res.data = df
    return res


def test_ticker_list_selects_arcticdb_not_the_weekly_snapshot():
    mod = _reload_with_source(None)  # default
    lib = MagicMock()
    lib.read_batch.return_value = [_batch_result(_arctic_frame("2026-07-29"))]

    with patch.object(mod, "_connect_universe_lib", return_value=lib):
        out = mod.read_latest_features(["AAPL"], ref_date=pd.Timestamp("2026-07-29").date())

    assert out == {"AAPL": {"rsi_14": 55.0, "momentum_20d": 0.03}}
    lib.read_batch.assert_called_once()


def test_no_ticker_list_falls_back_to_the_weekly_snapshot():
    """graph/research_graph.py discovers its universe by iterating the return
    value, so it structurally cannot use a per-symbol store. It must degrade
    to S3 rather than raise — but never silently.
    """
    mod = _reload_with_source(None)
    with (
        patch.object(mod, "_read_latest_features_s3", return_value={"AAPL": {}}) as s3_read,
        patch.object(mod, "_connect_universe_lib") as arctic,
    ):
        out = mod.read_latest_features()

    assert out == {"AAPL": {}}
    s3_read.assert_called_once()
    arctic.assert_not_called()


def test_env_override_forces_the_weekly_snapshot_even_with_tickers():
    """The rollback lever has to beat the ticker-list selector, or it isn't one."""
    mod = _reload_with_source("s3")
    with (
        patch.object(mod, "_read_latest_features_s3", return_value={"AAPL": {}}) as s3_read,
        patch.object(mod, "_connect_universe_lib") as arctic,
    ):
        mod.read_latest_features(["AAPL"])

    s3_read.assert_called_once()
    arctic.assert_not_called()


def test_stale_arcticdb_rows_raise_instead_of_ranking_the_universe():
    mod = _reload_with_source(None)
    lib = MagicMock()
    # Newest row is ~3 months old — the daily producer has stopped writing.
    lib.read_batch.return_value = [_batch_result(_arctic_frame("2026-05-01"))]

    with (
        patch.object(mod, "_connect_universe_lib", return_value=lib),
        pytest.raises(mod.FeatureStoreStalenessError, match="trading days behind"),
    ):
        mod.read_latest_features(["AAPL"], ref_date=pd.Timestamp("2026-07-29").date())


def test_read_failure_rate_above_the_ceiling_raises():
    """A near-total read failure must not look like a small universe."""
    mod = _reload_with_source(None)
    lib = MagicMock()
    lib.read_batch.return_value = [_batch_result(None) for _ in range(10)]

    with (
        patch.object(mod, "_connect_universe_lib", return_value=lib),
        pytest.raises(mod.FeatureStoreStalenessError, match="read-failure rate"),
    ):
        mod.read_latest_features(
            [f"T{i}" for i in range(10)],
            ref_date=pd.Timestamp("2026-07-29").date(),
        )


def test_only_the_columns_some_consumer_needs_are_projected():
    """A full-row read over ~900 symbols is the difference between a
    comfortable and a timed-out scanner Lambda.

    The projection is the UNION of the two consumers' column sets, not one
    caller's (alpha-engine-config-I6855) — the union costs four extra columns
    on one read and saves a whole round-trip over the universe. It is still a
    bounded projection, which is what this test exists to hold: the guard is
    against reading the full row, never against reading a second consumer's
    columns.
    """
    mod = _reload_with_source(None)
    lib = MagicMock()
    lib.read_batch.return_value = [_batch_result(_arctic_frame("2026-07-29"))]

    with patch.object(mod, "_connect_universe_lib", return_value=lib):
        mod.read_latest_features(["AAPL"], ref_date=pd.Timestamp("2026-07-29").date())

    requested = lib.read_batch.call_args[0][0][0]
    assert set(requested.columns) == set(mod._ARCTIC_UNION_COLS)
    assert set(requested.columns) == set(mod._TECHNICAL_COLS) | set(mod._FACTOR_TECHNICAL_COLS)
    assert "avg_volume_20d_raw" in requested.columns, (
        "the scanner's MIN_AVG_VOLUME gate needs raw shares, not the normalized avg_volume_20d ratio"
    )


def test_factor_loadings_honour_the_same_source_selection():
    mod = _reload_with_source(None)
    lib = MagicMock()
    idx = pd.DatetimeIndex([pd.Timestamp("2026-07-29")])
    lib.read_batch.return_value = [_batch_result(pd.DataFrame({"momentum_20d_zscore": [1.2]}, index=idx))]

    with patch.object(mod, "_connect_universe_lib", return_value=lib):
        out = mod.read_latest_factor_loadings(
            tickers=["AAPL"],
            ref_date=pd.Timestamp("2026-07-29").date(),
        )

    assert out == {"AAPL": {"momentum_20d_zscore": 1.2}}


# ── One union read per run (alpha-engine-config-I6855) ────────────────────
#
# A Scanner run reads technical features TWICE over the same ~900
# constituents with different column sets — scanner_orchestrator for
# tech_score, factor_scoring for the factor pillar. Each was a full
# read_batch round-trip: 59.3s + 39.9s measured on 2026-08-10, and 106s for
# the first alone on 2026-08-11 when the second never ran because the Lambda
# hit its 300s ceiling and the preopen pipeline terminated DEGRADED.


def _union_frame(last_date: str) -> pd.DataFrame:
    """A row carrying columns from BOTH callers' sets, so projection is real."""
    idx = pd.DatetimeIndex([pd.Timestamp(last_date)])
    return pd.DataFrame(
        {
            "rsi_14": [55.0],           # scanner only
            "momentum_20d": [0.03],     # both
            "return_60d": [0.11],       # factor only
        },
        index=idx,
    )


def test_second_reader_call_in_a_run_issues_no_second_read_batch():
    """The factor pillar's read is served from the scanner's union read."""
    mod = _reload_with_source(None)
    lib = MagicMock()
    lib.read_batch.return_value = [_batch_result(_union_frame("2026-07-29"))]
    ref = pd.Timestamp("2026-07-29").date()

    with patch.object(mod, "_connect_universe_lib", return_value=lib):
        mod.read_latest_features(["AAPL"], ref_date=ref)
        mod.read_latest_features(["AAPL"], columns=("return_60d",), ref_date=ref)

    assert lib.read_batch.call_count == 1


def test_the_single_read_projects_the_union_not_one_callers_columns():
    mod = _reload_with_source(None)
    lib = MagicMock()
    lib.read_batch.return_value = [_batch_result(_union_frame("2026-07-29"))]

    with patch.object(mod, "_connect_universe_lib", return_value=lib):
        mod.read_latest_features(["AAPL"], ref_date=pd.Timestamp("2026-07-29").date())

    requested = list(lib.read_batch.call_args[0][0])[0].columns
    assert set(mod._TECHNICAL_COLS) <= set(requested)
    assert set(mod._FACTOR_TECHNICAL_COLS) <= set(requested)


def test_each_caller_still_receives_only_the_columns_it_asked_for():
    """The widening is invisible above this module — no caller sees the union."""
    mod = _reload_with_source(None)
    lib = MagicMock()
    lib.read_batch.return_value = [_batch_result(_union_frame("2026-07-29"))]
    ref = pd.Timestamp("2026-07-29").date()

    with patch.object(mod, "_connect_universe_lib", return_value=lib):
        scanner_rows = mod.read_latest_features(["AAPL"], ref_date=ref)
        factor_rows = mod.read_latest_features(["AAPL"], columns=("return_60d",), ref_date=ref)

    assert set(scanner_rows["AAPL"]) == {"rsi_14", "momentum_20d"}
    assert set(factor_rows["AAPL"]) == {"return_60d"}


def test_a_ticker_left_with_no_requested_column_is_dropped_not_emptied():
    """Preserves _read_arctic_latest's contract under the wider read.

    Before the union, a ticker carrying only the OTHER caller's columns
    never appeared in this caller's result. It must still not appear — an
    empty feature dict is a different value than a missing key, and
    scanner_orchestrator branches on presence.
    """
    mod = _reload_with_source(None)
    idx = pd.DatetimeIndex([pd.Timestamp("2026-07-29")])
    factor_only = pd.DataFrame({"return_60d": [0.11]}, index=idx)
    lib = MagicMock()
    lib.read_batch.return_value = [_batch_result(factor_only)]

    with patch.object(mod, "_connect_universe_lib", return_value=lib):
        rows = mod.read_latest_features(["AAPL"], ref_date=pd.Timestamp("2026-07-29").date())

    assert rows is None  # no ticker survived the projection → empty → None


def test_a_different_ref_date_is_a_miss_not_a_stale_hit():
    """An operator replay sharing a warm container must not read the wrong day."""
    mod = _reload_with_source(None)
    lib = MagicMock()
    lib.read_batch.side_effect = [
        [_batch_result(_union_frame("2026-07-29"))],
        [_batch_result(_union_frame("2026-07-28"))],
    ]

    with patch.object(mod, "_connect_universe_lib", return_value=lib):
        mod.read_latest_features(["AAPL"], ref_date=pd.Timestamp("2026-07-29").date())
        mod.read_latest_features(["AAPL"], ref_date=pd.Timestamp("2026-07-28").date())

    assert lib.read_batch.call_count == 2


def test_a_different_universe_is_a_miss():
    mod = _reload_with_source(None)
    lib = MagicMock()
    lib.read_batch.side_effect = [
        [_batch_result(_union_frame("2026-07-29"))],
        [_batch_result(_union_frame("2026-07-29"))],
    ]
    ref = pd.Timestamp("2026-07-29").date()

    with patch.object(mod, "_connect_universe_lib", return_value=lib):
        mod.read_latest_features(["AAPL"], ref_date=ref)
        mod.read_latest_features(["MSFT"], ref_date=ref)

    assert lib.read_batch.call_count == 2


def test_the_memo_holds_one_entry_so_a_warm_container_cannot_accumulate():
    mod = _reload_with_source(None)
    lib = MagicMock()
    lib.read_batch.side_effect = [
        [_batch_result(_union_frame("2026-07-29"))],
        [_batch_result(_union_frame("2026-07-28"))],
    ]

    with patch.object(mod, "_connect_universe_lib", return_value=lib):
        mod.read_latest_features(["AAPL"], ref_date=pd.Timestamp("2026-07-29").date())
        mod.read_latest_features(["AAPL"], ref_date=pd.Timestamp("2026-07-28").date())

    assert len(mod._ARCTIC_UNION_CACHE) == 1


def test_factor_scoring_reads_its_column_list_from_the_reader():
    """One declaration. A second copy would drift the union silently."""
    import scoring.factor_scoring as fs

    from data.fetchers import feature_store_reader as mod

    assert fs._FACTOR_TECHNICAL_COLS is mod._FACTOR_TECHNICAL_COLS
