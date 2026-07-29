"""Per-group feature-snapshot resolution for the daily scanner.

Guards the chokepoint that made a weekday scanner run impossible: before
this, ``compute_and_write_factor_profiles`` read
``features/{run_date}/{group}.parquet`` by EXACT key with no fallback, so a
non-Saturday ``run_date`` raised ``NoSuchKey``, the caller's fail-soft
``except`` swallowed it, and the ``attractiveness_top_20`` champion cut
resolved empty.

The two groups have genuinely different cadences — technical is rewritten
every trading day, fundamental is collected weekly — so the fix is per-group
latest-on-or-before resolution with per-group staleness bounds, NOT a single
shared "close enough" window.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from scoring.factor_scoring import (
    _MAX_FUNDAMENTAL_STALENESS_TD,
    _MAX_TECHNICAL_STALENESS_TD,
    FactorSnapshotStalenessError,
    _list_feature_snapshot_dates,
    _resolve_group_snapshot,
)

RUN_DATE = "2026-07-29"  # a Wednesday — the case the old code could not serve


def _parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow", index=False)
    return buf.getvalue()


def _s3_with(objects: dict[str, pd.DataFrame]) -> MagicMock:
    """MagicMock S3 serving exactly ``{key: frame}``, 404-ing everything else."""
    s3 = MagicMock()

    dates = sorted({k.split("/")[1] for k in objects})
    s3.list_objects_v2.return_value = {
        "CommonPrefixes": [{"Prefix": f"features/{d}/"} for d in dates],
        "IsTruncated": False,
    }

    def _get(Bucket, Key):  # noqa: N803 — boto3 kwarg casing
        if Key not in objects:
            raise KeyError(f"NoSuchKey: {Key}")
        return {"Body": io.BytesIO(_parquet_bytes(objects[Key]))}

    s3.get_object.side_effect = _get
    return s3


FRAME = pd.DataFrame({"ticker": ["AAPL", "MSFT"], "value": [1.0, 2.0]})


def test_groups_resolve_independently_technical_today_fundamental_carried_forward():
    """The whole point: today's technical + last Saturday's fundamental.

    Against the pre-change code this scenario raised NoSuchKey, because
    features/2026-07-29/fundamental.parquet does not exist.
    """
    s3 = _s3_with(
        {
            f"features/{RUN_DATE}/technical.parquet": FRAME,
            "features/2026-07-25/technical.parquet": FRAME,
            "features/2026-07-25/fundamental.parquet": FRAME,
        }
    )
    dates = _list_feature_snapshot_dates(s3, "b")

    tech_df, tech_date = _resolve_group_snapshot(
        s3,
        "b",
        "technical",
        RUN_DATE,
        _MAX_TECHNICAL_STALENESS_TD,
        candidate_dates=dates,
    )
    fund_df, fund_date = _resolve_group_snapshot(
        s3,
        "b",
        "fundamental",
        RUN_DATE,
        _MAX_FUNDAMENTAL_STALENESS_TD,
        candidate_dates=dates,
    )

    assert tech_date == RUN_DATE, "technical must use today's snapshot, not carry forward"
    assert fund_date == "2026-07-25", "fundamental must carry forward from the weekly snapshot"
    assert not tech_df.empty and not fund_df.empty


def test_a_partition_missing_the_group_is_skipped_not_fatal():
    """A partition can exist for one group and not the other.

    Weekday MorningEnrich territory: technical partitions appear daily while
    fundamental only lands on Saturdays, so resolution must walk BACK past
    partitions that lack the requested group rather than stopping at the
    newest partition name.
    """
    s3 = _s3_with(
        {
            f"features/{RUN_DATE}/technical.parquet": FRAME,
            "features/2026-07-28/technical.parquet": FRAME,
            "features/2026-07-25/fundamental.parquet": FRAME,
        }
    )
    dates = _list_feature_snapshot_dates(s3, "b")

    _, fund_date = _resolve_group_snapshot(
        s3,
        "b",
        "fundamental",
        RUN_DATE,
        _MAX_FUNDAMENTAL_STALENESS_TD,
        candidate_dates=dates,
    )
    assert fund_date == "2026-07-25"


def test_carry_forward_beyond_the_bound_raises_rather_than_ranking_on_it():
    """Carry-forward is bounded. Silently ranking ~900 names on a fundamental
    snapshot months old would trade the staleness this change removes for a
    worse one that no longer shows up on the calendar.
    """
    s3 = _s3_with(
        {
            f"features/{RUN_DATE}/technical.parquet": FRAME,
            "features/2026-05-01/fundamental.parquet": FRAME,
        }
    )
    dates = _list_feature_snapshot_dates(s3, "b")

    with pytest.raises(FactorSnapshotStalenessError, match="fundamental"):
        _resolve_group_snapshot(
            s3,
            "b",
            "fundamental",
            RUN_DATE,
            _MAX_FUNDAMENTAL_STALENESS_TD,
            candidate_dates=dates,
        )


def test_technical_carries_a_tighter_bound_than_fundamental():
    """Per-group bounds, not one shared window — a technical snapshot as old
    as an acceptable fundamental one means the daily producer has stalled.
    """
    assert _MAX_TECHNICAL_STALENESS_TD < _MAX_FUNDAMENTAL_STALENESS_TD

    s3 = _s3_with({"features/2026-05-01/technical.parquet": FRAME})
    dates = _list_feature_snapshot_dates(s3, "b")

    with pytest.raises(FactorSnapshotStalenessError, match="technical"):
        _resolve_group_snapshot(
            s3,
            "b",
            "technical",
            RUN_DATE,
            _MAX_TECHNICAL_STALENESS_TD,
            candidate_dates=dates,
        )


def test_future_dated_partitions_are_never_used():
    """Resolution is on-or-before. A partition dated after run_date is a
    lookahead leak into the ranking that produces the traded universe.
    """
    s3 = _s3_with(
        {
            "features/2026-07-25/technical.parquet": FRAME,
            "features/2026-08-15/technical.parquet": FRAME,
        }
    )
    dates = _list_feature_snapshot_dates(s3, "b")

    _, resolved = _resolve_group_snapshot(
        s3,
        "b",
        "technical",
        "2026-07-27",
        _MAX_TECHNICAL_STALENESS_TD,
        candidate_dates=dates,
    )
    assert resolved == "2026-07-25"


def test_no_snapshot_at_all_raises_named_error():
    s3 = _s3_with({"features/2026-07-25/technical.parquet": FRAME})
    dates = _list_feature_snapshot_dates(s3, "b")

    with pytest.raises(FactorSnapshotStalenessError, match="no features"):
        _resolve_group_snapshot(
            s3,
            "b",
            "fundamental",
            RUN_DATE,
            _MAX_FUNDAMENTAL_STALENESS_TD,
            candidate_dates=dates,
        )


def test_snapshot_listing_is_paginated():
    """The features/ prefix has outlived a single 1000-key page."""
    s3 = MagicMock()
    s3.list_objects_v2.side_effect = [
        {
            "CommonPrefixes": [{"Prefix": "features/2026-07-25/"}],
            "IsTruncated": True,
            "NextContinuationToken": "t1",
        },
        {"CommonPrefixes": [{"Prefix": "features/2026-07-29/"}], "IsTruncated": False},
    ]
    assert _list_feature_snapshot_dates(s3, "b") == ["2026-07-25", "2026-07-29"]


def test_provenance_sidecar_records_each_group_resolved_date():
    """Which snapshot produced a ranking must be recoverable from an artifact,
    not inferred from the calendar — the calendar stopped being a reliable
    proxy the moment carry-forward became possible.
    """
    from scoring.factor_scoring import write_factor_profiles_to_s3

    profiles = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "sector": ["Technology"],
            "quality_score": [0.5],
        }
    )
    mock_s3 = MagicMock()
    provenance = {
        "run_date": RUN_DATE,
        "groups": {
            "technical": {"resolved_date": RUN_DATE, "carried_forward": False},
            "fundamental": {"resolved_date": "2026-07-25", "carried_forward": True},
        },
    }
    with patch("boto3.client", return_value=mock_s3):
        write_factor_profiles_to_s3(
            profiles,
            RUN_DATE,
            bucket="test-bucket",
            provenance=provenance,
        )

    keys = [c.kwargs["Key"] for c in mock_s3.put_object.call_args_list]
    assert f"factors/profiles/{RUN_DATE}/provenance.json" in keys

    import json

    idx = keys.index(f"factors/profiles/{RUN_DATE}/provenance.json")
    body = json.loads(mock_s3.put_object.call_args_list[idx].kwargs["Body"])
    assert body["groups"]["fundamental"]["carried_forward"] is True
    assert body["groups"]["technical"]["carried_forward"] is False


def test_by_ticker_payload_shape_is_unchanged_by_provenance():
    """provenance rides in its own object; by_ticker.json stays a flat
    ticker->record mapping every consumer iterates.
    """
    from scoring.factor_scoring import write_factor_profiles_to_s3

    profiles = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "sector": ["Technology"],
            "quality_score": [0.5],
        }
    )
    mock_s3 = MagicMock()
    with patch("boto3.client", return_value=mock_s3):
        write_factor_profiles_to_s3(
            profiles,
            RUN_DATE,
            bucket="test-bucket",
            provenance={"run_date": RUN_DATE, "groups": {}},
        )

    import json

    calls = {c.kwargs["Key"]: c.kwargs["Body"] for c in mock_s3.put_object.call_args_list}
    payload = json.loads(calls[f"factors/profiles/{RUN_DATE}/by_ticker.json"])
    assert set(payload) == {"AAPL"}
    assert "run_date" not in payload
