"""The producer must accept exactly what ``now_dual()`` hands its only caller.

alpha-engine-config-I8177. ``scripts/build_agent_quality.build_agent_quality``
was annotated ``target_date: date`` / ``run_date: date`` and immediately called
``target_date.isoformat()``. Its only production caller,
``lambda/eval_rolling_mean_handler.py``, passed ``now_dual().trading_day`` and
``now_dual().calendar_date`` — which ``krepis.dates.now_dual()`` returns as ISO
**strings**, not ``date`` objects.

Python does not enforce annotations at runtime, so nothing caught it. Measured
in ``/aws/lambda/alpha-engine-research-eval-rolling-mean``, every weekly run
from the wiring merge (crucible-research#304, 2026-06-23) to 2026-08-22:

    WARNING [eval_rolling_mean_handler] agent_quality build failed
    (non-fatal): 'str' object has no attribute 'isoformat'

``backtest/{date}/agent_quality.json`` consequently existed on NO date anywhere
in the bucket, and eight report-card components graded ``N/A-MISSING-INPUT``
for two months: ``agent_validation_failure_rate``, ``cost_per_signal``,
``retry_storm_count``, ``agent_latency_p95``, ``judge_rubric_distribution``
(agent tile) and ``judge_rubric_pass_rate``, ``signal_volume_adequacy``,
``judge_outcome_ic`` (research tile).

The unit suite could not catch it because every existing test constructs
``date(2026, 6, 12)`` by hand — it exercised a carrier type the caller never
uses. These tests drive the producer with the REAL carrier the fleet's
canonical date helper emits, so the contract is pinned at the seam that
actually broke rather than at a convenient one.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import boto3
import pytest
from moto import mock_aws

from krepis.dates import now_dual
from scripts.build_agent_quality import _coerce_date, build_agent_quality

_BUCKET = "alpha-engine-research"


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=_BUCKET)
        yield client


# --------------------------------------------------------------------------
# The seam that broke: what now_dual() actually returns.
# --------------------------------------------------------------------------

def test_now_dual_really_returns_strings_not_dates() -> None:
    """Pin the upstream fact this whole bug rests on.

    If ``krepis.dates`` ever changes ``DualDate`` to carry ``date`` objects,
    this test fails and tells the reader the coercion below is now redundant —
    rather than leaving a silent, permanently-unexercised branch behind.
    """
    dual = now_dual()
    assert isinstance(dual.trading_day, str)
    assert isinstance(dual.calendar_date, str)


def test_producer_accepts_the_now_dual_carrier_verbatim(s3) -> None:
    """The exact call the weekly Lambda makes. This raised TypeError before."""
    dual = now_dual()
    artifact = build_agent_quality(
        s3, _BUCKET, dual.trading_day, run_date=dual.calendar_date,
    )
    assert artifact["status"] == "ok"
    assert artifact["date"] == dual.trading_day
    assert artifact["run_date"] == dual.calendar_date


def test_producer_still_accepts_date_objects(s3) -> None:
    """Widening the carrier must not break the CLI / test path."""
    artifact = build_agent_quality(
        s3, _BUCKET, date(2026, 6, 12), run_date=date(2026, 6, 13),
    )
    assert artifact["date"] == "2026-06-12"
    assert artifact["run_date"] == "2026-06-13"


def test_mixed_carriers_normalize_consistently(s3) -> None:
    a = build_agent_quality(s3, _BUCKET, "2026-06-12", run_date=date(2026, 6, 13))
    b = build_agent_quality(s3, _BUCKET, date(2026, 6, 12), run_date="2026-06-13")
    assert a["date"] == b["date"] == "2026-06-12"
    assert a["run_date"] == b["run_date"] == "2026-06-13"


def test_run_date_defaults_to_target_date_for_both_carriers(s3) -> None:
    for carrier in ("2026-06-12", date(2026, 6, 12)):
        artifact = build_agent_quality(s3, _BUCKET, carrier)
        assert artifact["date"] == artifact["run_date"] == "2026-06-12"


# --------------------------------------------------------------------------
# Coercion is a NARROWING contract, not a shrug — bad input still fails loud.
# --------------------------------------------------------------------------

def test_garbage_string_raises_naming_the_field() -> None:
    with pytest.raises(ValueError, match="target_date"):
        _coerce_date("not-a-date", "target_date")


def test_datetime_is_rejected_not_silently_truncated() -> None:
    """A ``datetime`` is a different quantity; truncating it hides a real bug."""
    with pytest.raises(TypeError, match="target_date"):
        _coerce_date(datetime(2026, 6, 12, 13, 0, tzinfo=timezone.utc), "target_date")


@pytest.mark.parametrize("bad", [None, 20260612, 1749686400.0, ["2026-06-12"]])
def test_non_date_types_raise_typeerror(bad) -> None:
    with pytest.raises(TypeError, match="run_date"):
        _coerce_date(bad, "run_date")


def test_date_subclass_is_preserved() -> None:
    assert _coerce_date(date(2026, 6, 12), "target_date") == date(2026, 6, 12)
