"""Consumer-side contract test: `scanner_predictor_direct` reads the predictor's
DAILY research-free artifact (alpha-engine-config-I10067).

The producer is crucible-predictor `inference/stages/research_free.py` ->
`inference/research_free_inference.py::run_research_free_inference`, which writes
`predictor/predictions_research_free/{date}.json` (+ `latest.json`) once per
trading day in the weekday PredictorInference Lambda. Producer-side field set is
pinned there by `tests/test_research_free_producer_contract.py`; this file pins
the consumer half against a fixture built to that exact schema.

WHY THIS TEST EXISTS
--------------------
Until 2026-09-05 `load_research_free_pool` read crucible-backtester's
`predictor/research_free_backfill/predictor_outcomes_research_free.parquet`
instead. That parquet is written by the **Backtester** stage of the weekly Step
Function, AFTER the ResearchPredictorParallel join; this arm runs inside Branch
A's ChallengerShadow, BEFORE it. The live arm was therefore reading, mid-run, a
file the same run had not written yet, and failed on every canonical Saturday
(measured on `watch-rerun-2026-09-04-1/2/3`; it passed only on rerun 4, after a
Backtester had run). Under `alpha-engine-config-I6891` a failed ChallengerShadow
arm fails the whole weekly SF.

The parquet path is GONE from this module. `test_the_offline_backtester_parquet_
is_not_read_by_the_live_arm` keeps it gone: an offline analysis artifact written
later in the same run can never be a live arm's input.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from producers.filling_arms import (
    PREDICTIONS_RESEARCH_FREE_KEY,
    FillingShadowError,
    load_research_free_pool,
)

_RUN_DATE = "2026-09-11"  # a Friday: the weekly arm's run_date is the last trading day


# ── The producer's schema, reproduced exactly ────────────────────────────────

def _producer_envelope(run_date: str, rows: list[tuple[str, float]]) -> dict:
    """Byte-for-byte the envelope crucible-predictor
    `inference/research_free_inference.py::run_research_free_inference` writes.

    Reproduced as a literal rather than imported: the producer lives in another
    repo and is not installed here, so a fixture is the only way to bind the
    consumer to the producer's shape. Keep this in step with
    crucible-predictor `tests/test_research_free_producer_contract.py`.
    """
    return {
        "schema_version": 1,
        "date": run_date,
        "generated_at": f"{run_date}T13:20:00+00:00",
        "n_predictions": len(rows),
        "n_scanner_pool": len(rows),
        "n_errors": 0,
        "feature_names": ["momentum_score", "expected_move", "macro_vix_level"],
        "n_research_features_missing": 4,
        "predictions": [
            {
                "ticker": t,
                "prediction_date": run_date,
                "predicted_alpha": a,
                "n_research_features_missing": 4,
            }
            for t, a in rows
        ],
    }


class _FakeS3:
    def __init__(self, docs: dict):
        self.docs = docs
        self.reads: list[str] = []

    def get_object(self, Bucket, Key):  # noqa: N803 — boto3 signature
        self.reads.append(Key)
        if Key not in self.docs:
            from botocore.exceptions import ClientError

            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": _Body(self.docs[Key])}


class _Body:
    def __init__(self, doc):
        self._doc = doc

    def read(self):
        return json.dumps(self._doc).encode()


# ── The contract ─────────────────────────────────────────────────────────────

def test_the_arm_reads_the_daily_research_free_artifact_for_its_own_run_date():
    key = PREDICTIONS_RESEARCH_FREE_KEY.format(date=_RUN_DATE)
    s3 = _FakeS3({key: _producer_envelope(_RUN_DATE, [("AAA", 0.01), ("BBB", 0.03)])})

    ranked, pool_source = load_research_free_pool(s3, "bucket", _RUN_DATE)

    assert s3.reads == [key], (
        "the arm must read exactly the run_date's daily artifact — no other key, "
        "and never a whole-history file it has to filter by date."
    )
    assert pool_source == "predictions_research_free", (
        "pool_source is what alpha-engine-config-I10067's closes-when predicate "
        "reads off arm_pool; changing it silently invalidates that predicate."
    )
    assert [t for t, _ in ranked] == ["BBB", "AAA"], "ranked by predicted_alpha desc"


def test_the_ranking_is_predicted_alpha_descending():
    rows = [("LOW", -0.02), ("MID", 0.00), ("TOP", 0.05)]
    key = PREDICTIONS_RESEARCH_FREE_KEY.format(date=_RUN_DATE)
    s3 = _FakeS3({key: _producer_envelope(_RUN_DATE, rows)})

    ranked, _ = load_research_free_pool(s3, "bucket", _RUN_DATE)
    assert [t for t, _ in ranked] == ["TOP", "MID", "LOW"]


def test_a_missing_artifact_raises_and_names_the_producer():
    """§3: a filling arm that silently skips a cohort date is the miss/broken
    conflation. And the message must name the producer — this arm's entire
    history is a consumer pointed at the wrong artifact."""
    s3 = _FakeS3({})
    with pytest.raises(FillingShadowError) as exc:
        load_research_free_pool(s3, "bucket", _RUN_DATE)

    msg = str(exc.value)
    assert PREDICTIONS_RESEARCH_FREE_KEY.format(date=_RUN_DATE) in msg
    assert "crucible-predictor" in msg
    assert "run_research_free_inference" in msg


def test_yesterdays_artifact_does_not_satisfy_todays_run_date():
    """The parquet path silently offered the arm a stale cohort to filter and
    fail on. A dated key cannot: a run_date with no artifact is simply absent,
    and absent RAISES."""
    stale = (date.fromisoformat(_RUN_DATE) - timedelta(days=7)).isoformat()
    s3 = _FakeS3({
        PREDICTIONS_RESEARCH_FREE_KEY.format(date=stale):
            _producer_envelope(stale, [("AAA", 0.01)]),
    })
    with pytest.raises(FillingShadowError):
        load_research_free_pool(s3, "bucket", _RUN_DATE)


def test_an_artifact_with_no_usable_alpha_raises_rather_than_reporting_health():
    key = PREDICTIONS_RESEARCH_FREE_KEY.format(date=_RUN_DATE)
    empty = _producer_envelope(_RUN_DATE, [])
    s3 = _FakeS3({key: empty})
    with pytest.raises(FillingShadowError) as exc:
        load_research_free_pool(s3, "bucket", _RUN_DATE)
    assert "no usable predicted_alpha" in str(exc.value)


def test_a_nan_alpha_is_dropped_not_ranked():
    key = PREDICTIONS_RESEARCH_FREE_KEY.format(date=_RUN_DATE)
    doc = _producer_envelope(_RUN_DATE, [("AAA", 0.01), ("BAD", float("nan"))])
    s3 = _FakeS3({key: doc})
    ranked, _ = load_research_free_pool(s3, "bucket", _RUN_DATE)
    assert [t for t, _ in ranked] == ["AAA"]


def test_both_predictor_fed_arms_read_the_predictor_through_one_shape_reader():
    """`scanner_predictor_direct` and `scanner_top20_predictor` now consume the
    same producer, on the same daily cadence, through the same JSON envelope
    shape — the divergence between them was the whole defect."""
    import inspect

    from producers import filling_arms

    src = inspect.getsource(filling_arms.load_research_free_pool)
    assert "_predictions_by_ticker" in src, (
        "load_research_free_pool must reuse the shared predictions shape reader "
        "rather than re-implementing a second parser for the same producer."
    )


def test_the_offline_backtester_parquet_is_not_read_by_the_live_arm():
    """The backtester's `predictor/research_free_backfill/
    predictor_outcomes_research_free.parquet` is an OFFLINE lift-analysis
    artifact written after the ResearchPredictorParallel join, later in the same
    weekly run than this arm. It can never be a live arm's input again."""
    import inspect

    from producers import filling_arms

    src = inspect.getsource(filling_arms)
    live = src.split("def load_research_free_pool", 1)[1].split("\ndef ", 1)[0]
    assert "research_free_backfill" not in live
    assert "read_parquet" not in live
    assert not hasattr(filling_arms, "RESEARCH_FREE_PARQUET_KEY"), (
        "RESEARCH_FREE_PARQUET_KEY is gone — the live arm reads the predictor's "
        "daily artifact (alpha-engine-config-I10067)."
    )


def test_the_consumed_key_matches_the_producers_declared_key():
    """Cross-repo literal, kept as a literal by the same convention the rest of
    this module uses (a stable S3 contract, not shared code). It must equal
    crucible-predictor `config.py::PREDICTIONS_RESEARCH_FREE_KEY`."""
    assert PREDICTIONS_RESEARCH_FREE_KEY == (
        "predictor/predictions_research_free/{date}.json"
    )
