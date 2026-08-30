"""Bound + single-scan guards for the ``decision_artifacts/_eval/`` walk
(alpha-engine-config-I9205).

``backtest/{date}/agent_quality.json`` has NEVER been written. After the
``'str' has no attribute 'isoformat'`` producer bug (I8177) was fixed, its
first three real executions all TIMED OUT against the 240s block ceiling in
``lambda/eval_rolling_mean_handler.py``. The driver was an unbounded,
DUPLICATED N+1 S3 scan: ``build_agent_quality`` walked the whole ``_eval/``
prefix issuing one sequential ``GetObject`` per artifact, and then
``build_judge_outcome_ic_block`` walked it a second time with no cache
between them. Measured live 2026-08-28: 4,679 artifacts / 26,058,127 bytes,
~117.8 ms per sequential ``get_object`` → ~551s per scan, ~1,102s for both,
against a 240s ceiling — and neither scan emitted a single log line, so the
timeout was unattributable from CloudWatch.

These tests fail if any of that reappears:
- the eval prefix is listed and fetched EXACTLY ONCE per artifact build;
- the object-count ceiling raises BEFORE any ``GetObject`` is issued;
- the ceiling is live on the Lambda handler path (not just on the loader);
- no caller can pin or disable either bound (both resolve from the module
  constants at call time).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime, timedelta

import boto3
import pytest
from moto import mock_aws

import evals.judge_outcome_ic as jic
from evals.judge_outcome_ic import (
    DEFAULT_EVAL_LOOKBACK_DAYS,
    MAX_EVAL_OBJECTS,
    EvalScanBudgetExceeded,
    build_judge_outcome_ic_block,
    load_eval_artifacts,
)
from scripts.build_agent_quality import build_agent_quality

_BUCKET = "alpha-engine-research"
_EVAL_PREFIX = "decision_artifacts/_eval/"
_DATE = "2026-06-12"
_RUN_DATE = "2026-06-13"

_CAP1, _TD1 = "2026-06-06", "2026-06-05"
_CAP2, _TD2 = "2026-06-13", "2026-06-12"

_OUTCOMES_DDL = """
CREATE TABLE score_performance_outcomes (
    id             INTEGER PRIMARY KEY,
    signal_id      TEXT NOT NULL,
    symbol         TEXT NOT NULL,
    score_date     TEXT NOT NULL,
    horizon_days   INTEGER NOT NULL,
    beat_spy       INTEGER,
    stock_return   REAL,
    spy_return     REAL,
    log_alpha      REAL,
    is_primary     INTEGER NOT NULL,
    resolved_at    TEXT NOT NULL,
    schema_version INTEGER NOT NULL DEFAULT 1,
    UNIQUE(signal_id, horizon_days)
);
"""


class _CountingS3:
    """Transparent proxy over a boto3 S3 client that counts, per prefix, how
    many ``list_objects_v2`` paginations and ``get_object`` calls are made.
    Thread-safe — the loader fetches on a bounded pool."""

    def __init__(self, inner):
        self._inner = inner
        self._lock = threading.Lock()
        self.gets: list[str] = []
        self.lists: list[str] = []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def get_object(self, **kwargs):
        with self._lock:
            self.gets.append(kwargs["Key"])
        return self._inner.get_object(**kwargs)

    def get_paginator(self, name):
        inner_paginator = self._inner.get_paginator(name)
        if name != "list_objects_v2":
            return inner_paginator
        outer = self

        class _P:
            def paginate(self, **kwargs):
                with outer._lock:
                    outer.lists.append(kwargs.get("Prefix", ""))
                return inner_paginator.paginate(**kwargs)

        return _P()

    def eval_gets(self) -> list[str]:
        return [k for k in self.gets if k.startswith(_EVAL_PREFIX)]

    def eval_lists(self) -> list[str]:
        return [p for p in self.lists if p.startswith(_EVAL_PREFIX)]


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=_BUCKET)
        yield _CountingS3(client)


def _put_json(s3, key, obj):
    s3.put_object(Bucket=_BUCKET, Key=key, Body=json.dumps(obj).encode())


def _thesis_eval(ticker, capture_date, score):
    agent_id = f"thesis_update:technology:{ticker}"
    y, m, d = capture_date.split("-")
    return {
        "schema_version": 2, "run_id": capture_date, "judge_run_id": "2606131230",
        "timestamp": f"{capture_date}T13:00:00Z", "judged_agent_id": agent_id,
        "judged_artifact_s3_key": f"decision_artifacts/{y}/{m}/{d}/run1/{agent_id}.json",
        "rubric_id": "eval_rubric_thesis_update", "rubric_version": "1.0.0",
        "judge_model": "claude-haiku-4-5",
        "dimension_scores": [
            {"dimension": "depth", "score": score, "reasoning": "r"},
            {"dimension": "grounding", "score": score, "reasoning": "r"},
        ],
        "overall_reasoning": "ok", "judge_skip_reason": None,
    }


def _seed_evals(s3) -> int:
    """Flat canonical ``_eval/`` layout — 6 attributable evals plus the
    ``latest.json`` sidecar (which is skipped by name, not fetched).
    Returns the number of real eval objects."""
    scores = {(_CAP1, "AAA"): 5, (_CAP1, "BBB"): 3, (_CAP1, "CCC"): 1,
              (_CAP2, "AAA"): 5, (_CAP2, "BBB"): 4, (_CAP2, "CCC"): 1}
    for (cap, ticker), score in scores.items():
        doc = _thesis_eval(ticker, cap, score)
        _put_json(
            s3,
            f"{_EVAL_PREFIX}2606131230_{doc['judged_agent_id']}.{cap}"
            f".claude-haiku-4-5.json",
            doc,
        )
    _put_json(s3, f"{_EVAL_PREFIX}latest.json", {"artifact_key": "x"})
    return len(scores)


def _outcomes_conn():
    conn = sqlite3.connect(":memory:")
    conn.executescript(_OUTCOMES_DDL)
    alphas = {("AAA", _TD1): 0.10, ("BBB", _TD1): 0.02, ("CCC", _TD1): -0.05,
              ("AAA", _TD2): 0.01, ("BBB", _TD2): 0.03, ("CCC", _TD2): -0.02}
    for i, ((sym, sd), alpha) in enumerate(alphas.items()):
        conn.execute(
            "INSERT INTO score_performance_outcomes (signal_id, symbol, score_date,"
            " horizon_days, beat_spy, stock_return, spy_return, log_alpha,"
            " is_primary, resolved_at) VALUES (?,?,?,21,?,0.01,0.005,?,1,'2026-07-06')",
            (f"sig{i}", sym, sd, 1 if alpha > 0 else 0, alpha),
        )
    conn.commit()
    return conn


# ── Deliverable 4: the prefix is walked ONCE per build ─────────────────────


class TestSingleScan:
    def test_eval_prefix_is_listed_and_fetched_exactly_once(self, s3):
        n_evals = _seed_evals(s3)
        _put_json(s3, f"signals/{_DATE}/signals.json",
                  {"signals": {"AAA": {"ticker": "AAA"}}})

        art = build_agent_quality(
            s3, _BUCKET, _DATE, run_date=_RUN_DATE, outcomes_conn=_outcomes_conn(),
            cw=_NoCW(),
        )

        assert art["judge_outcome_ic"]["status"] == "ok"
        # One pagination of the eval prefix, one GetObject per real artifact.
        # Two of either means the duplicate scan is back.
        assert s3.eval_lists() == [_EVAL_PREFIX]
        assert len(s3.eval_gets()) == n_evals
        assert len(set(s3.eval_gets())) == n_evals  # no key fetched twice
        # latest.json is skipped by name, never fetched.
        assert f"{_EVAL_PREFIX}latest.json" not in s3.gets

    def test_injected_evals_issue_no_s3_reads_at_all(self, s3):
        _seed_evals(s3)
        evals = load_eval_artifacts(s3, _BUCKET)
        before = len(s3.eval_gets())
        blk = build_judge_outcome_ic_block(
            s3, _BUCKET, conn=_outcomes_conn(), evals=evals,
        )
        assert blk["status"] == "ok"
        assert len(s3.eval_gets()) == before  # the injected list is reused

    def test_evals_and_eval_prefix_together_is_a_contradiction(self, s3):
        with pytest.raises(ValueError, match="not\\s+both"):
            build_judge_outcome_ic_block(
                s3, _BUCKET, conn=_outcomes_conn(), evals=[],
                eval_prefix=_EVAL_PREFIX,
            )


class _NoCW:
    """CloudWatch stub with no prod telemetry — keeps the CW blocks off the
    artifact without reaching the network."""

    def get_metric_data(self, **kwargs):
        return {"MetricDataResults": [{"Id": "q", "Values": []}]}

    def get_paginator(self, name):
        class _P:
            def paginate(self, **kwargs):
                return [{"Metrics": []}]

        return _P()


# ── Deliverable 5: the scan is bounded, and the bound is asserted ──────────


class TestObjectCountCeiling:
    def test_ceiling_raises_before_any_get_object(self, s3):
        _seed_evals(s3)
        with pytest.raises(EvalScanBudgetExceeded) as exc:
            load_eval_artifacts(s3, _BUCKET, max_objects=2)
        # The message must carry the numbers a human needs to re-set it.
        assert "6" in str(exc.value) and "2" in str(exc.value)
        # Nothing was fetched — the ceiling is checked on the LISTING.
        assert s3.eval_gets() == []

    def test_ceiling_is_live_on_the_lambda_handler_path(self, s3, monkeypatch):
        """The bound must bind the producer the weekly Lambda actually calls,
        not only the loader in isolation. ``build_agent_quality`` is what
        ``lambda/eval_rolling_mean_handler.py`` invokes."""
        _seed_evals(s3)
        monkeypatch.setattr(jic, "MAX_EVAL_OBJECTS", 2)
        with pytest.raises(EvalScanBudgetExceeded):
            build_agent_quality(
                s3, _BUCKET, _DATE, run_date=_RUN_DATE,
                outcomes_conn=_outcomes_conn(), cw=_NoCW(),
            )
        assert s3.eval_gets() == []

    def test_ceiling_cannot_be_pinned_or_disabled_by_a_caller(self):
        """Both bounds default to ``None`` and resolve from the module
        constants at CALL time, so no call site can freeze a stale ceiling
        and there is no value meaning 'unbounded'."""
        import inspect

        sig = inspect.signature(load_eval_artifacts)
        assert sig.parameters["max_objects"].default is None
        assert sig.parameters["lookback_days"].default is None
        assert isinstance(MAX_EVAL_OBJECTS, int) and MAX_EVAL_OBJECTS > 0
        assert isinstance(DEFAULT_EVAL_LOOKBACK_DAYS, int)

    def test_at_the_ceiling_is_allowed(self, s3):
        n = _seed_evals(s3)
        assert len(load_eval_artifacts(s3, _BUCKET, max_objects=n)) == n


class TestLastModifiedWindow:
    def test_objects_outside_the_window_are_never_fetched(self, s3):
        _seed_evals(s3)
        # moto stamps LastModified = now; evaluate the window from a clock
        # far enough ahead that every object falls outside it.
        future = datetime.now(UTC) + timedelta(days=DEFAULT_EVAL_LOOKBACK_DAYS + 5)
        assert load_eval_artifacts(s3, _BUCKET, now=future) == []
        assert s3.eval_gets() == []  # windowed out on the LISTING, not after

    def test_objects_inside_the_window_are_fetched(self, s3):
        n = _seed_evals(s3)
        assert len(load_eval_artifacts(s3, _BUCKET, now=datetime.now(UTC))) == n

    def test_lookback_zero_requests_full_history_and_is_still_ceilinged(self, s3):
        n = _seed_evals(s3)
        future = datetime.now(UTC) + timedelta(days=10_000)
        assert len(load_eval_artifacts(s3, _BUCKET, lookback_days=0, now=future)) == n
        with pytest.raises(EvalScanBudgetExceeded):
            load_eval_artifacts(s3, _BUCKET, lookback_days=0, max_objects=1)


# ── Deliverable 6: the scan reports what it did ───────────────────────────


class TestInstrumentation:
    def test_scan_logs_counts_and_phase_timings(self, s3, caplog):
        n = _seed_evals(s3)
        with caplog.at_level("INFO", logger="evals.judge_outcome_ic"):
            load_eval_artifacts(s3, _BUCKET)
        line = next(m for m in caplog.messages if "eval scan" in m)
        for token in (f"listed={n}", f"fetched={n}", f"parsed={n}",
                      "windowed_out=", "bytes=", "workers=", "list_s=",
                      "fetch_s=", "total_s=", "ceiling="):
            assert token in line, f"{token!r} missing from scan log: {line}"

    def test_build_logs_per_phase_timings(self, s3, caplog):
        _seed_evals(s3)
        _put_json(s3, f"signals/{_DATE}/signals.json",
                  {"signals": {"AAA": {"ticker": "AAA"}}})
        with caplog.at_level("INFO"):
            build_agent_quality(
                s3, _BUCKET, _DATE, run_date=_RUN_DATE,
                outcomes_conn=_outcomes_conn(), cw=_NoCW(),
            )
        line = next(m for m in caplog.messages if "[build_agent_quality] built" in m)
        for token in ("n_signals=1", "n_evals=6", "ic_status=ok", "signals_s=",
                      "cost_s=", "evals_s=", "judge_outcome_ic_s=",
                      "cloudwatch_s=", "total_s="):
            assert token in line, f"{token!r} missing from build log: {line}"
