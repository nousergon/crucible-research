"""
Unit tests for ``scripts/build_agent_quality.py`` — the report-card
agent-quality producer (config alpha-engine-config#1149).

Locks down:
- Each metric block is emitted only from real persisted input, with the exact
  contract the evaluator consumer (crucible-evaluator#59) reads.
- Independent degradation: a missing source omits ONLY its block (never a
  fabricated value), so the consumer N/As just that component.
- cost_per_signal sums _cost_raw identically to aggregate_costs (implausible
  rows dropped).
- judge metrics computed over REAL evals only (skip-markers excluded).
- Date split: signals key off --date (trading day), cost/eval off --run-date.
- judge_outcome_ic wiring (old ROADMAP L480 re-scope): the block is NEVER
  silently absent — "ok" end-to-end through the S3 research.db snapshot,
  "insufficient" on absent history, "error" (+ WARN, siblings intact) on a
  broken precondition.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from datetime import date

import boto3
import pytest
from moto import mock_aws

from scripts.build_agent_quality import build_agent_quality, write_agent_quality

_BUCKET = "alpha-engine-research"
_DATE = date(2026, 6, 12)        # trading day
_RUN_DATE = date(2026, 6, 13)    # calendar Saturday the run executed


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=_BUCKET)
        yield client


def _put_json(s3, key, obj):
    s3.put_object(Bucket=_BUCKET, Key=key, Body=json.dumps(obj).encode())


def _put_jsonl(s3, key, rows: Iterable[dict]):
    body = "\n".join(json.dumps(r, default=str) for r in rows).encode()
    s3.put_object(Bucket=_BUCKET, Key=key, Body=body)


def _signals(n):
    return {"signals": {f"T{i}": {"ticker": f"T{i}", "signal": "ENTER", "score": 70} for i in range(n)}}


def _cost_row(cost, run_id="2026-06-13", agent="sector_quant:tech"):
    return {"schema_version": 2, "timestamp": "2026-06-13T13:30:00+00:00", "run_id": run_id,
            "agent_id": agent, "model_name": "claude-haiku-4-5", "call_seq": 1,
            "input_tokens": 1000, "output_tokens": 200, "cost_usd": cost}


def _eval(scores, skip=None):
    return {"schema_version": 2, "run_id": "2026-06-12", "judge_run_id": "jr",
            "timestamp": "2026-06-13T00:00:00+00:00", "judged_agent_id": "ic_cio",
            "rubric_id": "r", "rubric_version": "1.0", "judge_model": "claude-haiku-4-5",
            "dimension_scores": [{"dimension": f"d{i}", "score": s, "reasoning": "r"}
                                 for i, s in enumerate(scores)],
            "overall_reasoning": "ok", "judge_skip_reason": skip}


def _full_run(s3):
    _put_json(s3, f"signals/{_DATE.isoformat()}/signals.json", _signals(30))
    _put_jsonl(s3, f"decision_artifacts/_cost_raw/{_RUN_DATE.isoformat()}/a/x.jsonl",
               [_cost_row(0.50), _cost_row(0.25), _cost_row(0.25)])  # total $1.00
    base = f"decision_artifacts/_eval/{_RUN_DATE.isoformat()}/jr"
    _put_json(s3, f"{base}/ic_cio.json", _eval([4, 4, 4, 3, 5, 4]))      # pass
    _put_json(s3, f"{base}/sector_quant.json", _eval([3, 3, 4, 4, 4, 4]))  # pass
    _put_json(s3, f"{base}/macro.json", _eval([2, 4, 4, 4, 4, 4]))        # fail (a dim < 3)
    _put_json(s3, f"{base}/skip.json", _eval([], skip="degenerate_input"))  # excluded


class TestFullRun:
    @pytest.fixture(autouse=True)
    def _seed(self, s3):
        _full_run(s3)

    def test_all_four_blocks_present(self, s3):
        art = build_agent_quality(s3, _BUCKET, _DATE, run_date=_RUN_DATE)
        assert art["status"] == "ok"
        assert art["date"] == _DATE.isoformat()
        assert art["run_date"] == _RUN_DATE.isoformat()
        assert set(art) >= {"signal_volume_adequacy", "cost_per_signal",
                            "judge_rubric_pass_rate", "judge_rubric_distribution"}

    def test_signal_volume(self, s3):
        art = build_agent_quality(s3, _BUCKET, _DATE, run_date=_RUN_DATE)
        assert art["signal_volume_adequacy"] == {"value": 30, "n": 30}

    def test_cost_per_signal(self, s3):
        art = build_agent_quality(s3, _BUCKET, _DATE, run_date=_RUN_DATE)
        c = art["cost_per_signal"]
        assert c["total_cost_usd"] == 1.0
        assert c["n"] == 30
        assert c["value"] == pytest.approx(1.0 / 30, rel=1e-3)

    def test_judge_pass_rate(self, s3):
        # 2 of 3 real evals pass (macro fails); skip-marker excluded.
        art = build_agent_quality(s3, _BUCKET, _DATE, run_date=_RUN_DATE)
        prr = art["judge_rubric_pass_rate"]
        assert prr["n"] == 3
        assert prr["value"] == pytest.approx(2 / 3, rel=1e-3)

    def test_judge_distribution(self, s3):
        # 18 dim-scores: score 4 appears 13x (4+4+5) → modal concentration 13/18.
        art = build_agent_quality(s3, _BUCKET, _DATE, run_date=_RUN_DATE)
        dist = art["judge_rubric_distribution"]
        assert dist["n"] == 3
        assert dist["value"] == pytest.approx(13 / 18, rel=1e-3)

    def test_roundtrip_write(self, s3):
        art = build_agent_quality(s3, _BUCKET, _DATE, run_date=_RUN_DATE)
        key = write_agent_quality(s3, _BUCKET, art)
        assert key == f"backtest/{_DATE.isoformat()}/agent_quality.json"
        got = json.loads(s3.get_object(Bucket=_BUCKET, Key=key)["Body"].read())
        assert got["signal_volume_adequacy"]["value"] == 30


class TestIndependentDegradation:
    def test_empty_run_only_status(self, s3):
        art = build_agent_quality(s3, _BUCKET, _DATE, run_date=_RUN_DATE)
        assert art["status"] == "ok"
        assert not any(isinstance(v, dict) and "value" in v for v in art.values())

    def test_no_signals_drops_cost_and_volume(self, s3):
        # cost rows present but no signals → no denominator → both blocks absent.
        _put_jsonl(s3, f"decision_artifacts/_cost_raw/{_RUN_DATE.isoformat()}/a/x.jsonl", [_cost_row(1.0)])
        art = build_agent_quality(s3, _BUCKET, _DATE, run_date=_RUN_DATE)
        assert "signal_volume_adequacy" not in art
        assert "cost_per_signal" not in art

    def test_signals_without_cost_keeps_volume(self, s3):
        _put_json(s3, f"signals/{_DATE.isoformat()}/signals.json", _signals(12))
        art = build_agent_quality(s3, _BUCKET, _DATE, run_date=_RUN_DATE)
        assert art["signal_volume_adequacy"] == {"value": 12, "n": 12}
        assert "cost_per_signal" not in art

    def test_only_skip_evals_drops_judge(self, s3):
        base = f"decision_artifacts/_eval/{_RUN_DATE.isoformat()}/jr"
        _put_json(s3, f"{base}/skip.json", _eval([], skip="degenerate_input"))
        art = build_agent_quality(s3, _BUCKET, _DATE, run_date=_RUN_DATE)
        assert "judge_rubric_pass_rate" not in art
        assert "judge_rubric_distribution" not in art

    def test_implausible_cost_rows_dropped(self, s3):
        _put_json(s3, f"signals/{_DATE.isoformat()}/signals.json", _signals(10))
        # one real row + one test-pollution row (bad run_id, huge tokens).
        _put_jsonl(s3, f"decision_artifacts/_cost_raw/{_RUN_DATE.isoformat()}/a/x.jsonl",
                   [_cost_row(2.0), {"run_id": "run-x", "cost_usd": 999.0,
                                     "input_tokens": 9_000_000, "output_tokens": 1}])
        art = build_agent_quality(s3, _BUCKET, _DATE, run_date=_RUN_DATE)
        assert art["cost_per_signal"]["total_cost_usd"] == 2.0


class TestDateSplit:
    def test_run_date_defaults_to_date(self, s3):
        # everything under the SAME date → run_date defaults to date.
        _put_json(s3, f"signals/{_DATE.isoformat()}/signals.json", _signals(5))
        _put_jsonl(s3, f"decision_artifacts/_cost_raw/{_DATE.isoformat()}/a/x.jsonl", [_cost_row(1.0, run_id="2026-06-12")])
        art = build_agent_quality(s3, _BUCKET, _DATE)  # no run_date
        assert art["run_date"] == _DATE.isoformat()
        assert art["cost_per_signal"]["total_cost_usd"] == 1.0


class _StubCW:
    """Minimal CloudWatch stub: returns Failures/Invocations sums by inspecting
    the metric-math Expression. Records the expressions seen (for env-filter asserts)."""

    def __init__(self, failures: float, invocations: float) -> None:
        self._f, self._i = failures, invocations
        self.exprs: list[str] = []

    def get_metric_data(self, MetricDataQueries, StartTime, EndTime):  # noqa: N803
        expr = MetricDataQueries[0]["Expression"]
        self.exprs.append(expr)
        if "Failures" in expr:
            v = self._f
        elif "Invocations" in expr:
            v = self._i
        else:
            v = 0
        return {"MetricDataResults": [{"Id": "q", "Values": [v] if v else []}]}


class TestAgentValidationFailureRate:
    """config#1154/#1149 — fleet Failures/Invocations from AlphaEngine/Agents prod telemetry."""

    def test_failure_rate_from_prod_telemetry(self, s3):
        _full_run(s3)
        cw = _StubCW(failures=3, invocations=120)
        art = build_agent_quality(s3, _BUCKET, _DATE, run_date=_RUN_DATE, cw=cw)
        blk = art["agent_validation_failure_rate"]
        assert blk["value"] == round(3 / 120, 4)
        assert blk["n"] == 120
        # Reads PROD only (skips the test-polluted agent_id-only series).
        assert cw.exprs and all('env="prod"' in e for e in cw.exprs)

    def test_absent_when_no_prod_invocations(self, s3):
        _full_run(s3)
        art = build_agent_quality(s3, _BUCKET, _DATE, run_date=_RUN_DATE, cw=_StubCW(0, 0))
        assert "agent_validation_failure_rate" not in art

    def test_cw_error_is_non_fatal(self, s3):
        _full_run(s3)

        class _BoomCW:
            def get_metric_data(self, **kwargs):
                raise RuntimeError("AccessDenied")

        art = build_agent_quality(s3, _BUCKET, _DATE, run_date=_RUN_DATE, cw=_BoomCW())
        assert "agent_validation_failure_rate" not in art
        assert art["signal_volume_adequacy"]["value"] == 30  # other components intact


class _StubCWFull:
    """CloudWatch stub: list_metrics paginator (env=prod agent_ids) + per-agent
    MetricStat get_metric_data, plus the SUM(SEARCH) Expression path for the
    fleet failure rate. ``per_agent`` is {metric: {agent_id: value}}."""

    def __init__(self, per_agent):
        self._pa = per_agent

    def get_paginator(self, name):
        assert name == "list_metrics"
        pa = self._pa

        class _P:
            def paginate(self, Namespace, MetricName, Dimensions):  # noqa: N803
                agents = pa.get(MetricName, {})
                metrics = [
                    {"Dimensions": [{"Name": "agent_id", "Value": a},
                                    {"Name": "env", "Value": "prod"}]}
                    for a in agents
                ]
                return [{"Metrics": metrics}]

        return _P()

    def get_metric_data(self, MetricDataQueries, StartTime, EndTime):  # noqa: N803
        results = []
        for q in MetricDataQueries:
            ms = q.get("MetricStat")
            if ms:
                metric = ms["Metric"]["MetricName"]
                agent = next(d["Value"] for d in ms["Metric"]["Dimensions"]
                             if d["Name"] == "agent_id")
                v = self._pa.get(metric, {}).get(agent)
                results.append({"Id": q["Id"], "Values": [v] if v is not None else []})
            else:  # Expression (SUM(SEARCH)) — fleet failure rate
                expr = q["Expression"]
                metric = ("Failures" if "Failures" in expr
                          else "Invocations" if "Invocations" in expr else None)
                total = sum(self._pa.get(metric, {}).values()) if metric else 0
                results.append({"Id": q["Id"], "Values": [total] if total else []})
        return {"MetricDataResults": results}


class TestRetryStormAndLatency:
    """config#1149 — retry_storm_count + agent_latency_p95 from per-agent prod telemetry."""

    def _cw(self):
        return _StubCWFull({
            "Invocations": {"a": 100, "b": 50},
            "Failures": {"a": 2, "b": 1},
            # a: 5 attempts all recovered (5==5, no storm); b: no attempts;
            # c: 3 attempts, only 1 recovered (3>1) → reached ceiling.
            "RetryAttempts": {"a": 5, "b": 0, "c": 3},
            "RetrySuccesses": {"a": 5, "b": 0, "c": 1},
            "DurationMs": {"a": 8000, "b": 70000, "c": 12000},  # b worst tail
        })

    def test_retry_storm_counts_unrecovered_agents(self, s3):
        _full_run(s3)
        art = build_agent_quality(s3, _BUCKET, _DATE, run_date=_RUN_DATE, cw=self._cw())
        assert art["retry_storm_count"]["value"] == 1   # only c
        assert art["retry_storm_count"]["n"] == 3

    def test_latency_p95_is_worst_agent(self, s3):
        _full_run(s3)
        art = build_agent_quality(s3, _BUCKET, _DATE, run_date=_RUN_DATE, cw=self._cw())
        assert art["agent_latency_p95"]["value"] == 70000.0  # b's p95
        assert art["agent_latency_p95"]["n"] == 3

    def test_failure_rate_via_full_stub(self, s3):
        _full_run(s3)
        art = build_agent_quality(s3, _BUCKET, _DATE, run_date=_RUN_DATE, cw=self._cw())
        assert art["agent_validation_failure_rate"]["value"] == round(3 / 150, 4)

    def test_absent_when_no_prod_agents(self, s3):
        _full_run(s3)
        art = build_agent_quality(s3, _BUCKET, _DATE, run_date=_RUN_DATE, cw=_StubCWFull({}))
        assert "retry_storm_count" not in art
        assert "agent_latency_p95" not in art


# ── judge_outcome_ic wiring (old ROADMAP L480 re-scope) ───────────────────────
#
# Statistics + attribution are unit-tested in tests/test_judge_outcome_ic.py;
# this class locks the PRODUCER wiring: end-to-end through moto S3 (flat
# canonical _eval/ layout + the research.db snapshot download), injected-conn
# path, and the never-silently-absent isolation posture.

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

# Saturday capture dates → the Friday trading days research stamped.
_CAP1, _TD1 = "2026-06-06", "2026-06-05"
_CAP2, _TD2 = "2026-06-13", "2026-06-12"


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


def _seed_judge_history(s3):
    """Flat canonical _eval/ layout (config#793) — 2 dates x 3 tickers, judge
    score monotone with realized alpha on date1, one inversion on date2."""
    scores = {(_CAP1, "AAA"): 5, (_CAP1, "BBB"): 3, (_CAP1, "CCC"): 1,
              (_CAP2, "AAA"): 5, (_CAP2, "BBB"): 4, (_CAP2, "CCC"): 1}
    for (cap, ticker), score in scores.items():
        doc = _thesis_eval(ticker, cap, score)
        key = (f"decision_artifacts/_eval/2606131230_"
               f"{doc['judged_agent_id']}.{cap}.claude-haiku-4-5.json")
        _put_json(s3, key, doc)
    # An unattributable slate-level eval + the latest.json sidecar (skipped).
    cio = _eval([4, 4, 4])
    _put_json(s3, "decision_artifacts/_eval/2606131230_ic_cio.x.claude-haiku-4-5.json", cio)
    _put_json(s3, "decision_artifacts/_eval/latest.json", {"artifact_key": "x"})


def _outcomes_db(path=":memory:"):
    conn = sqlite3.connect(path)
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


class TestJudgeOutcomeIC:
    def test_ok_end_to_end_via_s3_db_snapshot(self, s3, tmp_path):
        _seed_judge_history(s3)
        db_path = tmp_path / "research.db"
        _outcomes_db(str(db_path)).close()
        s3.upload_file(str(db_path), _BUCKET, "research.db")

        art = build_agent_quality(s3, _BUCKET, _DATE, run_date=_RUN_DATE)
        blk = art["judge_outcome_ic"]
        assert blk["status"] == "ok"
        assert blk["schema_version"] == 1
        assert blk["horizon_days"] == 21
        assert blk["overall"]["n_eval_dates"] == 2
        assert blk["overall"]["n"] == 6
        assert blk["overall"]["date_ic_mean"] == pytest.approx(0.75)
        assert set(blk["by_dimension"]) == {"depth", "grounding"}
        assert blk["n_unattributable"] == 1  # the ic_cio eval

    def test_ok_via_injected_conn(self, s3):
        _seed_judge_history(s3)
        conn = _outcomes_db()
        art = build_agent_quality(
            s3, _BUCKET, _DATE, run_date=_RUN_DATE, outcomes_conn=conn,
        )
        assert art["judge_outcome_ic"]["status"] == "ok"
        conn.execute("SELECT 1")  # injected conn is NOT closed by the block

    def test_insufficient_on_absent_history(self, s3):
        # No eval artifacts + an empty outcomes store → legitimate
        # "insufficient", never an error and never silently absent.
        conn = _outcomes_db()
        conn.execute("DELETE FROM score_performance_outcomes")
        art = build_agent_quality(
            s3, _BUCKET, _DATE, run_date=_RUN_DATE, outcomes_conn=conn,
        )
        blk = art["judge_outcome_ic"]
        assert blk["status"] == "insufficient"
        assert blk["overall"]["n"] == 0

    def test_error_isolation_on_broken_precondition(self, s3):
        # Eval history exists but the research.db snapshot is MISSING from
        # S3 — a broken precondition, surfaced as an explicit error status
        # (+ WARN) while sibling components and the artifact write survive.
        _full_run(s3)
        _seed_judge_history(s3)
        art = build_agent_quality(s3, _BUCKET, _DATE, run_date=_RUN_DATE)
        blk = art["judge_outcome_ic"]
        assert blk["status"] == "error"
        assert "error" in blk
        assert art["signal_volume_adequacy"]["value"] == 30  # siblings intact
        key = write_agent_quality(s3, _BUCKET, art)
        got = json.loads(s3.get_object(Bucket=_BUCKET, Key=key)["Body"].read())
        assert got["judge_outcome_ic"]["status"] == "error"

    def test_block_never_silently_absent(self, s3):
        # Even a completely empty bucket run carries the block (error state:
        # the research.db precondition is broken in this synthetic world).
        art = build_agent_quality(s3, _BUCKET, _DATE, run_date=_RUN_DATE)
        assert "judge_outcome_ic" in art


class TestFlatLayoutRegression:
    """Regression: config#1840 — _load_evals must enumerate the flat _eval/
    layout, not the legacy nested _eval/{run_date}/ partition (config#793 swap).
    This test locks the fix so a future rename/reorganization can't silently
    re-break the rubric blocks."""

    def test_rubric_metrics_load_from_flat_layout(self, s3):
        """Evals in the flat layout are loaded for rubric_pass_rate/distribution."""
        # Seed: flat layout evals (NOT the old nested structure).
        # We reuse the TestJudgeOutcomeIC fixture (_seed_judge_history) which
        # already uses the canonical flat layout.
        _seed_judge_history(s3)
        # Also need signals for volume adequacy + one cost row.
        _put_json(s3, f"signals/{_DATE.isoformat()}/signals.json", _signals(30))
        _put_jsonl(s3, f"decision_artifacts/_cost_raw/{_RUN_DATE.isoformat()}/a/x.jsonl",
                   [_cost_row(0.50)])

        art = build_agent_quality(s3, _BUCKET, _DATE, run_date=_RUN_DATE)
        # The flat-layout evals are now loaded, so rubric blocks are populated.
        assert "judge_rubric_pass_rate" in art, (
            "judge_rubric_pass_rate missing — _load_evals did not load "
            "from flat _eval/ layout"
        )
        assert art["judge_rubric_pass_rate"]["value"] > 0, (
            "rubric metrics are zero — evals loaded but invalid structure"
        )
        assert art["judge_rubric_pass_rate"]["n"] == 7, (
            "Expected 6 evals from flat layout (2 dates x 3 tickers + 1 slate-level unattributable)"
        )
        assert "judge_rubric_distribution" in art
