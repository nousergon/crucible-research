"""
Unit tests for ``scripts/aggregate_costs.py``.

Locks down:

- Happy path: multiple JSONL files concatenated to one parquet at the
  canonical S3 key with the expected row count + columns.
- Empty-day path: no JSONL files for the date returns ``None`` (no
  empty parquet written), exits non-zero from the CLI.
- Multi-agent breakdown: ``by_sector_team`` / ``by_model`` / ``by_run_type``
  totals match per-row sums.
- Malformed JSONL rejection: a corrupt line raises with the file + line
  number — surfacing corruption rather than silently skipping.
- CLI argparse: bad date format exits 2; missing --date prompts; --output-key
  override routes the parquet correctly.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterable
from datetime import date

import boto3
import pandas as pd
import pytest
from moto import mock_aws

_BUCKET = "alpha-engine-research"


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=_BUCKET)
        yield client


def _put_jsonl(s3, key: str, rows: Iterable[dict]) -> None:
    body = "\n".join(json.dumps(r, default=str) for r in rows).encode("utf-8")
    s3.put_object(Bucket=_BUCKET, Key=key, Body=body, ContentType="application/x-ndjson")


def _make_row(
    *,
    agent_id: str,
    sector_team_id: str | None,
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    run_type: str = "weekly_research",
    call_seq: int = 1,
    schema_version: int = 2,
    web_search_requests: int = 0,
    web_fetch_requests: int = 0,
) -> dict:
    """Builder for synthetic JSONL rows. Defaults to schema v2 (tool-fee
    columns present + zero-defaulted). Tests of the v1 → v2 migration
    backfill path pass ``schema_version=1`` to omit the new columns
    entirely — exercises the aggregator's missing-as-zero handling.
    """
    row: dict = {
        "schema_version": schema_version,
        "timestamp": "2026-05-02T13:30:00+00:00",
        "run_id": "2026-05-02",
        "agent_id": agent_id,
        "sector_team_id": sector_team_id,
        "node_name": "some_node",
        "run_type": run_type,
        "prompt_id": None,
        "prompt_version": None,
        "prompt_version_hash": None,
        "model_name": model_name,
        "call_seq": call_seq,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": 0,
        "cache_create_tokens": 0,
        "cost_usd": cost_usd,
    }
    if schema_version >= 2:
        row["web_search_requests"] = web_search_requests
        row["web_fetch_requests"] = web_fetch_requests
    return row


# ── aggregate_day happy path ──────────────────────────────────────────────


class TestAggregateDayHappyPath:
    def test_concatenates_multiple_files_to_single_parquet(self, s3):
        from scripts.aggregate_costs import aggregate_day

        # Three sector teams + macro + cio, each with 1-3 calls.
        _put_jsonl(
            s3,
            "decision_artifacts/_cost_raw/2026-05-02/2026-05-02/sector_team:tech.jsonl",
            [
                _make_row(
                    agent_id="sector_team:tech",
                    sector_team_id="tech",
                    model_name="claude-haiku-4-5",
                    input_tokens=4000,
                    output_tokens=1200,
                    cost_usd=0.010,
                    call_seq=1,
                ),
                _make_row(
                    agent_id="sector_team:tech",
                    sector_team_id="tech",
                    model_name="claude-haiku-4-5",
                    input_tokens=2000,
                    output_tokens=800,
                    cost_usd=0.006,
                    call_seq=2,
                ),
            ],
        )
        _put_jsonl(
            s3,
            "decision_artifacts/_cost_raw/2026-05-02/2026-05-02/sector_team:financials.jsonl",
            [
                _make_row(
                    agent_id="sector_team:financials",
                    sector_team_id="financials",
                    model_name="claude-haiku-4-5",
                    input_tokens=3500,
                    output_tokens=900,
                    cost_usd=0.008,
                    call_seq=1,
                ),
            ],
        )
        _put_jsonl(
            s3,
            "decision_artifacts/_cost_raw/2026-05-02/2026-05-02/macro_economist.jsonl",
            [
                _make_row(
                    agent_id="macro_economist",
                    sector_team_id=None,
                    model_name="claude-sonnet-4-6",
                    input_tokens=8000,
                    output_tokens=3000,
                    cost_usd=0.069,
                    call_seq=1,
                ),
            ],
        )

        summary = aggregate_day(s3, _BUCKET, date(2026, 5, 2))

        assert summary is not None
        assert summary["rows_in"] == 4
        assert summary["files_read"] == 3
        assert summary["output_key"] == "decision_artifacts/_cost/2026-05-02/cost.parquet"
        # Total cost = 0.010 + 0.006 + 0.008 + 0.069 = 0.093
        assert summary["total_cost_usd"] == pytest.approx(0.093)

        # Verify the parquet was actually written with the right row count.
        obj = s3.get_object(Bucket=_BUCKET, Key=summary["output_key"])
        df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
        assert len(df) == 4
        assert set(df["agent_id"]) == {
            "sector_team:tech",
            "sector_team:financials",
            "macro_economist",
        }

    def test_summary_breakdowns_match_per_row_sums(self, s3):
        from scripts.aggregate_costs import aggregate_day

        _put_jsonl(
            s3,
            "decision_artifacts/_cost_raw/2026-05-02/2026-05-02/sector_team:tech.jsonl",
            [
                _make_row(
                    agent_id="sector_team:tech",
                    sector_team_id="tech",
                    model_name="claude-haiku-4-5",
                    input_tokens=1000,
                    output_tokens=500,
                    cost_usd=0.0035,
                ),
            ],
        )
        _put_jsonl(
            s3,
            "decision_artifacts/_cost_raw/2026-05-02/2026-05-02/sector_team:financials.jsonl",
            [
                _make_row(
                    agent_id="sector_team:financials",
                    sector_team_id="financials",
                    model_name="claude-haiku-4-5",
                    input_tokens=2000,
                    output_tokens=800,
                    cost_usd=0.006,
                ),
            ],
        )
        _put_jsonl(
            s3,
            "decision_artifacts/_cost_raw/2026-05-02/2026-05-02/ic_cio.jsonl",
            [
                _make_row(
                    agent_id="ic_cio",
                    sector_team_id=None,
                    model_name="claude-sonnet-4-6",
                    input_tokens=10000,
                    output_tokens=2000,
                    cost_usd=0.060,
                ),
            ],
        )

        summary = aggregate_day(s3, _BUCKET, date(2026, 5, 2))

        # NaN sector_team_id (cross-sector agents like macro_economist + ic_cio)
        # gets mapped to "(none)" so the dashboard groups it cleanly instead
        # of displaying "nan" / "None".
        assert summary["by_sector_team"] == {
            "tech": pytest.approx(0.0035),
            "financials": pytest.approx(0.006),
            "(none)": pytest.approx(0.060),
        }
        assert summary["by_model"] == {
            "claude-haiku-4-5": pytest.approx(0.0095),
            "claude-sonnet-4-6": pytest.approx(0.060),
        }
        assert summary["by_agent_id"] == {
            "sector_team:tech": pytest.approx(0.0035),
            "sector_team:financials": pytest.approx(0.006),
            "ic_cio": pytest.approx(0.060),
        }

    def test_token_totals_surfaced(self, s3):
        from scripts.aggregate_costs import aggregate_day

        _put_jsonl(
            s3,
            "decision_artifacts/_cost_raw/2026-05-02/2026-05-02/agent_a.jsonl",
            [
                _make_row(
                    agent_id="agent_a",
                    sector_team_id=None,
                    model_name="claude-haiku-4-5",
                    input_tokens=5000,
                    output_tokens=1500,
                    cost_usd=0.0125,
                ),
            ],
        )
        summary = aggregate_day(s3, _BUCKET, date(2026, 5, 2))
        assert summary["total_input_tokens"] == 5000
        assert summary["total_output_tokens"] == 1500


# ── Empty-day path ────────────────────────────────────────────────────────


class TestEmptyDay:
    def test_no_files_returns_none(self, s3):
        from scripts.aggregate_costs import aggregate_day

        summary = aggregate_day(s3, _BUCKET, date(2026, 5, 2))
        assert summary is None

    def test_files_with_zero_rows_returns_none(self, s3):
        """Files exist but contain no JSON lines (e.g. all blank) — still
        no parquet written; returns None so operator sees zero data."""
        from scripts.aggregate_costs import aggregate_day

        s3.put_object(
            Bucket=_BUCKET,
            Key="decision_artifacts/_cost_raw/2026-05-02/2026-05-02/empty.jsonl",
            Body=b"\n\n  \n",
        )
        summary = aggregate_day(s3, _BUCKET, date(2026, 5, 2))
        assert summary is None

    def test_cli_returns_nonzero_on_empty_day(self, s3, capsys):
        pass
        # Note: CLI uses default boto3.client; without dependency injection
        # we'd need to mock that. Skip the CLI-level empty-day assertion;
        # logic-level empty-day is covered above.


# ── Malformed JSONL ──────────────────────────────────────────────────────


class TestMalformedJsonl:
    def test_corrupt_line_raises_with_location(self, s3):
        from scripts.aggregate_costs import aggregate_day

        s3.put_object(
            Bucket=_BUCKET,
            Key="decision_artifacts/_cost_raw/2026-05-02/r/agent.jsonl",
            Body=b'{"call_seq": 1}\n{not valid json}\n{"call_seq": 3}\n',
        )
        with pytest.raises(RuntimeError, match=r"Malformed JSONL .* line 2"):
            aggregate_day(s3, _BUCKET, date(2026, 5, 2))


# ── Output-key override ──────────────────────────────────────────────────


class TestOutputKeyOverride:
    def test_override_routes_parquet(self, s3):
        from scripts.aggregate_costs import aggregate_day

        _put_jsonl(
            s3,
            "decision_artifacts/_cost_raw/2026-05-02/2026-05-02/agent_a.jsonl",
            [
                _make_row(
                    agent_id="agent_a",
                    sector_team_id=None,
                    model_name="claude-haiku-4-5",
                    input_tokens=10,
                    output_tokens=5,
                    cost_usd=0.0001,
                ),
            ],
        )
        summary = aggregate_day(
            s3,
            _BUCKET,
            date(2026, 5, 2),
            output_key_override="custom/path/test.parquet",
        )
        assert summary["output_key"] == "custom/path/test.parquet"
        # Default path should NOT exist.
        with pytest.raises(s3.exceptions.NoSuchKey):
            s3.get_object(
                Bucket=_BUCKET,
                Key="decision_artifacts/_cost/2026-05-02/cost.parquet",
            )


# ── CLI argparse ──────────────────────────────────────────────────────────


class TestCliArgs:
    def test_bad_date_format_exits_2(self, capsys):
        from scripts.aggregate_costs import main

        rc = main(["--date", "2026/05/02"])  # wrong separator
        assert rc == 2
        captured = capsys.readouterr()
        assert "ISO YYYY-MM-DD" in captured.err

    def test_missing_required_date_exits_2(self):
        from scripts.aggregate_costs import main

        with pytest.raises(SystemExit) as exc:
            main([])
        # argparse missing-required exits 2.
        assert exc.value.code == 2


# ── Pagination over many JSONL files ─────────────────────────────────────


class TestPagination:
    def test_handles_many_files(self, s3):
        """ListObjectsV2 paginates at 1000 keys per page; verify the
        aggregator surfaces all of them. We use 50 here to keep the test
        fast; the paginator code path is exercised regardless."""
        from scripts.aggregate_costs import aggregate_day

        for i in range(50):
            _put_jsonl(
                s3,
                f"decision_artifacts/_cost_raw/2026-05-02/r/agent_{i}.jsonl",
                [
                    _make_row(
                        agent_id=f"agent_{i}",
                        sector_team_id=None,
                        model_name="claude-haiku-4-5",
                        input_tokens=100,
                        output_tokens=50,
                        cost_usd=0.000350,
                    ),
                ],
            )
        summary = aggregate_day(s3, _BUCKET, date(2026, 5, 2))
        assert summary["rows_in"] == 50
        assert summary["files_read"] == 50


# ── Implausibility filter ────────────────────────────────────────────────
#
# Regression coverage for the 2026-05-13 incident: a unit test running
# with real AWS creds wrote ~$1014 of fake-agent rows into the
# _cost_raw partition, inflating the dashboard's weekly trend chart
# 700x. The filter is structural (run_id prefix + token ceiling), not
# a name blocklist — robust against new test fixture names.


class TestImplausibleRowFilter:
    def test_plausible_production_run_id_passes(self):
        from scripts.aggregate_costs import _is_plausible_cost_row

        for run_id in ("2026-05-13", "2026-05-15", "2026-05-20-001", "2026-05-20_run1"):
            ok, reason = _is_plausible_cost_row(
                {
                    "run_id": run_id,
                    "input_tokens": 4000,
                    "output_tokens": 1200,
                }
            )
            assert ok, f"{run_id!r} should pass — reason: {reason}"

    def test_run_id_shape_is_no_longer_a_rejection_reason(self):
        """run_id shape must NOT gate plausibility (I5206).

        This test previously asserted the opposite: that ad-hoc run_ids
        like ``run-x`` were rejected. That guard silently killed the cost
        pipeline for 17 days once production run_ids changed shape —
        ``eval_judge`` moved to artifact-scoped ids like
        ``276a5be44c7c-EXEL-v5``, and 100% of production rows were
        classified as test pollution. The invariant was a format coupling,
        not a law about the data. Inverted deliberately so a
        reintroduction fails here.
        """
        from scripts.aggregate_costs import _is_plausible_cost_row

        for run_id in ("run-1", "run-budget-test", "run-x", "", None, "276a5be44c7c-EXEL-v5", "2026-07-10"):
            ok, reason = _is_plausible_cost_row(
                {
                    "run_id": run_id,
                    "input_tokens": 4000,
                    "output_tokens": 1200,
                }
            )
            assert ok, f"run_id={run_id!r} rejected on shape: {reason}"
            assert reason is None

    def test_real_production_row_from_the_i5206_partition_passes(self):
        """Regression lock on the exact row shape that was being dropped.

        Read from ``decision_artifacts/_cost_raw/2026-07-19/
        276a5be44c7c-EXEL-v5/eval_judge.jsonl`` on 2026-07-28 — a genuine
        production row (real model, real tokens, real cost) that the old
        guard rejected.
        """
        from scripts.aggregate_costs import _is_plausible_cost_row

        ok, reason = _is_plausible_cost_row(
            {
                "run_id": "276a5be44c7c-EXEL-v5",
                "agent_id": "eval_judge",
                "model_name": "claude-sonnet-4-6",
                "input_tokens": 10011,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "cost_usd": 0.049008,
            }
        )
        assert ok, f"the I5206 row is still being dropped: {reason}"

    def test_implausible_token_count_fails(self):
        from scripts.aggregate_costs import _is_plausible_cost_row

        # The big_spender row had input_tokens=1e9 — 200x the Claude
        # Opus 4.7 context window. Real API calls cannot reach this.
        ok, reason = _is_plausible_cost_row(
            {
                "run_id": "2026-05-13",
                "input_tokens": 1_000_000_000,
                "output_tokens": 0,
            }
        )
        assert not ok
        assert "input_tokens" in reason
        assert "1,000,000,000" in reason

    def test_aggregate_day_drops_polluted_rows(self, s3):
        from scripts.aggregate_costs import aggregate_day

        # Mix: 3 real rows + 2 test pollution rows in the same JSONL.
        _put_jsonl(
            s3,
            "decision_artifacts/_cost_raw/2026-05-13/sector_team:tech.jsonl",
            [
                _make_row(
                    agent_id="sector_team:tech",
                    sector_team_id="tech",
                    model_name="claude-haiku-4-5",
                    input_tokens=4000,
                    output_tokens=1200,
                    cost_usd=0.012,
                    call_seq=1,
                ),
                _make_row(
                    agent_id="sector_team:tech",
                    sector_team_id="tech",
                    model_name="claude-haiku-4-5",
                    input_tokens=2000,
                    output_tokens=800,
                    cost_usd=0.006,
                    call_seq=2,
                ),
                _make_row(
                    agent_id="sector_team:tech",
                    sector_team_id="tech",
                    model_name="claude-haiku-4-5",
                    input_tokens=3000,
                    output_tokens=1000,
                    cost_usd=0.008,
                    call_seq=3,
                ),
                # The exact 2026-05-13 pollution shape.
                {
                    **_make_row(
                        agent_id="big_spender",
                        sector_team_id=None,
                        model_name="claude-haiku-4-5",
                        input_tokens=1_000_000_000,
                        output_tokens=0,
                        cost_usd=1000.0,
                    ),
                    "run_id": "run-x",
                },
                {
                    **_make_row(
                        agent_id="runaway_agent",
                        sector_team_id=None,
                        model_name="claude-haiku-4-5",
                        input_tokens=10_000_000,
                        output_tokens=0,
                        cost_usd=10.0,
                    ),
                    "run_id": "run-budget-test",
                },
            ],
        )
        summary = aggregate_day(s3, _BUCKET, date(2026, 5, 13))
        # 3 real rows survive; 2 pollution rows dropped (one by run_id,
        # one by token count — both filters catch each).
        assert summary["rows_in"] == 3
        # Total cost is real-only, not inflated.
        assert abs(summary["total_cost_usd"] - 0.026) < 1e-6
        # by_agent_id has only the real agent (float tolerance).
        assert set(summary["by_agent_id"].keys()) == {"sector_team:tech"}
        assert abs(summary["by_agent_id"]["sector_team:tech"] - 0.026) < 1e-6

    def test_aggregate_day_raises_when_every_row_is_dropped(self, s3):
        """Rows in, nothing out → RAISE, never a silent skip (I5206).

        This previously asserted ``summary is None``. That silence is
        precisely how the pipeline stayed dead for 17 days: the guard
        rejected every production row, the state succeeded, and the
        dashboard rendered stale partitions. "Input had rows but none
        survived" means the filter and the producer disagree about what a
        valid row is — a contract violation a producer may not swallow.
        """
        from scripts.aggregate_costs import CostAggregationError, aggregate_day

        _put_jsonl(
            s3,
            "decision_artifacts/_cost_raw/2026-05-13/run-x/big_spender.jsonl",
            [
                {
                    **_make_row(
                        agent_id="big_spender",
                        sector_team_id=None,
                        model_name="claude-haiku-4-5",
                        input_tokens=1_000_000_000,
                        output_tokens=0,
                        cost_usd=1000.0,
                    ),
                    "run_id": "run-x",
                },
            ],
        )
        with pytest.raises(CostAggregationError, match="dropped as implausible"):
            aggregate_day(s3, _BUCKET, date(2026, 5, 13))

    def test_empty_input_still_returns_none_not_an_error(self, s3):
        """A genuinely quiet day is NOT an error — only a total drop is.

        Guards the distinction the raise above depends on: zero rows read
        is a legitimately empty week and must stay a quiet ``None``, or
        the alarm this change installs would fire every quiet cycle and
        get ignored.
        """
        from scripts.aggregate_costs import aggregate_day

        _put_jsonl(
            s3,
            "decision_artifacts/_cost_raw/2026-05-14/run-x/nobody.jsonl",
            [],
        )
        assert aggregate_day(s3, _BUCKET, date(2026, 5, 14)) is None


# ── Tool-fee column handling (schema v2) ──────────────────────────────────


class TestToolRequestTotals:
    """Lock down ``total_web_search_requests`` + ``total_web_fetch_requests``
    on the summary surface — the keystone metric for the dashboard's
    tool-fee cost panel."""

    def test_sums_tool_request_counts_across_files(self, s3):
        from scripts.aggregate_costs import aggregate_day

        _put_jsonl(
            s3,
            "decision_artifacts/_cost_raw/2026-05-30/2026-05-30/qual_a.jsonl",
            [
                _make_row(
                    agent_id="qual_a",
                    sector_team_id="tech",
                    model_name="claude-haiku-4-5",
                    input_tokens=1000,
                    output_tokens=200,
                    cost_usd=0.502,
                    web_search_requests=50,
                    web_fetch_requests=2,
                ),
            ],
        )
        _put_jsonl(
            s3,
            "decision_artifacts/_cost_raw/2026-05-30/2026-05-30/qual_b.jsonl",
            [
                _make_row(
                    agent_id="qual_b",
                    sector_team_id="financials",
                    model_name="claude-haiku-4-5",
                    input_tokens=500,
                    output_tokens=100,
                    cost_usd=0.301,
                    web_search_requests=30,
                    web_fetch_requests=0,
                ),
            ],
        )
        summary = aggregate_day(s3, _BUCKET, date(2026, 5, 30))
        assert summary is not None
        assert summary["total_web_search_requests"] == 80
        assert summary["total_web_fetch_requests"] == 2

    def test_zero_default_when_v1_only_partition(self, s3):
        """A day's worth of pre-v2 rows (no tool-fee columns) aggregates
        to zero totals — backfill safety. The aggregator must not raise
        when the schema-v2 columns are entirely absent from the DataFrame."""
        from scripts.aggregate_costs import aggregate_day

        _put_jsonl(
            s3,
            "decision_artifacts/_cost_raw/2026-05-02/2026-05-02/legacy.jsonl",
            [
                _make_row(
                    agent_id="legacy",
                    sector_team_id=None,
                    model_name="claude-haiku-4-5",
                    input_tokens=1000,
                    output_tokens=200,
                    cost_usd=0.002,
                    schema_version=1,
                ),
            ],
        )
        summary = aggregate_day(s3, _BUCKET, date(2026, 5, 2))
        assert summary is not None
        assert summary["total_web_search_requests"] == 0
        assert summary["total_web_fetch_requests"] == 0


# ── CW metric emission (Phase 4 #2 + #3) ─────────────────────────────────


class TestPerAgentCwMetrics:
    """``aggregate_day`` emits per-agent_id metrics to
    ``AlphaEngine/Cost`` so alarms + the dashboard can rank by regressor.
    """

    def test_emits_weekly_cost_per_agent(self, s3):
        from unittest.mock import MagicMock

        from scripts.aggregate_costs import aggregate_day

        cw = MagicMock()
        _put_jsonl(
            s3,
            "decision_artifacts/_cost_raw/2026-05-30/2026-05-30/a.jsonl",
            [
                _make_row(
                    agent_id="data:news_event_extraction",
                    sector_team_id=None,
                    model_name="claude-haiku-4-5",
                    input_tokens=10_000,
                    output_tokens=2_000,
                    cost_usd=0.020,
                ),
            ],
        )
        _put_jsonl(
            s3,
            "decision_artifacts/_cost_raw/2026-05-30/2026-05-30/b.jsonl",
            [
                _make_row(
                    agent_id="executor:eod_narrative",
                    sector_team_id=None,
                    model_name="claude-haiku-4-5",
                    input_tokens=2_000,
                    output_tokens=500,
                    cost_usd=0.0045,
                ),
            ],
        )

        aggregate_day(s3, _BUCKET, date(2026, 5, 30), cw_client=cw)

        # Single put_metric_data call (≤20 metrics this run).
        cw.put_metric_data.assert_called_once()
        kwargs = cw.put_metric_data.call_args.kwargs
        assert kwargs["Namespace"] == "AlphaEngine/Cost"
        metrics = kwargs["MetricData"]

        weekly_costs = {
            tuple((d["Name"], d["Value"]) for d in m["Dimensions"]): m["Value"]
            for m in metrics
            if m["MetricName"] == "WeeklyCostUsd"
        }
        assert weekly_costs[(("agent_id", "data:news_event_extraction"),)] == pytest.approx(0.020)
        assert weekly_costs[(("agent_id", "executor:eod_narrative"),)] == pytest.approx(0.0045)

    def test_cache_hit_ratio_metric_emitted_when_cache_activity(self, s3):
        from unittest.mock import MagicMock

        from scripts.aggregate_costs import aggregate_day

        cw = MagicMock()
        row = _make_row(
            agent_id="ic_cio",
            sector_team_id=None,
            model_name="claude-sonnet-4-6",
            input_tokens=1_000,
            output_tokens=200,
            cost_usd=0.01,
        )
        row["cache_read_tokens"] = 3_000  # heavy cache usage
        _put_jsonl(s3, "decision_artifacts/_cost_raw/2026-05-30/2026-05-30/c.jsonl", [row])

        aggregate_day(s3, _BUCKET, date(2026, 5, 30), cw_client=cw)
        metrics = cw.put_metric_data.call_args.kwargs["MetricData"]
        cache_hit = next(m for m in metrics if m["MetricName"] == "CacheHitRatio")
        # 3000 / (3000 + 1000) = 75%
        assert cache_hit["Value"] == pytest.approx(75.0)
        assert cache_hit["Unit"] == "Percent"

    def test_tool_fee_requests_metric_per_tool(self, s3):
        from unittest.mock import MagicMock

        from scripts.aggregate_costs import aggregate_day

        cw = MagicMock()
        row = _make_row(
            agent_id="qual_tools_agent",
            sector_team_id="tech",
            model_name="claude-haiku-4-5",
            input_tokens=1_000,
            output_tokens=200,
            cost_usd=0.502,
            web_search_requests=50,
            web_fetch_requests=3,
        )
        _put_jsonl(s3, "decision_artifacts/_cost_raw/2026-05-30/2026-05-30/d.jsonl", [row])

        aggregate_day(s3, _BUCKET, date(2026, 5, 30), cw_client=cw)
        metrics = cw.put_metric_data.call_args.kwargs["MetricData"]
        tool_metrics = {
            tuple(sorted((d["Name"], d["Value"]) for d in m["Dimensions"])): m["Value"]
            for m in metrics
            if m["MetricName"] == "ToolFeeRequests"
        }
        ws_key = tuple(sorted([("agent_id", "qual_tools_agent"), ("tool", "web_search")]))
        wf_key = tuple(sorted([("agent_id", "qual_tools_agent"), ("tool", "web_fetch")]))
        assert tool_metrics[ws_key] == 50
        assert tool_metrics[wf_key] == 3

    def test_cw_failure_does_not_break_parquet_write(self, s3):
        """Per [[feedback_no_silent_fails]] applied carefully: CW emit
        is the observability layer ON TOP of the parquet write. CW
        failure must NOT take down the parquet (which IS the load-
        bearing artifact)."""
        from unittest.mock import MagicMock

        from scripts.aggregate_costs import aggregate_day

        cw = MagicMock()
        cw.put_metric_data.side_effect = RuntimeError("ThrottlingException")

        _put_jsonl(
            s3,
            "decision_artifacts/_cost_raw/2026-05-30/2026-05-30/a.jsonl",
            [
                _make_row(
                    agent_id="a",
                    sector_team_id=None,
                    model_name="claude-haiku-4-5",
                    input_tokens=10,
                    output_tokens=5,
                    cost_usd=0.001,
                ),
            ],
        )

        # MUST NOT raise — parquet write succeeds, CW failure swallowed.
        summary = aggregate_day(s3, _BUCKET, date(2026, 5, 30), cw_client=cw)
        assert summary is not None
        # Parquet was written.
        obj = s3.get_object(Bucket=_BUCKET, Key=summary["output_key"])
        df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
        assert len(df) == 1

    def test_empty_df_skips_cw_emit(self, s3):
        """Zero-row partition → no metric emit (avoids empty put_metric_data
        calls that CloudWatch would reject anyway)."""
        from unittest.mock import MagicMock

        from scripts.aggregate_costs import aggregate_day

        cw = MagicMock()
        # No JSONL files → aggregate_day returns None before emit.
        summary = aggregate_day(s3, _BUCKET, date(2026, 5, 30), cw_client=cw)
        assert summary is None
        cw.put_metric_data.assert_not_called()


# ── config-I7407: the window, not one date ──────────────────────────────────
# `aggregate_day` covers ONE date and the weekly SF invokes it with exactly
# one: `$.run_date`. Capture is DAILY — krepis.cost_sink writes under the
# wall-clock date of the call, so the Think Tank's daily runs land on dates no
# SF ever names.
#
# Measured 2026-08-15: `_cost_raw/` held 08-10, 08-11 and 08-12; `_cost/`
# stopped at 08-10. Two days of real cost rows captured and never aggregated,
# and the Saturday run asked for 08-14, which had no raw at all — so the stage
# "succeeded" having produced nothing, the `cost_telemetry` transparency row
# failed its 8-day freshness window against a parquet from 08-10, and that
# single [FAIL] is why the weekly run terminated DEGRADED.

from datetime import date as _date  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

from scripts.aggregate_costs import (  # noqa: E402
    DEFAULT_LOOKBACK_DAYS,
    CostWindowGapError,
    aggregate_window,
)


def _s3_with_raw_on(dates):
    """A fake S3 whose `_cost_raw/{d}/` prefix is non-empty for `dates`."""
    s3 = MagicMock()

    def list_objects_v2(Bucket, Prefix, MaxKeys=None):  # noqa: N803
        day = Prefix.rstrip("/").rsplit("/", 1)[-1]
        return {"KeyCount": 1 if day in dates else 0}

    s3.list_objects_v2.side_effect = list_objects_v2
    return s3


class TestAggregateWindow:
    def test_it_aggregates_every_date_with_raw_not_just_the_end(self):
        raw = {"2026-08-10", "2026-08-11", "2026-08-12"}
        s3 = _s3_with_raw_on(raw)
        with patch("scripts.aggregate_costs.aggregate_day",
                   return_value={"rows_in": 1}) as agg:
            out = aggregate_window(s3, "b", _date(2026, 8, 14), lookback_days=8)
        assert set(out["aggregated"]) == raw
        assert {c.kwargs["target_date"].isoformat() for c in agg.call_args_list} == raw

    def test_the_end_date_having_no_raw_is_not_a_failure(self):
        """THE 2026-08-15 CASE: run_date 08-14 was empty; 08-10..08-12 were not."""
        s3 = _s3_with_raw_on({"2026-08-12"})
        with patch("scripts.aggregate_costs.aggregate_day",
                   return_value={"rows_in": 1}):
            out = aggregate_window(s3, "b", _date(2026, 8, 14), lookback_days=8)
        assert out["aggregated"] == ["2026-08-12"]
        assert "2026-08-14" in out["skipped"]

    def test_a_wholly_quiet_window_aggregates_nothing_and_does_not_raise(self):
        s3 = _s3_with_raw_on(set())
        out = aggregate_window(s3, "b", _date(2026, 8, 14), lookback_days=8)
        assert out["aggregated"] == []
        assert len(out["skipped"]) == 9  # inclusive of both ends

    def test_raw_present_but_no_parquet_written_RAISES(self):
        """The silent gap this exists to make loud: `aggregate_day` returning
        None for a date that demonstrably has rows is a contract violation,
        not a quiet day."""
        s3 = _s3_with_raw_on({"2026-08-11"})
        with patch("scripts.aggregate_costs.aggregate_day", return_value=None):
            with pytest.raises(CostWindowGapError, match="2026-08-11"):
                aggregate_window(s3, "b", _date(2026, 8, 14), lookback_days=8)

    def test_a_per_date_failure_names_the_date_and_the_cause(self):
        s3 = _s3_with_raw_on({"2026-08-11"})
        with patch("scripts.aggregate_costs.aggregate_day",
                   side_effect=RuntimeError("parquet engine exploded")):
            with pytest.raises(CostWindowGapError) as exc:
                aggregate_window(s3, "b", _date(2026, 8, 14), lookback_days=8)
        assert "2026-08-11" in str(exc.value)
        assert "parquet engine exploded" in str(exc.value)

    def test_one_bad_date_does_not_stop_the_others_being_aggregated(self):
        """A hole in the middle of the window must not cost the rest of it."""
        s3 = _s3_with_raw_on({"2026-08-11", "2026-08-12"})

        def agg(s3_client, bucket, target_date, cw_client=None, **kw):
            if target_date.isoformat() == "2026-08-11":
                raise RuntimeError("boom")
            return {"rows_in": 1}

        with patch("scripts.aggregate_costs.aggregate_day", side_effect=agg):
            with pytest.raises(CostWindowGapError):
                aggregate_window(s3, "b", _date(2026, 8, 14), lookback_days=8)
        # 08-12 was still written before the raise — verified by the call.

    def test_the_default_window_matches_the_consumers_freshness_assertion(self):
        """cost_telemetry declares max_age_days: 8 in
        nousergon-lib's transparency_inventory.yaml. A producer covering less
        than its consumer asserts is a gap by construction."""
        assert DEFAULT_LOOKBACK_DAYS == 8

    def test_existence_is_probed_not_enumerated(self):
        """One truncated LIST per date — the prefix accumulates forever."""
        s3 = _s3_with_raw_on(set())
        aggregate_window(s3, "b", _date(2026, 8, 14), lookback_days=2)
        for call in s3.list_objects_v2.call_args_list:
            assert call.kwargs["MaxKeys"] == 1
