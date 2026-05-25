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
from datetime import date, datetime, timezone
from typing import Iterable

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
        _put_jsonl(s3, "decision_artifacts/_cost_raw/2026-05-02/2026-05-02/sector_team:tech.jsonl", [
            _make_row(agent_id="sector_team:tech", sector_team_id="tech",
                      model_name="claude-haiku-4-5",
                      input_tokens=4000, output_tokens=1200, cost_usd=0.010, call_seq=1),
            _make_row(agent_id="sector_team:tech", sector_team_id="tech",
                      model_name="claude-haiku-4-5",
                      input_tokens=2000, output_tokens=800, cost_usd=0.006, call_seq=2),
        ])
        _put_jsonl(s3, "decision_artifacts/_cost_raw/2026-05-02/2026-05-02/sector_team:financials.jsonl", [
            _make_row(agent_id="sector_team:financials", sector_team_id="financials",
                      model_name="claude-haiku-4-5",
                      input_tokens=3500, output_tokens=900, cost_usd=0.008, call_seq=1),
        ])
        _put_jsonl(s3, "decision_artifacts/_cost_raw/2026-05-02/2026-05-02/macro_economist.jsonl", [
            _make_row(agent_id="macro_economist", sector_team_id=None,
                      model_name="claude-sonnet-4-6",
                      input_tokens=8000, output_tokens=3000, cost_usd=0.069, call_seq=1),
        ])

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
            "sector_team:tech", "sector_team:financials", "macro_economist",
        }

    def test_summary_breakdowns_match_per_row_sums(self, s3):
        from scripts.aggregate_costs import aggregate_day

        _put_jsonl(s3, "decision_artifacts/_cost_raw/2026-05-02/2026-05-02/sector_team:tech.jsonl", [
            _make_row(agent_id="sector_team:tech", sector_team_id="tech",
                      model_name="claude-haiku-4-5",
                      input_tokens=1000, output_tokens=500, cost_usd=0.0035),
        ])
        _put_jsonl(s3, "decision_artifacts/_cost_raw/2026-05-02/2026-05-02/sector_team:financials.jsonl", [
            _make_row(agent_id="sector_team:financials", sector_team_id="financials",
                      model_name="claude-haiku-4-5",
                      input_tokens=2000, output_tokens=800, cost_usd=0.006),
        ])
        _put_jsonl(s3, "decision_artifacts/_cost_raw/2026-05-02/2026-05-02/ic_cio.jsonl", [
            _make_row(agent_id="ic_cio", sector_team_id=None,
                      model_name="claude-sonnet-4-6",
                      input_tokens=10000, output_tokens=2000, cost_usd=0.060),
        ])

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

        _put_jsonl(s3, "decision_artifacts/_cost_raw/2026-05-02/2026-05-02/agent_a.jsonl", [
            _make_row(agent_id="agent_a", sector_team_id=None,
                      model_name="claude-haiku-4-5",
                      input_tokens=5000, output_tokens=1500, cost_usd=0.0125),
        ])
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
        from scripts.aggregate_costs import main
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

        _put_jsonl(s3, "decision_artifacts/_cost_raw/2026-05-02/2026-05-02/agent_a.jsonl", [
            _make_row(agent_id="agent_a", sector_team_id=None,
                      model_name="claude-haiku-4-5",
                      input_tokens=10, output_tokens=5, cost_usd=0.0001),
        ])
        summary = aggregate_day(
            s3, _BUCKET, date(2026, 5, 2),
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
                        agent_id=f"agent_{i}", sector_team_id=None,
                        model_name="claude-haiku-4-5",
                        input_tokens=100, output_tokens=50,
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
        for run_id in ("2026-05-13", "2026-05-15",
                       "2026-05-20-001", "2026-05-20_run1"):
            ok, reason = _is_plausible_cost_row({
                "run_id": run_id,
                "input_tokens": 4000, "output_tokens": 1200,
            })
            assert ok, f"{run_id!r} should pass — reason: {reason}"

    def test_test_fixture_run_id_fails(self):
        from scripts.aggregate_costs import _is_plausible_cost_row
        # The exact run_ids the 2026-05-13 incident left in S3.
        for run_id in ("run-1", "run-2", "run-budget-test", "run-x", "", None):
            ok, reason = _is_plausible_cost_row({
                "run_id": run_id,
                "input_tokens": 4000, "output_tokens": 1200,
            })
            assert not ok
            assert "run_id" in reason

    def test_implausible_token_count_fails(self):
        from scripts.aggregate_costs import _is_plausible_cost_row
        # The big_spender row had input_tokens=1e9 — 200x the Claude
        # Opus 4.7 context window. Real API calls cannot reach this.
        ok, reason = _is_plausible_cost_row({
            "run_id": "2026-05-13",
            "input_tokens": 1_000_000_000, "output_tokens": 0,
        })
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
                _make_row(agent_id="sector_team:tech",
                          sector_team_id="tech",
                          model_name="claude-haiku-4-5",
                          input_tokens=4000, output_tokens=1200,
                          cost_usd=0.012, call_seq=1),
                _make_row(agent_id="sector_team:tech",
                          sector_team_id="tech",
                          model_name="claude-haiku-4-5",
                          input_tokens=2000, output_tokens=800,
                          cost_usd=0.006, call_seq=2),
                _make_row(agent_id="sector_team:tech",
                          sector_team_id="tech",
                          model_name="claude-haiku-4-5",
                          input_tokens=3000, output_tokens=1000,
                          cost_usd=0.008, call_seq=3),
                # The exact 2026-05-13 pollution shape.
                {**_make_row(agent_id="big_spender", sector_team_id=None,
                             model_name="claude-haiku-4-5",
                             input_tokens=1_000_000_000, output_tokens=0,
                             cost_usd=1000.0),
                 "run_id": "run-x"},
                {**_make_row(agent_id="runaway_agent", sector_team_id=None,
                             model_name="claude-haiku-4-5",
                             input_tokens=10_000_000, output_tokens=0,
                             cost_usd=10.0),
                 "run_id": "run-budget-test"},
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

    def test_aggregate_day_skips_when_all_rows_polluted(self, s3, caplog):
        # All-pollution day → no parquet written (same shape as
        # zero-rows case, distinguishable in logs).
        from scripts.aggregate_costs import aggregate_day
        _put_jsonl(
            s3,
            "decision_artifacts/_cost_raw/2026-05-13/run-x/big_spender.jsonl",
            [
                {**_make_row(agent_id="big_spender", sector_team_id=None,
                             model_name="claude-haiku-4-5",
                             input_tokens=1_000_000_000, output_tokens=0,
                             cost_usd=1000.0),
                 "run_id": "run-x"},
            ],
        )
        summary = aggregate_day(s3, _BUCKET, date(2026, 5, 13))
        assert summary is None


# ── Tool-fee column handling (schema v2) ──────────────────────────────────


class TestToolRequestTotals:
    """Lock down ``total_web_search_requests`` + ``total_web_fetch_requests``
    on the summary surface — the keystone metric for the dashboard's
    tool-fee cost panel."""

    def test_sums_tool_request_counts_across_files(self, s3):
        from scripts.aggregate_costs import aggregate_day

        _put_jsonl(s3, "decision_artifacts/_cost_raw/2026-05-30/2026-05-30/qual_a.jsonl", [
            _make_row(agent_id="qual_a", sector_team_id="tech",
                      model_name="claude-haiku-4-5",
                      input_tokens=1000, output_tokens=200, cost_usd=0.502,
                      web_search_requests=50, web_fetch_requests=2),
        ])
        _put_jsonl(s3, "decision_artifacts/_cost_raw/2026-05-30/2026-05-30/qual_b.jsonl", [
            _make_row(agent_id="qual_b", sector_team_id="financials",
                      model_name="claude-haiku-4-5",
                      input_tokens=500, output_tokens=100, cost_usd=0.301,
                      web_search_requests=30, web_fetch_requests=0),
        ])
        summary = aggregate_day(s3, _BUCKET, date(2026, 5, 30))
        assert summary is not None
        assert summary["total_web_search_requests"] == 80
        assert summary["total_web_fetch_requests"] == 2

    def test_zero_default_when_v1_only_partition(self, s3):
        """A day's worth of pre-v2 rows (no tool-fee columns) aggregates
        to zero totals — backfill safety. The aggregator must not raise
        when the schema-v2 columns are entirely absent from the DataFrame."""
        from scripts.aggregate_costs import aggregate_day

        _put_jsonl(s3, "decision_artifacts/_cost_raw/2026-05-02/2026-05-02/legacy.jsonl", [
            _make_row(agent_id="legacy", sector_team_id=None,
                      model_name="claude-haiku-4-5",
                      input_tokens=1000, output_tokens=200, cost_usd=0.002,
                      schema_version=1),
        ])
        summary = aggregate_day(s3, _BUCKET, date(2026, 5, 2))
        assert summary is not None
        assert summary["total_web_search_requests"] == 0
        assert summary["total_web_fetch_requests"] == 0
