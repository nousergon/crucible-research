"""Think-tank → LLM-as-judge wiring (config#1579 P2).

Covers: rubric registration, capture emission at thesis/theme writes,
the family-selection seams on the batch enumeration (agent_id_prefixes +
extra_dates), and plan building over captured thinktank artifacts.
"""

from __future__ import annotations

import json

import boto3
from moto import mock_aws

from evals.judge import resolve_rubric_for_agent
from evals.orchestrator import build_batch_plan, list_capture_keys
from thinktank.capture import THEME_AGENT_ID, THESIS_AGENT_ID
from thinktank.client import LLMCallResult

BUCKET = "alpha-engine-research"


# ── rubric registration ──────────────────────────────────────────────────────


def test_thinktank_agent_ids_resolve_to_rubrics():
    assert resolve_rubric_for_agent("thinktank_thesis") == "eval_rubric_thinktank_thesis"
    assert resolve_rubric_for_agent("thinktank_theme") == "eval_rubric_thinktank_theme"
    # coarse ids only — a per-ticker id must NOT silently map (low-N floor lesson)
    assert resolve_rubric_for_agent("thinktank_thesis:AAPL") is None


# ── capture emission ─────────────────────────────────────────────────────────


def _fake_result() -> LLMCallResult:
    from pydantic import BaseModel

    class _Stub(BaseModel):
        ok: bool = True

    return LLMCallResult(
        parsed=_Stub(),
        raw_text="{}",
        model="fake/model",
        tier="thesis",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
    )


def test_thesis_capture_emits_decision_artifact(monkeypatch):
    monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "true")
    from thinktank.capture import emit_thesis_capture

    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        key = emit_thesis_capture(
            base_run_id="run1",
            ticker="AAPL",
            version=1,
            trading_day="2026-07-01",
            result=_fake_result(),
            system="sys",
            user="usr",
            prompt_version_hash="abc",
            input_data_snapshot={"ticker": "AAPL", "board_row": {"x": 1}},
            agent_output={"ticker": "AAPL", "thesis": {"stance": "attractive"}},
            bucket=BUCKET,
            s3_client=s3,
        )
        assert key is not None and f"/{THESIS_AGENT_ID}/" in key
        # partitioned by TRADING day, not capture wall-clock date
        assert key.startswith("decision_artifacts/2026/07/01/")
        artifact = json.loads(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read())
        assert artifact["agent_id"] == THESIS_AGENT_ID
        assert artifact["run_id"] == "run1-AAPL-v1"
        assert artifact["agent_output"]["thesis"]["stance"] == "attractive"
        assert artifact["model_metadata"]["model_name"] == "fake/model"


def test_capture_disabled_is_noop(monkeypatch):
    monkeypatch.delenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", raising=False)
    from thinktank.capture import emit_theme_capture

    assert (
        emit_theme_capture(
            base_run_id="r",
            kind="macro",
            key_slug="macro",
            version=1,
            trading_day="2026-07-01",
            result=_fake_result(),
            system="s",
            user="u",
            prompt_version_hash=None,
            input_data_snapshot={"kind": "macro"},
            agent_output={"kind": "macro"},
            bucket=BUCKET,
            s3_client=None,
        )
        is None
    )


# ── enumeration seams ────────────────────────────────────────────────────────


def _put_capture(s3, *, date: str, agent_id: str, run_id: str):
    y, m, d = date.split("-")
    key = f"decision_artifacts/{y}/{m}/{d}/{agent_id}/{run_id}.json"
    body = {
        "schema_version": 2,
        "run_id": run_id,
        "timestamp": f"{date}T12:00:00+00:00",
        "agent_id": agent_id,
        "model_metadata": None,
        "full_prompt_context": None,
        "input_data_snapshot": {"x": 1},
        "agent_output": {"y": 2},
    }
    s3.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(body))
    return key


def test_list_capture_keys_agent_prefix_filter():
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        k1 = _put_capture(s3, date="2026-07-02", agent_id=THESIS_AGENT_ID, run_id="a")
        k2 = _put_capture(s3, date="2026-07-02", agent_id=THEME_AGENT_ID, run_id="b")
        k3 = _put_capture(s3, date="2026-07-02", agent_id="ic_cio", run_id="c")

        allk = list_capture_keys(s3, date="2026-07-02", bucket=BUCKET)
        assert {k1, k2, k3} <= set(allk)

        fam = list_capture_keys(
            s3, date="2026-07-02", bucket=BUCKET, agent_id_prefixes=["thinktank_"]
        )
        assert set(fam) == {k1, k2}


def test_build_batch_plan_extra_dates_and_family_filter(monkeypatch, tmp_path):
    # rubric prompts must resolve — they do via the config-repo checkout
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        # thinktank artifacts across two weekday partitions + one graph
        # artifact on the same dates that must be filtered OUT
        _put_capture(s3, date="2026-06-29", agent_id=THESIS_AGENT_ID, run_id="r1")
        _put_capture(s3, date="2026-06-30", agent_id=THEME_AGENT_ID, run_id="r2")
        _put_capture(s3, date="2026-06-30", agent_id="ic_cio", run_id="r3")

        plan = build_batch_plan(
            date="2026-06-29",
            extra_dates=["2026-06-30"],
            agent_id_prefixes=["thinktank_"],
            bucket=BUCKET,
            s3_client=s3,
        )
        agent_ids = sorted(e["agent_id"] for e in plan["plan_entries"])
        assert agent_ids == sorted([THESIS_AGENT_ID, THEME_AGENT_ID])
        # weekly-cadence default: one Haiku entry per artifact
        assert len(plan["requests"]) == 2


def test_build_batch_plan_default_shape_unchanged():
    """No extra params → single-date enumeration, exactly as before."""
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        _put_capture(s3, date="2026-07-02", agent_id="ic_cio", run_id="r1")
        _put_capture(s3, date="2026-07-01", agent_id="ic_cio", run_id="r0")
        plan = build_batch_plan(date="2026-07-02", bucket=BUCKET, s3_client=s3)
        assert [e["run_id"] for e in plan["plan_entries"]] == ["r1"]


def test_expand_lookback_dates_trading_days():
    from evals.orchestrator import expand_lookback_dates

    # Sat 2026-07-04: Fri 7/3 is the July-4th observed NYSE holiday —
    # the lookback must skip both the weekend AND the holiday. (This is
    # the first scheduled Saturday pass, so the case is live.)
    assert expand_lookback_dates("2026-07-04", 6) == [
        "2026-07-02", "2026-07-01", "2026-06-30",
        "2026-06-29", "2026-06-26", "2026-06-25",
    ]
    assert expand_lookback_dates("2026-07-04", 0) == []
    # plain weekend crossing
    assert expand_lookback_dates("2026-07-01", 3) == [
        "2026-06-30", "2026-06-29", "2026-06-26",
    ]


def test_effective_capture_ceiling_date_clears_the_write_lag():
    """alpha-engine-config-I9331: Think Tank writes day D's captures on
    D+1 ~14:35 UTC; the weekly judge enters ~05:11 UTC on its anchor day.
    The ceiling must be at least 2 calendar days behind the anchor so
    D+1's write always lands on a day strictly before the judge runs."""
    from evals.orchestrator import (
        CAPTURE_WRITE_SETTLE_DAYS,
        effective_capture_ceiling_date,
    )

    assert CAPTURE_WRITE_SETTLE_DAYS >= 2

    # The measured incident date: weekly SF anchored 2026-08-29 (Saturday),
    # entered EvalJudgeSubmitWeekly at 05:11 UTC while decision_artifacts/
    # 2026/08/28/ (Friday) was still 9+ hours from being written.
    ceiling = effective_capture_ceiling_date("2026-08-29")
    assert ceiling == "2026-08-27"
    # RED against the pre-fix wiring: the old code passed the raw anchor
    # straight to expand_lookback_dates, whose newest returned date is
    # `previous_trading_day(anchor)` — exactly the partition that had not
    # been written yet.
    from krepis.trading_calendar import previous_trading_day
    from datetime import date as _date

    unsafe_newest = str(previous_trading_day(_date(2026, 8, 29)))
    assert unsafe_newest == "2026-08-28"
    assert ceiling != unsafe_newest


def test_compute_judge_window_dates_never_includes_the_unsafe_partition():
    from evals.orchestrator import compute_judge_window_dates

    window = compute_judge_window_dates("2026-08-29", 6)
    assert "2026-08-28" not in window  # written same day as the judge run
    assert "2026-08-29" not in window  # the (non-trading) anchor itself
    assert window[0] == "2026-08-27"   # newest safe partition
    assert len(window) == 7


def test_compute_judge_window_dates_recovers_the_deferred_day_next_cycle():
    """The Friday a week's window defers is squarely inside the FOLLOWING
    week's unchanged lookback — deferred one cycle, never dropped."""
    from evals.orchestrator import compute_judge_window_dates

    this_week = compute_judge_window_dates("2026-08-29", 6)
    next_week = compute_judge_window_dates("2026-09-05", 6)
    assert "2026-08-28" not in this_week
    assert "2026-08-28" in next_week


def test_build_batch_plan_names_an_empty_expected_partition():
    """alpha-engine-config-I9331 deliverable 3: a trading day inside the
    window with zero captures is a named finding, not a silent absence.
    A non-trading day in the window (the raw anchor here) is not flagged."""
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        _put_capture(s3, date="2026-07-01", agent_id="ic_cio", run_id="r0")
        # 2026-07-02 (a trading day) is left empty on purpose.
        plan = build_batch_plan(
            date="2026-07-02",
            extra_dates=["2026-07-01"],
            bucket=BUCKET,
            s3_client=s3,
        )
        assert plan["capture_partition_counts"] == {
            "2026-07-02": 0, "2026-07-01": 1,
        }
        assert plan["empty_trading_day_partitions"] == ["2026-07-02"]


def test_build_batch_plan_skips_already_judged(monkeypatch):
    """Weekend-boundary correctness: a re-enumerated partition only
    contributes captures no ACTUAL eval has scored yet."""
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        k1 = _put_capture(s3, date="2026-06-29", agent_id=THESIS_AGENT_ID, run_id="r1")
        k2 = _put_capture(s3, date="2026-06-29", agent_id=THESIS_AGENT_ID, run_id="r2")
        # r1 was judged by a prior batch — indexed in _eval_by_capture
        s3.put_object(
            Bucket=BUCKET,
            Key="decision_artifacts/_eval_by_capture/2026-06-29/manifest.json",
            Body=json.dumps({"entries": [{"judged_artifact_s3_key": k1}]}),
        )
        plan = build_batch_plan(
            date="2026-07-04",
            extra_dates=["2026-06-29"],
            agent_id_prefixes=["thinktank_"],
            bucket=BUCKET,
            s3_client=s3,
        )
        assert [e["run_id"] for e in plan["plan_entries"]] == ["r2"]
        assert plan["skipped_already_judged"] == 1
        assert k2 not in {""}  # keep k2 referenced


def test_single_date_plan_skips_dedup_lookup():
    """Default single-date invocation must not consult manifests at all
    (byte-identical legacy behavior)."""
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        k1 = _put_capture(s3, date="2026-07-04", agent_id="ic_cio", run_id="r1")
        # a (bogus) manifest claiming r1 was judged must be IGNORED on
        # the single-date path
        s3.put_object(
            Bucket=BUCKET,
            Key="decision_artifacts/_eval_by_capture/2026-07-04/manifest.json",
            Body=json.dumps({"entries": [{"judged_artifact_s3_key": k1}]}),
        )
        plan = build_batch_plan(date="2026-07-04", bucket=BUCKET, s3_client=s3)
        assert [e["run_id"] for e in plan["plan_entries"]] == ["r1"]
        assert plan["skipped_already_judged"] == 0
