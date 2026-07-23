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
