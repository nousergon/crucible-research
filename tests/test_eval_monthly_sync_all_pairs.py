"""The 2026-09-05 configuration, rehearsed: the MONTHLY all-pairs sweep on the
SYNCHRONOUS rung (alpha-engine-config-I9442).

## Why this file exists

2026-09-05 is the first Saturday of the month, so ``ComputeEvalCadence`` routes
the weekly Step Function to ``EvalJudgeSubmitFirstSaturday`` — the all-pairs
calibration sweep, ``force_sonnet_pass=True``, every mapped artifact judged by
BOTH tiers. It is also the first first-Saturday since the judge transport
changed: Brian's 2026-08-29 ruling (*"I will not fund the anthropic account, at
this point we shouldn't be using the anthropic api at all"*,
alpha-engine-config-I9263) removed the provider batch route, so ``submit_batch``
now degrades to the ``sync-{date}`` rung and ``process_batch_results`` judges
every plan entry itself.

Those two facts have never met. Measured against the suite on 2026-08-31:

  * ``force_sonnet_pass=True`` is exercised only against the BATCH rung
    (``test_eval_orchestrator_batch.py::test_first_saturday_path_skips_escalation_tail``),
    where the Sonnet entries arrive as batch results and the sync queue is empty;
  * the sync rung is exercised only against the WEEKLY plan
    (``force_sonnet_pass=False``), where every entry carries ``haiku_model``.

So the Sonnet half of the sync queue — 114 of the ~228 entries the monthly sweep
will plan — is a code path with no test and no production execution behind it,
and it runs unattended for the first time on a Saturday. That is the
rehearsal-only defect class this fleet keeps meeting (a Map ItemProcessor
reading an omitted ``ItemSelector`` key fails at runtime past valid ASL and
green CI); the answer is to run the configuration, not to read it.

## What each test pins, and why it is the failure that would actually happen

1. **Coverage over the doubled plan.** ``judge_coverage.enforce_coverage`` is a
   HARD failure (I9309) and coverage is measured over plan ENTRIES, not over
   artifacts. A monthly plan has two entries per artifact. Any grading path that
   silently handled only the Haiku half would return 50% coverage and fail the
   stage — the exact shape of ``EvalJudgeProcess`` going ``DegradedRun``.

2. **A CloudWatch datapoint per graded eval, on BOTH tiers.** This is the
   upstream half of ``EvalRollingMean``'s ``EvalFloorUnmeasurable``
   (alpha-engine-config-I9321/I9442): the floor is derived from
   ``AlphaEngine/Eval/agent_quality_score`` streams, so a tier that grades
   without emitting leaves the floor measuring half the corpus while every layer
   reports OK. ``principles.md`` §2.7 — a component emitting nothing is not
   healthy, it is unobserved.

3. **No escalation tail.** The tail re-judges borderline Haiku evals on Sonnet.
   On the monthly cadence every artifact ALREADY has a Sonnet eval, so a tail
   that ran here would double-bill the expensive tier on exactly the corpus that
   already paid for it — and would inflate ``sonnet_evaluated`` past the plan,
   which is how a coverage ledger stops meaning anything.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import boto3
import pytest
from moto import mock_aws
from nousergon_lib.decision_capture import (
    DecisionArtifact,
    FullPromptContext,
    ModelMetadata,
)

from evals import judge as judge_mod
from evals import orchestrator as orch

# Same test seam and rationale as tests/test_eval_orchestrator_batch.py: this
# suite exercises MECHANICS over arbitrary agent_id labels, so it must not be
# coupled to the live rubric taxonomy (alpha-engine-config-I9330).
_TEST_RUBRIC_MAP = {
    "sector_quant": "eval_rubric_sector_quant",
    "sector_qual": "eval_rubric_sector_qual",
    "sector_peer_review": "eval_rubric_sector_peer_review",
    "macro_economist": "eval_rubric_macro_economist",
    "ic_cio": "eval_rubric_ic_cio",
    "thesis_update": "eval_rubric_thesis_update",
}


def _synthetic_resolve_rubric_for_agent(agent_id: str) -> str | None:
    for prefix, rubric in _TEST_RUBRIC_MAP.items():
        if agent_id == prefix or agent_id.startswith(f"{prefix}:"):
            return rubric
    return None


@pytest.fixture(autouse=True)
def _synthetic_rubric_map(monkeypatch):
    monkeypatch.setattr(
        orch, "resolve_rubric_for_agent", _synthetic_resolve_rubric_for_agent,
    )
    monkeypatch.setattr(
        judge_mod, "resolve_rubric_for_agent", _synthetic_resolve_rubric_for_agent,
    )


_MONTHLY_DATE = "2026-09-05"
_PARTITION = "decision_artifacts/2026/09/05"

# The three mapped agents this corpus carries. Named, not counted, so a
# fixture edit that changes the corpus cannot silently change what "full
# coverage" means below.
_MAPPED_AGENTS = ("ic_cio", "sector_quant:technology", "macro_economist")


def _make_capture_dict(agent_id: str, *, run_id: str = "run-1") -> dict:
    if agent_id.startswith("sector_quant:"):
        snapshot = {
            "sector_tickers": ["AAPL"],
            "sector_tickers_count": 1,
            "technical_scores_team": {"AAPL": {"rsi_14": 55}},
        }
    else:
        snapshot = {"k": "v"}
    return DecisionArtifact(
        run_id=run_id,
        timestamp=f"{_MONTHLY_DATE}T22:30:00.000Z",
        agent_id=agent_id,
        model_metadata=ModelMetadata(model_name="claude-haiku-4-5"),
        full_prompt_context=FullPromptContext(
            system_prompt="<see config/prompts>",
            user_prompt="<rendered>",
        ),
        input_data_snapshot=snapshot,
        input_data_summary="k=v",
        agent_output={"out": "ok"},
    ).model_dump()


@pytest.fixture
def monthly_s3():
    """A first-Saturday capture partition: three mapped agents plus one
    unmapped agent that must be dropped from the plan on BOTH tiers."""
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="alpha-engine-research")
        for agent_id in (*_MAPPED_AGENTS, "unknown_xyz"):
            client.put_object(
                Bucket="alpha-engine-research",
                Key=f"{_PARTITION}/{agent_id}/run-1.json",
                Body=json.dumps(
                    _make_capture_dict(agent_id), default=str,
                ).encode("utf-8"),
            )
        yield client


def _fake_evaluate(artifact, *, judge_run_id, judge_model, judged_artifact_s3_key, **kw):
    """Stand in for the router-addressed judge call.

    Returns a real ``RubricEvalArtifact`` (not a mock) so persistence,
    the coverage ledger and CloudWatch emission all run their real code —
    the stub replaces the LLM call, nothing else.
    """
    from graph.state_schemas import RubricDimensionScore, RubricEvalArtifact

    return RubricEvalArtifact(
        run_id=artifact.run_id,
        judge_run_id=judge_run_id,
        timestamp=f"{_MONTHLY_DATE}T22:30:00Z",
        judged_agent_id=artifact.agent_id,
        rubric_id="eval_rubric_test",
        rubric_version="1.0.0",
        judge_model=judge_model,
        dimension_scores=[
            # Deliberately BELOW the escalation threshold. On the weekly
            # cadence this artifact would be re-judged on Sonnet; on the
            # monthly cadence it must not be, and a fixture scoring 4s
            # everywhere would pass test 3 without ever arming it.
            RubricDimensionScore(dimension="d", score=1, reasoning="r"),
        ],
        overall_reasoning="ok",
    )


def _submit_on_the_sync_rung(s3):
    """Build the first-Saturday plan and submit it with the batch capability
    unavailable — i.e. exactly what the live Lambda does since I9263."""
    from evals.judge_batch_transport import BatchCapabilityUnavailable

    plan = orch.build_batch_plan(
        date=_MONTHLY_DATE,
        bucket="alpha-engine-research",
        force_sonnet_pass=True,
        s3_client=s3,
    )

    def _no_batch_route(**kwargs):
        raise BatchCapabilityUnavailable(
            group="judge",
            capability="batches",
            exec_context="lambda",
            reason="no member of the group declares the batches capability",
        )

    with patch.object(orch, "resolve_batch_transport", side_effect=_no_batch_route):
        submit_result = orch.submit_batch(plan, s3_client=s3)
    return plan, submit_result


class TestMonthlyAllPairsOnTheSyncRung:
    def test_plan_carries_both_tiers_and_degrades_to_the_sync_rung(self, monthly_s3):
        """The 09-05 entry conditions, asserted before anything is judged:
        one entry per (artifact, tier) and a ``sync-`` batch id."""
        plan, submit_result = _submit_on_the_sync_rung(monthly_s3)

        assert len(plan["plan_entries"]) == 2 * len(_MAPPED_AGENTS)
        assert submit_result["processing_status"] == "ended_sync"
        assert submit_result["batch_id"] == f"sync-{_MONTHLY_DATE}"
        assert submit_result["request_count"] == 2 * len(_MAPPED_AGENTS)
        # The drop off the batch rung is recorded durably, not inferred.
        assert submit_result["degraded"] is True
        assert submit_result["degradation_s3_key"]

    def test_every_planned_entry_is_graded_so_coverage_passes(self, monthly_s3):
        """The stage-failing case. Coverage is measured over plan ENTRIES;
        a monthly plan has two per artifact, and grading only the Haiku half
        is a 50% shortfall that ``enforce_coverage`` turns into a
        DegradedRun."""
        from evals.judge_coverage import assess_coverage, enforce_coverage

        _, submit_result = _submit_on_the_sync_rung(monthly_s3)

        with patch.object(orch, "evaluate_artifact", side_effect=_fake_evaluate):
            summary = orch.process_batch_results(
                batch_id=submit_result["batch_id"],
                plan_s3_key=submit_result["plan_s3_key"],
                bucket="alpha-engine-research",
                s3_client=monthly_s3,
                emit_metrics=False,
            )

        n = len(_MAPPED_AGENTS)
        assert summary["sync_fallback_evaluated"] == 2 * n
        assert summary["haiku_evaluated"] == n
        assert summary["sonnet_evaluated"] == n
        assert summary["complete"] is True
        assert summary["degraded_transport"] is True

        verdict = assess_coverage(summary)
        assert verdict["planned"] == 2 * n
        assert verdict["graded"] == 2 * n
        assert verdict["ungraded"] == 0
        assert verdict["status"] == "PASS"
        # The gate the Step Function actually reads — it must not raise.
        enforce_coverage(summary)

    def test_both_tiers_emit_a_quality_datapoint(self, monthly_s3):
        """``EvalRollingMean`` derives the floor from
        ``AlphaEngine/Eval/agent_quality_score``. A tier that grades without
        emitting is the upstream half of ``EvalFloorUnmeasurable``
        (alpha-engine-config-I9321): the corpus looks judged and the floor is
        computed over half of it."""
        _, submit_result = _submit_on_the_sync_rung(monthly_s3)
        cw = MagicMock()

        with patch.object(orch, "evaluate_artifact", side_effect=_fake_evaluate):
            summary = orch.process_batch_results(
                batch_id=submit_result["batch_id"],
                plan_s3_key=submit_result["plan_s3_key"],
                bucket="alpha-engine-research",
                s3_client=monthly_s3,
                cloudwatch_client=cw,
                emit_metrics=True,
            )

        assert summary["metric_emission_failures"] == 0

        emitted_judges: list[str] = []
        for call in cw.put_metric_data.call_args_list:
            for datum in call.kwargs["MetricData"]:
                dims = {d["Name"]: d["Value"] for d in datum["Dimensions"]}
                if "judge_model" in dims:
                    emitted_judges.append(dims["judge_model"])

        assert emitted_judges, "no agent_quality_score datapoint was emitted"
        haiku = orch.DEFAULT_HAIKU_MODEL
        sonnet = orch.DEFAULT_SONNET_MODEL
        assert emitted_judges.count(haiku) > 0
        assert emitted_judges.count(sonnet) > 0
        # Both tiers must be represented equally: the sweep is all-pairs.
        assert emitted_judges.count(haiku) == emitted_judges.count(sonnet)

    def test_escalation_tail_does_not_run(self, monthly_s3):
        """Every fixture eval scores below the escalation threshold, so the
        weekly cadence WOULD escalate all of them. The monthly cadence must
        not: those artifacts already carry a Sonnet eval, and a tail here
        double-bills the expensive tier and inflates the coverage ledger."""
        _, submit_result = _submit_on_the_sync_rung(monthly_s3)

        with patch.object(
            orch, "evaluate_artifact", side_effect=_fake_evaluate,
        ) as ea_mock:
            summary = orch.process_batch_results(
                batch_id=submit_result["batch_id"],
                plan_s3_key=submit_result["plan_s3_key"],
                bucket="alpha-engine-research",
                s3_client=monthly_s3,
                emit_metrics=False,
            )

        n = len(_MAPPED_AGENTS)
        # Exactly one judge call per plan entry — no tail, no re-judge.
        assert ea_mock.call_count == 2 * n
        assert summary["sonnet_evaluated"] == n
        assert "sonnet_escalation" not in summary["budget_stopped_phases"]
