"""Fan-in coverage: the SET comparison a count could not make.

alpha-engine-config-I7179 — measured on five dates, every per-call cost
record under ``decision_artifacts/_cost_raw/`` came from the Think Tank,
which left the weekly pipeline on 2026-08-10, while every LLM-calling
stage of that pipeline emitted nothing. Every scalar on the dashboard was
correct; the number was about something else.
"""

from __future__ import annotations

import pytest

from scripts.cost_coverage import (
    CostCoverageError,
    CostCoverageUnmeasured,
    evaluate_coverage,
    observed_producers,
    stages_entered,
)

DECL = {
    "execution_arn": "arn:aws:states:us-east-1:1:execution:ne-weekly-freshness-pipeline:e1",
    "required_producers": {
        "ChallengerShadow": ["single-agent-quant"],
        "EvalJudgeProcess": ["evaljudge-batch"],
        "ReplayConcordance": ["replay-concordance"],
    },
    "conditional_producers": {"EvalJudgeProcess": ["evaljudge-sync"]},
    "allowed_producers": ["thinktank-*", "director-plan"],
}

ALL_STAGES = {"ChallengerShadow", "EvalJudgeProcess", "ReplayConcordance"}
FULL = {"single-agent-quant", "evaljudge-batch", "replay-concordance"}


class TestObservedProducers:
    def test_producer_is_the_key_stem_without_the_flush_sequence(self):
        keys = [
            "decision_artifacts/_cost_raw/2026-08-16/run-1/replay-concordance.0.jsonl",
            "decision_artifacts/_cost_raw/2026-08-16/run-1/replay-concordance.1.jsonl",
            "decision_artifacts/_cost_raw/2026-08-16/run-2/director-plan.0.jsonl",
        ]
        assert observed_producers(keys) == {"replay-concordance", "director-plan"}

    def test_non_jsonl_objects_are_ignored(self):
        assert observed_producers(["a/b/cost.parquet", "a/b/_SUCCESS"]) == set()

    def test_a_dotted_callsite_id_is_not_truncated(self):
        """Only a purely numeric final segment is the flush sequence — a
        callsite id may contain dots, and silently truncating one would
        invent a producer nobody declared."""
        assert observed_producers(["p/d/r/my.callsite.id.3.jsonl"]) == {
            "my.callsite.id"
        }
        assert observed_producers(["p/d/r/my.callsite.id.jsonl"]) == {
            "my.callsite.id"
        }

    def test_an_empty_object_still_counts_as_a_producer(self):
        """Derived from the KEY, not from row contents: a producer that
        wrote a malformed object still ran, and scoring its silence as
        absence would report the wrong defect."""
        assert observed_producers(["p/d/r/single-agent-quant.0.jsonl"]) == {
            "single-agent-quant"
        }


class TestTheDefectItself:
    def test_the_i7179_state_of_the_world_fails(self):
        """Five Think Tank files, every pipeline stage silent. This is the
        exact partition measured on 2026-07-30 and 2026-08-01."""
        observed = {
            "thinktank-pillar",
            "thinktank-sweep",
            "thinktank-themes-macro",
            "thinktank-themes-sector",
            "thinktank-thesis",
        }
        with pytest.raises(CostCoverageError) as exc:
            evaluate_coverage(
                observed=observed, declaration=DECL, entered=ALL_STAGES
            )
        message = str(exc.value)
        assert "replay-concordance" in message
        assert "single-agent-quant" in message
        assert "evaljudge-batch" in message

    def test_a_count_would_have_passed_the_same_partition(self):
        """The point of the whole module, asserted rather than asserted in
        prose: the failing case has MORE files than the healthy one."""
        broken = {f"thinktank-{n}" for n in "abcde"}
        healthy = FULL
        assert len(broken) > len(healthy)
        evaluate_coverage(observed=healthy, declaration=DECL, entered=ALL_STAGES)


class TestMissingDirection:
    def test_full_coverage_passes(self):
        verdict = evaluate_coverage(
            observed=FULL, declaration=DECL, entered=ALL_STAGES
        )
        assert verdict["missing"] == []
        assert verdict["undeclared"] == []
        assert verdict["covered"] == sorted(FULL)

    def test_one_silent_stage_fails(self):
        with pytest.raises(CostCoverageError, match="replay-concordance"):
            evaluate_coverage(
                observed=FULL - {"replay-concordance"},
                declaration=DECL,
                entered=ALL_STAGES,
            )

    def test_a_stage_that_did_not_run_is_not_required(self):
        """A degraded run legitimately skips stages. A check that fired on
        every degraded run would be turned off, and then the real breach
        would go unreported."""
        verdict = evaluate_coverage(
            observed={"single-agent-quant", "evaljudge-batch"},
            declaration=DECL,
            entered=ALL_STAGES - {"ReplayConcordance"},
        )
        assert verdict["stages_not_entered"] == ["ReplayConcordance"]
        assert verdict["missing"] == []


class TestUndeclaredDirection:
    def test_the_think_tank_is_allowed_by_pattern(self):
        evaluate_coverage(
            observed=FULL | {"thinktank-sweep", "thinktank-thesis"},
            declaration=DECL,
            entered=ALL_STAGES,
        )

    def test_director_plan_is_allowed_but_never_required(self):
        """Director runs at the TOP LEVEL, after the Parallel that contains
        AggregateCosts, so its rows cannot be in this parquet. Requiring it
        would fail every healthy run; refusing it would fail every run in
        which it happened to write first."""
        verdict = evaluate_coverage(
            observed=FULL | {"director-plan"},
            declaration=DECL,
            entered=ALL_STAGES,
        )
        assert "director-plan" not in verdict["expected"]
        assert verdict["undeclared"] == []

    def test_a_brand_new_producer_is_refused(self):
        """Since krepis 0.57.0 emits from the environment, a newly added
        LLM stage emits BY CONSTRUCTION. It therefore arrives here as an
        undeclared producer, and is refused until the Step Function names
        it — which is what stops the next stage added from reproducing the
        gap this whole issue is about."""
        with pytest.raises(CostCoverageError, match="brand-new-stage"):
            evaluate_coverage(
                observed=FULL | {"brand-new-stage"},
                declaration=DECL,
                entered=ALL_STAGES,
            )

    def test_a_conditional_producer_is_allowed_when_its_stage_ran(self):
        evaluate_coverage(
            observed=FULL | {"evaljudge-sync"},
            declaration=DECL,
            entered=ALL_STAGES,
        )

    def test_a_conditional_producer_is_never_required(self):
        verdict = evaluate_coverage(
            observed=FULL, declaration=DECL, entered=ALL_STAGES
        )
        assert "evaljudge-sync" not in verdict["expected"]

    def test_missing_is_reported_before_undeclared(self):
        """Both are breaches, but a silent stage is the one that costs
        money invisibly; an undeclared producer is a registration gap. The
        message a human reads first should be the expensive one."""
        with pytest.raises(CostCoverageError) as exc:
            evaluate_coverage(
                observed={"brand-new-stage"},
                declaration=DECL,
                entered=ALL_STAGES,
            )
        assert "ran and" in str(exc.value)


class FakeSfn:
    def __init__(self, pages=None, error=None):
        self.pages = pages if pages is not None else []
        self.error = error

    def get_paginator(self, name):
        outer = self

        class _P:
            def paginate(self, **kw):
                if outer.error:
                    raise outer.error
                return outer.pages

        return _P()


class TestStagesEntered:
    def test_reads_state_names_from_the_history(self):
        pages = [
            {"events": [
                {"type": "TaskStateEntered",
                 "stateEnteredEventDetails": {"name": "ReplayConcordance"}},
                {"type": "TaskStateExited"},
            ]},
            {"events": [
                {"type": "ChoiceStateEntered",
                 "stateEnteredEventDetails": {"name": "CheckSkipAggregateCosts"}},
            ]},
        ]
        assert stages_entered("arn:x", sfn_client=FakeSfn(pages)) == {
            "ReplayConcordance",
            "CheckSkipAggregateCosts",
        }

    def test_a_missing_arn_is_unmeasured_not_empty(self):
        """An empty set would drop every required producer out of the
        expected set, and the check would pass for exactly the reason it
        should have failed."""
        with pytest.raises(CostCoverageUnmeasured):
            stages_entered("", sfn_client=FakeSfn())

    def test_an_api_failure_is_unmeasured_not_a_finding(self):
        """A harness fault reported as a domain finding is a defect this
        fleet has shipped repeatedly, always in the alarming direction."""
        with pytest.raises(CostCoverageUnmeasured) as exc:
            stages_entered("arn:x", sfn_client=FakeSfn(error=RuntimeError("denied")))
        assert "NOT a coverage finding" in str(exc.value)

    def test_zero_states_is_unmeasured_not_universal_absence(self):
        with pytest.raises(CostCoverageUnmeasured):
            stages_entered("arn:x", sfn_client=FakeSfn([{"events": []}]))

    def test_unmeasured_is_a_different_exception_from_a_breach(self):
        assert not issubclass(CostCoverageUnmeasured, CostCoverageError)
        assert not issubclass(CostCoverageError, CostCoverageUnmeasured)


# ── alpha-engine-config-I9261 ────────────────────────────────────────────
#
# The 2026-08-29 weekly run terminated DegradedRun on
# `weekly_aggregate_costs_fail_open` reporting that `replay-concordance`
# "ran and emitted no cost record". It had emitted — 24 priced rows,
# $0.0182848, flushed one second after the stage exited — into
# `_cost_raw/2026-08-29/`, while the check read `_cost_raw/2026-08-28/`
# (the run_date). The five thinktank producers it reported as `observed`
# came from Friday's DAILY pipelines; this execution's `stages_entered`
# was ['ChallengerShadow', 'ReplayConcordance'].

import datetime as _dt


class _WindowS3:
    """S3 fake keyed by prefix, carrying LastModified like the real API."""

    def __init__(self, objects: dict[str, _dt.datetime]):
        self._objects = objects

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        outer = self

        class _P:
            def paginate(self, **kwargs):
                prefix = kwargs["Prefix"]
                contents = [
                    {"Key": k, "LastModified": ts}
                    for k, ts in sorted(outer._objects.items())
                    if k.startswith(prefix)
                ]
                return [{"Contents": contents} if contents else {}]

        return _P()


_ROOT = "decision_artifacts/_cost_raw"
_STARTED = _dt.datetime(2026, 8, 29, 9, 0, 49, tzinfo=_dt.timezone.utc)
_NOW = _dt.datetime(2026, 8, 29, 14, 2, 40, tzinfo=_dt.timezone.utc)


def _observed(**kw):
    from scripts.cost_coverage import observed_producers_for_execution

    return observed_producers_for_execution(
        kw.pop("s3"), "alpha-engine-research", _ROOT,
        started_at=_STARTED, now=_NOW, **kw,
    )


def test_a_producer_in_tomorrows_partition_is_this_executions_producer():
    """The exact live failure. run_date=2026-08-28, records in 2026-08-29.

    The weekly SF starts 09:00 UTC on the day AFTER its run_date, so every
    one of its producers writes to a partition the run_date-keyed read
    never opened.
    """
    s3 = _WindowS3({
        f"{_ROOT}/2026-08-29/krepis-b1444bfa3454/replay-concordance.0.jsonl":
            _dt.datetime(2026, 8, 29, 12, 27, 43, tzinfo=_dt.timezone.utc),
    })
    assert _observed(s3=s3) == {"replay-concordance"}


def test_another_runs_object_in_the_run_date_partition_is_not_coverage():
    """`single-agent-quant` scored `covered` off a 14-hours-earlier object.

    That coincidence is the only reason the live breach named one stage
    rather than two, and it is why the published `observed` set described
    a different execution. An object flushed before this execution started
    cannot be this execution's producer.
    """
    s3 = _WindowS3({
        # Friday's daily pipelines, hours before the weekly run started.
        f"{_ROOT}/2026-08-28/8e9fff668cc8/thinktank-sweep.0.jsonl":
            _dt.datetime(2026, 8, 28, 15, 16, 30, tzinfo=_dt.timezone.utc),
        f"{_ROOT}/2026-08-28/krepis-7faeb93de042/single-agent-quant.0.jsonl":
            _dt.datetime(2026, 8, 28, 22, 20, 7, tzinfo=_dt.timezone.utc),
    })
    assert _observed(s3=s3) == set()


def test_the_execution_window_spans_the_dates_it_ran_across():
    """A run that crosses UTC midnight writes to both partitions."""
    s3 = _WindowS3({
        f"{_ROOT}/2026-08-29/r1/replay-concordance.0.jsonl":
            _dt.datetime(2026, 8, 29, 12, 0, tzinfo=_dt.timezone.utc),
        f"{_ROOT}/2026-08-30/r2/single-agent-quant.0.jsonl":
            _dt.datetime(2026, 8, 30, 1, 0, tzinfo=_dt.timezone.utc),
    })
    from scripts.cost_coverage import observed_producers_for_execution

    observed = observed_producers_for_execution(
        s3, "alpha-engine-research", _ROOT,
        started_at=_STARTED,
        now=_dt.datetime(2026, 8, 30, 2, 0, tzinfo=_dt.timezone.utc),
    )
    assert observed == {"replay-concordance", "single-agent-quant"}


def test_a_listing_failure_is_unmeasured_not_an_empty_observed_set():
    """An empty `observed` would be reported as universal silence — the
    alarming direction — for a fault in the check itself."""
    from scripts.cost_coverage import CostCoverageUnmeasured

    class _Boom:
        def get_paginator(self, name):
            raise RuntimeError("s3 unreachable")

    with pytest.raises(CostCoverageUnmeasured, match="NOT a coverage finding"):
        _observed(s3=_Boom())


def test_a_listed_object_without_last_modified_is_unmeasured():
    from scripts.cost_coverage import CostCoverageUnmeasured

    s3 = _WindowS3({f"{_ROOT}/2026-08-29/r/replay-concordance.0.jsonl": None})
    with pytest.raises(CostCoverageUnmeasured, match="LastModified"):
        _observed(s3=s3)


def test_execution_started_at_refuses_to_default():
    """No fabricated start time — it would silently move the window."""
    from scripts.cost_coverage import CostCoverageUnmeasured, execution_started_at

    with pytest.raises(CostCoverageUnmeasured, match="execution ARN"):
        execution_started_at("")

    class _Boom:
        def describe_execution(self, **kw):
            raise RuntimeError("denied")

    with pytest.raises(CostCoverageUnmeasured, match="NOT a coverage finding"):
        execution_started_at("arn:x", sfn_client=_Boom())


def test_execution_started_at_normalises_to_utc():
    from scripts.cost_coverage import execution_started_at

    class _Sfn:
        def describe_execution(self, **kw):
            return {"startDate": _dt.datetime(2026, 8, 29, 9, 0, 49)}

    assert execution_started_at("arn:x", sfn_client=_Sfn()) == _STARTED.replace(
        microsecond=0
    )
