"""Wrapper tests for the Inspect AI golden-trace Task (L4579 · #652).

``inspect_ai`` is a CI-only dependency (requirements-evals.txt), not in
the Lambda image. These tests skip cleanly when it isn't installed so the
default CI suite (``pip install -r requirements.txt``) stays green; when
Inspect IS present they assert the Task is constructible and that the
deterministic contract scorer agrees with the core gate on the live
goldens.
"""

from __future__ import annotations

import asyncio

import pytest

inspect_ai = pytest.importorskip(
    "inspect_ai",
    reason="inspect_ai is a CI-only evals dep (requirements-evals.txt)",
)

from evals import golden_trace_inspect as gti  # noqa: E402
from evals.golden_trace import load_eval_pipeline_golden  # noqa: E402

_GOLDEN = load_eval_pipeline_golden()


def test_task_builds_with_one_sample_per_rubric_plus_parse_case():
    t = gti.golden_trace()
    samples = list(t.dataset)
    # 6 rubric render/drift samples + 1 parse-case sample.
    assert len(samples) == len(_GOLDEN["rubrics"]) + 1
    parse_samples = [
        s for s in samples if (s.metadata or {}).get(gti._META_IS_PARSE_CASE)
    ]
    assert len(parse_samples) == 1


def _run_scorer_for(sample):
    """Drive the solver then scorer for one sample, deterministically."""
    from inspect_ai.scorer import Target
    from inspect_ai.solver import TaskState

    state = TaskState(
        model="mockllm/model",
        sample_id=sample.id or 0,
        epoch=0,
        input=sample.input,
        messages=[],
        metadata=dict(sample.metadata or {}),
    )
    solve = gti.golden_trace_solver()
    score_fn = gti.golden_trace_scorer()

    async def _go():
        st = await solve(state, None)
        return await score_fn(st, Target(sample.target))

    return asyncio.run(_go())


def test_every_sample_scores_correct_against_live_goldens():
    from inspect_ai.scorer import CORRECT

    t = gti.golden_trace()
    for sample in t.dataset:
        result = _run_scorer_for(sample)
        assert result.value == CORRECT, (
            f"sample {sample.metadata} scored {result.value}: "
            f"{result.explanation}"
        )
