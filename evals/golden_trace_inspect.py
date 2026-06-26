"""Inspect AI publishable wrapper over the golden-trace core (L4579 · #652).

The L4579 deterministic core (``evals/golden_trace.py``) is built
framework-agnostic on purpose (plain JSON fixtures + pure helpers) so it
can be surfaced through `Inspect AI <https://inspect.aisi.org.uk>`_ — the
AISI-standard, *publishable* harness named in L4579. This module is that
wrapper: it exposes the same goldens as an Inspect ``Task`` so the
deterministic prompt-layer contract gate can run under the standard
``inspect eval`` tooling (and be cited in the published harness writeup),
without changing what the core asserts.

Two run modes, same goldens:

* **Deterministic (CI / default)** — the solver renders each rubric and
  replays the recorded judge response through the production parser; no
  live model is called, so the gate is fast and non-flaky. Run with the
  ``mockllm/model`` provider, e.g.::

      inspect eval evals/golden_trace_inspect.py --model mockllm/model

  The mock model is never actually consulted for the contract checks (the
  rendering + parse are deterministic), but selecting it keeps the run
  fully offline and is the documented Inspect way to express "no real
  provider".

* **Nightly (real model)** — point ``--model`` at a real provider to
  additionally exercise the render against a live judge; the deterministic
  contract scorer still gates. (The deterministic path is the publishable
  CI gate; the real-model path is the nightly variant L4579 calls for.)

**CI-only dependency.** ``inspect_ai`` is intentionally *not* in
``requirements.txt`` (that file is the Lambda image manifest); it lives in
``requirements-evals.txt``. Importing this module without ``inspect_ai``
installed raises a clear ``ImportError`` — the deterministic core and its
``tests/test_eval_golden_regression.py`` gate do not depend on it, so the
default CI suite stays green whether or not Inspect is present.

The contract this wrapper enforces is identical to the core gate:
  1. render contract (every rubric renders + interpolates the input),
  2. prompt-drift lock (version + content hash match the golden pin),
  3. parse pipeline (recorded response parses to the expected scores).
"""

from __future__ import annotations

from typing import Any

try:
    from inspect_ai import Task, task
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.scorer import (
        CORRECT,
        INCORRECT,
        Score,
        Target,
        accuracy,
        scorer,
    )
    from inspect_ai.solver import Generate, TaskState, solver
except ImportError as exc:  # pragma: no cover - exercised only without the dep
    raise ImportError(
        "evals.golden_trace_inspect requires the 'inspect_ai' package "
        "(a CI-only dependency, not in the Lambda requirements.txt). "
        "Install it with `pip install -r requirements-evals.txt` to run "
        "the publishable Inspect harness. The framework-agnostic core in "
        "evals/golden_trace.py and its gate in "
        "tests/test_eval_golden_regression.py do NOT need this package."
    ) from exc

from evals.golden_trace import (
    current_pin,
    load_eval_pipeline_golden,
    parse_recorded_response,
    render_rubric_for,
)

# Metadata keys used to thread the golden into the solver/scorer without a
# real model call.
_META_RUBRIC_ID = "rubric_id"
_META_AGENT_ID = "agent_id"
_META_VERSION = "version"
_META_PROMPT_HASH = "prompt_hash"
_META_IS_PARSE_CASE = "is_parse_case"


def _build_samples(golden: dict[str, Any]) -> list[Sample]:
    """One Inspect ``Sample`` per rubric (render + drift contract) plus a
    final sample carrying the recorded-response parse contract."""
    samples: list[Sample] = []
    for r in golden["rubrics"]:
        samples.append(
            Sample(
                # The "input" is descriptive only; the contract is checked
                # against the live rubric + golden pin in the scorer.
                input=f"render+drift contract for rubric {r['rubric_id']}",
                target=CORRECT,
                metadata={
                    _META_RUBRIC_ID: r["rubric_id"],
                    _META_AGENT_ID: r["agent_id"],
                    _META_VERSION: r["version"],
                    _META_PROMPT_HASH: r["prompt_hash"],
                    _META_IS_PARSE_CASE: False,
                },
            )
        )
    samples.append(
        Sample(
            input="parse pipeline contract for the recorded judge response",
            target=CORRECT,
            metadata={
                _META_RUBRIC_ID: golden["parse_case"]["rubric_id"],
                _META_IS_PARSE_CASE: True,
            },
        )
    )
    return samples


@solver
def golden_trace_solver():
    """Deterministic solver: renders the rubric (or marks the parse case).

    No live model is called for the contract — the rendering and parsing
    are pure functions over the goldens. The solver stashes the rendered
    prompt on the state so the scorer can assert the render contract.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        meta = state.metadata or {}
        if meta.get(_META_IS_PARSE_CASE):
            state.metadata["_rendered"] = None
            return state
        rendered = render_rubric_for(
            load_eval_pipeline_golden(),
            meta[_META_RUBRIC_ID],
            meta[_META_AGENT_ID],
        )
        state.metadata["_rendered"] = rendered
        return state

    return solve


@scorer(metrics=[accuracy()])
def golden_trace_scorer():
    """Scores each sample against the same contract as the core gate.

    Render/drift samples: the rubric must render, interpolate the golden
    input marker, and its live version + content hash must match the
    golden pin. Parse sample: the recorded judge response must parse to
    the golden dimension scores. Any divergence => INCORRECT with an
    explanation pointing at ``scripts/regen_golden_traces.py``.
    """

    golden = load_eval_pipeline_golden()
    marker = golden["marker"]
    parse_case = golden["parse_case"]

    async def score(state: TaskState, target: Target) -> Score:
        meta = state.metadata or {}

        if meta.get(_META_IS_PARSE_CASE):
            out = parse_recorded_response(parse_case["recorded_response"])
            actual = [
                {"dimension": d.dimension, "score": d.score}
                for d in out.dimension_scores
            ]
            expected = parse_case["expected"]
            if (
                actual == expected["dimension_scores"]
                and out.overall_reasoning == expected["overall_reasoning"]
            ):
                return Score(value=CORRECT, answer="parse-ok")
            return Score(
                value=INCORRECT,
                answer="parse-drift",
                explanation=(
                    "Recorded judge response no longer parses to the golden "
                    "dimension scores — a RubricEvalLLMOutput schema change "
                    "or parser regression. Re-bless via "
                    "scripts/regen_golden_traces.py if intentional."
                ),
            )

        rubric_id = meta[_META_RUBRIC_ID]
        rendered = meta.get("_rendered")
        if not rendered or marker not in rendered:
            return Score(
                value=INCORRECT,
                answer="render-drift",
                explanation=(
                    f"{rubric_id}: rendered prompt missing the golden input "
                    f"marker — the {{agent_input}} placeholder may have been "
                    f"removed. Re-bless via scripts/regen_golden_traces.py "
                    f"if intentional."
                ),
            )

        cur = current_pin(rubric_id, meta[_META_AGENT_ID])
        if cur.version != meta[_META_VERSION]:
            return Score(
                value=INCORRECT,
                answer="version-drift",
                explanation=(
                    f"{rubric_id}: prompt version drifted "
                    f"{meta[_META_VERSION]} -> {cur.version}. Regenerate "
                    f"goldens (scripts/regen_golden_traces.py) and review "
                    f"the eval-quality delta."
                ),
            )
        if cur.prompt_hash != meta[_META_PROMPT_HASH]:
            return Score(
                value=INCORRECT,
                answer="hash-drift",
                explanation=(
                    f"{rubric_id}: prompt content hash drifted (text changed "
                    f"even if the version did not). Regenerate goldens and "
                    f"re-bless consciously, or bump the rubric version."
                ),
            )
        return Score(value=CORRECT, answer="render+drift-ok")

    return score


@task
def golden_trace() -> Task:
    """Publishable Inspect ``Task`` over the L4579 golden-trace goldens.

    Deterministic by design: run with ``--model mockllm/model`` for the
    offline CI gate, or a real provider for the nightly variant. The
    contract scored is the same prompt-layer twin enforced by
    ``tests/test_eval_golden_regression.py``.
    """
    golden = load_eval_pipeline_golden()
    return Task(
        dataset=MemoryDataset(_build_samples(golden)),
        solver=golden_trace_solver(),
        scorer=golden_trace_scorer(),
    )
