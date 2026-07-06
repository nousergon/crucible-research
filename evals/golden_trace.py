"""Golden-trace eval-regression core (L4579).

The asymmetry this closes: production *code* is contract-tested
everywhere (e.g. ``test_schema_contract`` locks the feature store), but
the LLM-judge **prompts + graph** have zero CI gate — a rubric edit, a
``RubricEvalLLMOutput`` schema change, or a graph rewire can silently
regress eval quality and only surface weeks later in production. This
module is the prompt-layer twin of those contract tests.

**Deterministic by design — no live LLM calls in the gate.** The golden
fixtures are *recorded* judge responses + their expected parses, so the
CI gate is fast and non-flaky. What it locks:

1. **Render contract** — every rubric still loads and renders against a
   golden ``DecisionArtifact`` (catches a template-variable rename or a
   placeholder the renderer no longer supplies).
2. **Parse → score pipeline** — a recorded judge tool-use response still
   parses to the expected dimension scores via ``parse_batch_message``
   (catches a ``RubricEvalLLMOutput`` schema regression or a parser
   change that drops/transforms scores).
3. **Prompt drift** — each rubric's current ``version`` + content
   ``hash`` (from ``LoadedPrompt``) match the golden. ANY prompt edit
   changes the hash → the gate fails → the author must regenerate and
   consciously re-bless the golden (``scripts/regen_golden_traces.py``).
   This is the chokepoint that makes "a prompt change runs the eval
   suite" real.
4. **Graph trajectory contract** — the ``evals/trajectory.py`` reference
   (node set + ordering + sector-team count) matches the golden
   (catches an unreviewed edit that would weaken the runtime validator).

The core is intentionally framework-agnostic (plain JSON fixtures +
pure helpers) so an Inspect AI ``Task`` — the publishable, AISI-standard
wrapper named in L4579 — can consume the same goldens via the ``mockllm``
provider for the deterministic path and a real model for nightly runs.
That wrapper is the next slice; this PR ships the deterministic gate.

Regenerate fixtures with ``python scripts/regen_golden_traces.py`` after
an intentional prompt/schema/graph change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nousergon_lib.decision_capture import DecisionArtifact

from agents.prompt_loader import load_prompt
from evals.judge import _render_rubric, parse_batch_message

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
EVAL_PIPELINE_PATH = GOLDEN_DIR / "eval_pipeline.json"
GRAPH_TOPOLOGY_PATH = GOLDEN_DIR / "graph_topology.json"

GOLDEN_SCHEMA_VERSION = 1


# ── Fixture loading ────────────────────────────────────────────────────────


def load_eval_pipeline_golden() -> dict[str, Any]:
    return json.loads(EVAL_PIPELINE_PATH.read_text(encoding="utf-8"))


def load_graph_topology_golden() -> dict[str, Any]:
    return json.loads(GRAPH_TOPOLOGY_PATH.read_text(encoding="utf-8"))


# ── Rubric pins ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RubricPin:
    rubric_id: str
    agent_id: str
    version: str
    prompt_hash: str


def current_pin(rubric_id: str, agent_id: str) -> RubricPin:
    """Load the rubric prompt as it exists NOW and snapshot its
    version + content hash. The gate compares this to the golden; the
    regen script writes it."""
    lp = load_prompt(rubric_id)
    return RubricPin(
        rubric_id=rubric_id,
        agent_id=agent_id,
        version=lp.version,
        prompt_hash=lp.hash,
    )


# ── Pipeline runners (deterministic) ───────────────────────────────────────


def golden_decision_artifact(golden: dict[str, Any], agent_id: str) -> DecisionArtifact:
    """Build the shared golden DecisionArtifact for ``agent_id``."""
    ga = golden["golden_artifact"]
    return DecisionArtifact(
        run_id=ga["run_id"],
        timestamp=ga["timestamp"],
        agent_id=agent_id,
        input_data_snapshot=ga["input_data_snapshot"],
        agent_output=ga["agent_output"],
    )


def render_rubric_for(golden: dict[str, Any], rubric_id: str, agent_id: str) -> str:
    """Render ``rubric_id`` against the golden artifact (no LLM call).
    Raises if the template references a placeholder the renderer doesn't
    supply — exactly the regression we want to catch."""
    artifact = golden_decision_artifact(golden, agent_id)
    return _render_rubric(artifact, load_prompt(rubric_id))


def parse_recorded_response(recorded_response: dict[str, Any]):
    """Run a recorded judge tool-use response through the production
    parser. Returns the ``RubricEvalLLMOutput`` (raises on schema/parse
    regression)."""
    return parse_batch_message(recorded_response)
