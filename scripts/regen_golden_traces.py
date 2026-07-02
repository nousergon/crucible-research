"""Regenerate the golden-trace eval-regression fixtures (L4579).

Run after an INTENTIONAL change to a rubric prompt, the
``RubricEvalLLMOutput`` schema, or the graph trajectory contract, to
re-bless the goldens the CI gate (``tests/test_eval_golden_regression``)
locks against. The whole point of the gate is that this is a *conscious*
step — regenerating forces you to look at what changed.

Two modes:

* default (no flags) — refresh the rubric version + content-hash pins
  from the current ``alpha-engine-config`` prompts and rewrite the
  static golden artifact / parse case / graph contract. **No API key
  needed** (pins are a pure config read; the recorded parse case is a
  schema-valid authored fixture).
* ``--live`` — additionally call the real judge on the sector_quant
  golden and record its actual dimension scores as the parse case, so
  the fixture is a real captured trace. Needs ``ANTHROPIC_API_KEY``.

Usage:
    python scripts/regen_golden_traces.py
    python scripts/regen_golden_traces.py --live
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from evals import trajectory
from evals.golden_trace import (
    EVAL_PIPELINE_PATH,
    GRAPH_TOPOLOGY_PATH,
    GOLDEN_SCHEMA_VERSION,
    current_pin,
)


# Representative agent_id per rubric (must map via
# evals.judge.resolve_rubric_for_agent).
RUBRIC_AGENTS: dict[str, str] = {
    "eval_rubric_sector_quant": "sector_quant:technology",
    "eval_rubric_sector_qual": "sector_qual:technology",
    "eval_rubric_sector_peer_review": "sector_peer_review:technology",
    "eval_rubric_macro_economist": "macro_economist",
    "eval_rubric_ic_cio": "ic_cio",
    "eval_rubric_thesis_update": "thesis_update:technology:AAPL",
    # Think-tank family (config#1579 P2) — coarse agent_ids by design.
    "eval_rubric_thinktank_thesis": "thinktank_thesis",
    "eval_rubric_thinktank_theme": "thinktank_theme",
}

# Shared golden DecisionArtifact. The input carries a recognizable marker
# so the render test can prove ``agent_input`` was interpolated into the
# rubric (a template that drops the placeholder would fail).
_GOLDEN_MARKER = "GOLDEN_TRACE_MARKER_AAPL"
GOLDEN_ARTIFACT = {
    "run_id": "golden-run-001",
    "timestamp": "2026-05-09T22:30:00.000Z",
    "input_data_snapshot": {
        "ticker": "AAPL",
        "marker": _GOLDEN_MARKER,
        "sector": "technology",
        "quant_metrics": {"pe": 28.4, "rev_growth": 0.08},
        "regime": "neutral",
    },
    "agent_output": {
        "ranked_picks": [{"ticker": "AAPL", "score": 72, "rationale": "stub"}],
        "summary": "golden output for the eval-regression gate",
    },
}

# Authored, schema-valid recorded judge response for the deterministic
# parse case (mirrors the sector_quant rubric's 6 dimensions). ``--live``
# replaces it with a real captured response.
_PARSE_DIMENSIONS = [
    ("numerical_grounding", 4),
    ("signal_calibration", 3),
    ("ranking_coherence", 4),
    ("regime_awareness", 3),
    ("reasoning_complexity", 2),
    ("output_completeness", 4),
]


def _authored_parse_case() -> dict:
    dims = [
        {
            "dimension": name,
            "score": score,
            "reasoning": f"golden reasoning for {name}",
        }
        for name, score in _PARSE_DIMENSIONS
    ]
    llm_input = {
        "dimension_scores": dims,
        "overall_reasoning": "Golden parse case; deterministic fixture.",
    }
    return {
        "rubric_id": "eval_rubric_sector_quant",
        "recorded_response": {
            "content": [
                {
                    "type": "tool_use",
                    "name": "RubricEvalLLMOutput",
                    "input": llm_input,
                }
            ]
        },
        "expected": {
            "dimension_scores": [
                {"dimension": d["dimension"], "score": d["score"]} for d in dims
            ],
            "overall_reasoning": llm_input["overall_reasoning"],
        },
    }


def _live_parse_case() -> dict:
    """Capture a real judge response on the sector_quant golden."""
    from alpha_engine_lib.decision_capture import DecisionArtifact
    from evals.judge import evaluate_artifact

    artifact = DecisionArtifact(
        run_id=GOLDEN_ARTIFACT["run_id"],
        timestamp=GOLDEN_ARTIFACT["timestamp"],
        agent_id="sector_quant:technology",
        input_data_snapshot=GOLDEN_ARTIFACT["input_data_snapshot"],
        agent_output=GOLDEN_ARTIFACT["agent_output"],
    )
    result = evaluate_artifact(artifact, judge_model="claude-haiku-4-5")
    dims = [
        {"dimension": d.dimension, "score": d.score, "reasoning": d.reasoning}
        for d in result.dimension_scores
    ]
    return {
        "rubric_id": "eval_rubric_sector_quant",
        "recorded_response": {
            "content": [
                {
                    "type": "tool_use",
                    "name": "RubricEvalLLMOutput",
                    "input": {
                        "dimension_scores": dims,
                        "overall_reasoning": result.overall_reasoning,
                    },
                }
            ]
        },
        "expected": {
            "dimension_scores": [
                {"dimension": d["dimension"], "score": d["score"]} for d in dims
            ],
            "overall_reasoning": result.overall_reasoning,
        },
    }


def build_eval_pipeline(live: bool) -> dict:
    rubrics = []
    for rubric_id, agent_id in sorted(RUBRIC_AGENTS.items()):
        pin = current_pin(rubric_id, agent_id)
        rubrics.append({
            "rubric_id": pin.rubric_id,
            "agent_id": pin.agent_id,
            "version": pin.version,
            "prompt_hash": pin.prompt_hash,
        })
    return {
        "schema": GOLDEN_SCHEMA_VERSION,
        "marker": _GOLDEN_MARKER,
        "golden_artifact": GOLDEN_ARTIFACT,
        "rubrics": rubrics,
        "parse_case": _live_parse_case() if live else _authored_parse_case(),
    }


def build_graph_topology() -> dict:
    return {
        "schema": GOLDEN_SCHEMA_VERSION,
        "required_nodes": list(trajectory.REQUIRED_NODES),
        "ordering_constraints": [list(c) for c in trajectory.ORDERING_CONSTRAINTS],
        "sector_team_count": trajectory.EXPECTED_SECTOR_TEAM_COUNT,
    }


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(f"wrote {path.relative_to(_REPO_ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--live", action="store_true",
        help="capture a real judge response (needs ANTHROPIC_API_KEY)",
    )
    args = ap.parse_args()

    _write(EVAL_PIPELINE_PATH, build_eval_pipeline(live=args.live))
    _write(GRAPH_TOPOLOGY_PATH, build_graph_topology())


if __name__ == "__main__":
    main()
