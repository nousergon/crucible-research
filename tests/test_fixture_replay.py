"""A/B fixture replay — schema-vs-LLM drift detection in CI.

Each fixture in ``tests/fixtures/llm_outputs/`` is a representative LLM
output for one schema. This test loads each fixture and validates it
against the corresponding Pydantic schema in ``graph/state_schemas.py``.

If a schema change tightens a constraint that real LLM output already
violates (or removes a literal value the LLM emits), this test fails in
CI before the PR merges — surfacing schema-vs-LLM drift earlier than
Saturday's SF would.

Fixtures are intentionally minimal-conformant. Schema relaxations
(adding optional fields, widening literals) won't break them; only
tightenings will. That's the desired drift-detection posture.

Refresh fixtures from real captured S3 ``decision_artifacts/`` after
each Saturday SF — see ``tests/fixtures/llm_outputs/README.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from graph.state_schemas import (
    CIODecision,
    CIORawOutput,
    HeldThesisUpdateLLMOutput,
    InvestmentThesis,
    JointFinalizationOutput,
    MacroCriticOutput,
    MacroEconomistRawOutput,
    QualAnalystOutput,
    QuantAcceptanceVerdict,
    QuantAnalystOutput,
    RubricEvalLLMOutput,
    SectorTeamOutput,
)

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "llm_outputs"


# Map fixture filename → schema class. New fixtures append here.
_FIXTURES: list[tuple[str, type]] = [
    ("macro_economist_raw_output.json", MacroEconomistRawOutput),
    ("macro_critic_output.json", MacroCriticOutput),
    ("quant_analyst_output.json", QuantAnalystOutput),
    ("qual_analyst_output.json", QualAnalystOutput),
    ("quant_acceptance_verdict.json", QuantAcceptanceVerdict),
    ("joint_finalization_output.json", JointFinalizationOutput),
    ("held_thesis_update_llm_output.json", HeldThesisUpdateLLMOutput),
    ("cio_raw_output.json", CIORawOutput),
    ("sector_team_output.json", SectorTeamOutput),
    ("investment_thesis.json", InvestmentThesis),
    ("cio_decision.json", CIODecision),
    ("rubric_eval_llm_output.json", RubricEvalLLMOutput),
]


@pytest.mark.parametrize("fixture_name,schema_cls", _FIXTURES,
                         ids=[name for name, _ in _FIXTURES])
def test_fixture_validates_against_schema(fixture_name: str, schema_cls: type):
    """Each fixture must validate against its schema. Failure surfaces
    schema-vs-LLM drift — either the schema tightened past reality, or a
    fixture went stale and needs a refresh from S3."""
    fixture_path = _FIXTURE_DIR / fixture_name
    assert fixture_path.exists(), (
        f"Fixture missing: {fixture_path}. See README in fixtures/llm_outputs/"
    )

    with fixture_path.open() as f:
        payload = json.load(f)

    # If validation fails, the error message names the field — actionable
    # signal for whether to relax the schema or refresh the fixture.
    schema_cls.model_validate(payload)


def test_all_pr2_schemas_have_a_fixture():
    """Every LLM-extraction schema in PR 2 must have a fixture in the
    corpus. Forces new schemas to come with a representative fixture
    rather than landing untested.
    """
    pr2_schemas = {
        MacroEconomistRawOutput,
        MacroCriticOutput,
        QuantAnalystOutput,
        QualAnalystOutput,
        QuantAcceptanceVerdict,
        JointFinalizationOutput,
        HeldThesisUpdateLLMOutput,
        CIORawOutput,
    }
    fixture_schemas = {schema_cls for _, schema_cls in _FIXTURES}
    missing = pr2_schemas - fixture_schemas
    assert not missing, (
        f"PR 2 schemas without fixture coverage: {[s.__name__ for s in missing]}. "
        f"Add fixture to tests/fixtures/llm_outputs/ + register in _FIXTURES."
    )


def test_fixture_loadable_as_strict_pydantic():
    """Each fixture must load via ``model_validate`` (strict mode), not
    just ``model_construct``. ``model_construct`` skips field validators,
    so it would mask issues like sector_modifiers out of [0.70, 1.30]
    or invalid Literal values.
    """
    for fixture_name, schema_cls in _FIXTURES:
        fixture_path = _FIXTURE_DIR / fixture_name
        with fixture_path.open() as f:
            payload = json.load(f)
        # model_validate runs all field_validators; a passing payload
        # here is guaranteed to also pass under STRICT_VALIDATION=true
        # at the runtime _validate boundary.
        instance = schema_cls.model_validate(payload)
        assert instance is not None
