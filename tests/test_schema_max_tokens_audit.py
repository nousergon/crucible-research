"""Schema-budget audit: each ``with_structured_output(Schema)`` call
site's live ``max_tokens`` tier must exceed an estimated worst-case
output size for that schema.

Closes the truncation-bug class observed 2026-05-03 (PR #100, #102):
qual_analyst was at 4096 tokens but its QualAnalystOutput
(list of 5 QualAssessment entries × ~800 tokens each + envelope ≈
~5500 tokens worst-case) routinely exceeded the cap. No real LLM
call was needed to detect this — the schema field structure plus a
conservative per-field token estimate gives enough signal for a
static audit. This test runs in CI and would have flagged
qual_analyst's 4096 cap before any SF run.

Approach: explicit audit table mapping each (site, schema, tier) to a
manually-estimated worst-case output token count. The test resolves
the ``tier`` to the LIVE value of the corresponding ``config``
constant (e.g. ``MAX_TOKENS_STRATEGIC``) and asserts
``live_tier >= estimated``.

Derive, don't transcribe. An earlier version of this audit hardcoded
the tier value as a literal (``10752``) alongside the estimate, so the
assertion compared ``literal >= estimate`` — it could only catch a
too-small *estimate*, never a too-small *tier*. If
``config/universe.yaml`` had silently set ``max_tokens_strategic``
below a schema's worst case (the exact 2026-05-03 incident class), the
transcribed literal would have stayed at the old value and the audit
would have passed while production truncated. Importing the live
constant makes a tier regression in the config repo fail THIS test in
CI — see alpha-engine-config#1295.

The estimates are deliberately conservative (assume verbose responses
+ headroom). Under-estimating is the failure mode — over-estimating
just means we set a slightly-loose budget which is cost-neutral
(Anthropic bills emitted tokens, not the cap).
"""

from __future__ import annotations

import pytest


# Each tuple: (site_label, schema_attr_name, tier,
# estimated_worst_case_tokens, justification_comment).
#
# ``tier`` is the NAME of the live ``config`` constant that supplies
# this call site's ``max_tokens`` (resolved at test time via
# ``getattr(config, tier)``) — NOT a transcribed literal. Use one of:
#   - "MAX_TOKENS_STRATEGIC"  (synthesis-class outputs)
#   - "MAX_TOKENS_PER_STOCK"  (single-ticker outputs)
#   - a bare int ONLY for sites that hardcode their cap off-tier and
#     are allowlisted in test_max_tokens_lint.py (e.g. macro critic 512).
#
# The tier name MUST match the constant the call site actually imports
# from ``config`` — keep it in sync with the agent module so the audit
# tracks the live source of truth (config/universe.yaml → config.py).
#
# When a schema gains/loses fields or list-cardinality changes, update
# the estimated value AND the comment explaining the new estimate.
# When a tier value bumps, change ONLY config/universe.yaml — the audit
# reads the new value automatically; no edit here is needed.
_AUDIT_TABLE: list[tuple[str, str, str | int, int, str]] = [
    (
        "peer_review._joint_finalization (Pass 1 selection)",
        "JointSelectionOutput",
        "MAX_TOKENS_STRATEGIC",
        300,  # selected_tickers (list of ~3 short symbols) + team_rationale ~200tok + envelope
        "Pass 1 of two-pass: ticker list + team_rationale only. Per-ticker rationale "
        "moves to Pass 2 (one bounded JointFinalizationDecision call per ticker).",
    ),
    (
        "peer_review._joint_finalization (Pass 2 per-ticker rationale)",
        "JointFinalizationDecision",
        "MAX_TOKENS_STRATEGIC",  # called via same finalization_llm; could drop to PER_STOCK
        200,  # ticker (short str) + rationale ≤50 words ~100tok + envelope
        "Pass 2 of two-pass: single-ticker rationale generation, called once per pick.",
    ),
    (
        "peer_review._quant_reviews_addition",
        "QuantAcceptanceVerdict",
        "MAX_TOKENS_PER_STOCK",
        400,  # accept (bool) + reason (str ~200tok) + envelope
        "Single accept/reject + reason text",
    ),
    (
        "qual_analyst (extraction)",
        "QualAnalystOutput",
        "MAX_TOKENS_STRATEGIC",
        6000,  # 5 QualAssessment × ~1000tok each + additional_candidate ~1000tok + envelope
        "5 assessments × (ticker + qual_score + bull/bear ~200tok each + catalysts list)",
    ),
    (
        "quant_analyst (extraction)",
        "QuantAnalystOutput",
        "MAX_TOKENS_STRATEGIC",
        4000,  # 5 QuantPick × ~600tok each (ticker + rationale + scores + catalysts) + envelope
        "5 picks × (ticker + quant_score + rationale + catalysts list ~600tok)",
    ),
    (
        "macro_agent.run_macro_agent (extraction)",
        "MacroEconomistRawOutput",
        "MAX_TOKENS_STRATEGIC",
        3500,  # macro_report (~500tok) + sector_modifiers dict (12×30) + sector_ratings (12×80) + envelope
        "Macro report + per-sector modifiers + per-sector ratings (12 sectors)",
    ),
    (
        "macro_agent (critic)",
        "MacroCriticOutput",
        512,  # hardcoded off-tier — allowlisted in test_max_tokens_lint.py
        400,  # action (str) + critique (~150tok) + suggested_regime (str) + envelope
        "Small structured response — action + critique + regime suggestion",
    ),
    (
        "ic_cio",
        "CIORawOutput",
        "MAX_TOKENS_STRATEGIC",
        4500,  # decisions list × per-decision rationale + entry_thesis + envelope
        "List of CIORawDecision × per-decision rationale + entry_thesis",
    ),
    (
        "ic_cio (critic)",
        "CIOCriticOutput",
        768,  # off-tier via the named _CRITIC_MAX_TOKENS constant in ic_cio.py
              # (named, so no test_max_tokens_lint allowlist entry is needed)
        500,  # action + critique (~150tok) + 3 short ticker lists + envelope
        "Small structured response — action + critique + flagged/drops/adds "
        "ticker lists (config#927)",
    ),
    (
        "evals.judge.evaluate_artifact",
        "RubricEvalLLMOutput",
        "MAX_TOKENS_STRATEGIC",  # DEFAULT_MAX_TOKENS routes through MAX_TOKENS_STRATEGIC
        3500,  # 6 RubricDimensionScore × ~450tok verbose reasoning + overall_reasoning + envelope (post output_completeness + reasoning_complexity addition)
        "6 dimensions × (dim + score + reasoning ~450tok at verbose end) + overall_reasoning + envelope. 6th dim added 2026-05-04 (output_completeness for sector rubrics; reasoning_complexity for all rubrics) — actual Sonnet retry-exhaustion at 8192 in 2026-05-04 force_sonnet smoke triggered the bump to 10752.",
    ),
    (
        "sector_team._update_thesis_for_held_stock",
        "HeldThesisUpdateLLMOutput",
        "MAX_TOKENS_STRATEGIC",  # reclassified off per-stock 2026-06-27
        1600,  # bull_case (~250) + bear_case (~250) + catalysts list[str]
        # (~8 × ~40 = 320) + conviction/score fields (~50) + tool-call
        # parameter-tag envelope (~150), all at the verbose end ≈ ~1020;
        # rounded up to 1600 for headroom.
        "Single-ticker but narrative-rich: bull_case + bear_case prose + a "
        "catalysts list + scores. The prior 800 (MAX_TOKENS_PER_STOCK) "
        "under-counted this at est=600 and truncated MDT's tool-call "
        "mid-`catalysts` on the 2026-06-27 SF run (string-not-list "
        "all-agents-strict hard-fail) — same class as the 2026-05-03 "
        "qual_analyst truncation. Now on MAX_TOKENS_STRATEGIC, the tier the "
        "other narrative-rich extraction outputs use; estimate raised above "
        "800 so a regression back to the per-stock tier fails this audit.",
    ),
]


def _resolve_tier(tier: str | int) -> int:
    """Resolve a tier column to its LIVE ``max_tokens`` value.

    A string is the NAME of a ``config`` constant (the live source of
    truth, sourced from ``config/universe.yaml`` in the
    alpha-engine-config repo) and is looked up dynamically — so a tier
    bump/drop in the config repo flows straight into this audit with no
    edit here. An int is an off-tier hardcoded cap (allowlisted in
    test_max_tokens_lint.py, e.g. the 512-token macro critic)."""
    if isinstance(tier, int):
        return tier

    import config

    assert hasattr(config, tier), (
        f"Audit table references config.{tier}, which does not exist. "
        f"If the tier constant was renamed, update the audit table (and "
        f"the agent call sites that import it)."
    )
    value = getattr(config, tier)
    assert isinstance(value, int), (
        f"config.{tier} is {value!r} (not an int) — tier constants must "
        f"resolve to an integer max_tokens value."
    )
    return value


def _resolve_schema(name: str):
    """Walk the known schema-defining modules to find the Pydantic
    model. Encapsulated so a future schema relocation only needs the
    lookup updated here, not the audit table itself."""
    from graph import state_schemas

    if hasattr(state_schemas, name):
        return getattr(state_schemas, name)

    # Fallbacks — schemas that live in module-local files (not the shared
    # nousergon_lib.agent_schemas re-export surface).
    from agents.investment_committee import ic_cio

    if hasattr(ic_cio, name):
        return getattr(ic_cio, name)

    raise AssertionError(
        f"Schema {name!r} not found in graph.state_schemas. If it lives "
        f"elsewhere, add a fallback to _resolve_schema()."
    )


@pytest.mark.parametrize(
    "site,schema_name,tier,estimated,justification",
    _AUDIT_TABLE,
    ids=lambda t: t if isinstance(t, str) else None,
)
def test_max_tokens_covers_schema_estimate(
    site, schema_name, tier, estimated, justification,
):
    """Each call site's LIVE ``max_tokens`` tier must exceed the
    estimated worst-case output for its schema (with safety margin).

    The tier is resolved from the live ``config`` constant — so this
    fails if EITHER:
      - The schema has changed shape and the estimate needs updating
        (look at the schema's model_fields, recompute, update the
        ``estimated`` column + comment), OR
      - The live tier in config/universe.yaml was set below a schema's
        worst case (fix the config — bump the tier in
        alpha-engine-config so config.py picks it up).

    Don't paper over by lowering the estimate without a clear
    justification — the estimates were calibrated against the
    2026-05-03 truncation incidents.
    """
    schema = _resolve_schema(schema_name)
    # Spot-check that the schema is a real Pydantic model (catches
    # rename / refactor that breaks the audit lookup silently).
    assert hasattr(schema, "model_fields"), (
        f"{schema_name} resolved but isn't a Pydantic model — "
        f"audit table entry is stale."
    )

    live_max_tokens = _resolve_tier(tier)
    tier_desc = tier if isinstance(tier, int) else f"config.{tier}={live_max_tokens}"

    assert live_max_tokens >= estimated, (
        f"\n{site}: live max_tokens={live_max_tokens} ({tier_desc}) < "
        f"estimated worst-case output {estimated} tokens for {schema_name}.\n"
        f"  Justification of estimate: {justification}\n"
        f"  Either bump the tier in alpha-engine-config/config/universe.yaml "
        f"(config.py reads it live) or recompute the estimate after a "
        f"schema change."
    )


def test_audit_table_covers_all_with_structured_output_sites():
    """Pin coverage: as new ``with_structured_output(Schema)`` call
    sites are added to the codebase, they must show up in the audit
    table. A grep over the repo finds the call sites; the test
    asserts every distinct schema is referenced in the audit.

    Catches the failure mode where a new agent ships, hits a tier
    that's too small for its schema, and we discover via SF failure
    instead of CI."""
    import re
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    pattern = re.compile(r"with_structured_output\(\s*([A-Z][A-Za-z0-9_]+)")

    schemas_in_use: set[str] = set()
    for path in (repo_root / "agents").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        for match in pattern.finditer(path.read_text()):
            schemas_in_use.add(match.group(1))
    for path in (repo_root / "evals").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        for match in pattern.finditer(path.read_text()):
            schemas_in_use.add(match.group(1))
    # graph/llm_cost_tracker.py contains a CIORawOutput reference inside
    # a docstring/test stub — already covered by the ic_cio site.

    audited_schemas = {row[1] for row in _AUDIT_TABLE}
    missing = schemas_in_use - audited_schemas
    assert not missing, (
        f"with_structured_output sites use schemas not in the audit "
        f"table: {sorted(missing)}. Add a row to _AUDIT_TABLE in "
        f"tests/test_schema_max_tokens_audit.py with the tier constant "
        f"name + an estimated worst-case output size."
    )
