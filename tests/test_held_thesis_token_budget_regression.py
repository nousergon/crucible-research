"""Regression guard for the 2026-06-27 held-thesis truncation hard-fail.

Incident: the Saturday SF run failed in Branch A (research) when the
held-stock thesis update for MDT could not produce valid structured
output after the bounded all-agents-strict parse re-roll:

    1 validation error for HeldThesisUpdateLLMOutput
    catalysts: Input should be a valid list
      [type=list_type, input_value='\\n<parameter name="catal...
       consensus estimates"]\\n', input_type=str]

Root cause: ``_update_thesis_for_held_stock`` was configured with
``max_tokens=MAX_TOKENS_PER_STOCK`` (800). HeldThesisUpdateLLMOutput is
single-ticker but narrative-rich (bull_case + bear_case prose + a
``catalysts`` list[str] + scores), and 800 truncated the Anthropic
tool-call mid-``<parameter name="catalysts">``. langchain then captured
the partial parameter block as a raw string where the schema requires a
list — so pydantic rejected it. Because 800 truncates DETERMINISTICALLY,
all three parse re-rolls failed identically and the run hard-failed
(no prior-thesis carry-forward, per all-agents-strict). This is the same
truncation-bug class as the 2026-05-03 qual_analyst incident
(PR #100/#102), recurring at a different under-budgeted call site.

Fix: reclassify the call site onto ``MAX_TOKENS_STRATEGIC`` — the tier
every other narrative-rich structured-output site already uses
(qual/quant analyst extraction, macro, ic_cio, evals.judge). The model
is unchanged (still PER_STOCK_MODEL / Haiku); only the output ceiling
moves, so nothing about WHAT the thesis concludes changes.

This guard pins the call site to the strategic tier at the source level
so a silent regression back to the per-stock budget is caught in CI
(the static ``test_schema_max_tokens_audit`` row is the complementary
estimate-vs-budget guard).
"""

from __future__ import annotations

import ast
from pathlib import Path

_SECTOR_TEAM = (
    Path(__file__).resolve().parent.parent
    / "agents"
    / "sector_teams"
    / "sector_team.py"
)


def _held_thesis_function() -> ast.FunctionDef:
    tree = ast.parse(_SECTOR_TEAM.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "_update_thesis_for_held_stock"
        ):
            return node
    raise AssertionError(
        "_update_thesis_for_held_stock not found in sector_team.py — this "
        "regression guard's anchor has moved; update it."
    )


def _max_tokens_constant_names() -> list[str]:
    """Return the identifier names passed as ``max_tokens=<NAME>`` inside
    the held-thesis update function (one per ChatAnthropic call)."""
    fn = _held_thesis_function()
    names: list[str] = []
    for node in ast.walk(fn):
        if isinstance(node, ast.keyword) and node.arg == "max_tokens":
            assert isinstance(node.value, ast.Name), (
                "held-thesis max_tokens must be a named tier constant "
                "(MAX_TOKENS_STRATEGIC), not a literal or expression — see "
                "tests/test_max_tokens_lint.py for the no-hardcoded rule."
            )
            names.append(node.value.id)
    return names


def test_held_thesis_uses_strategic_tier():
    """The held-thesis update must budget on the STRATEGIC tier.

    Regressing to MAX_TOKENS_PER_STOCK (800) re-introduces the
    2026-06-27 deterministic ``catalysts`` truncation hard-fail.
    """
    names = _max_tokens_constant_names()
    assert names, (
        "no max_tokens=<CONST> kwarg found in _update_thesis_for_held_stock "
        "— the ChatAnthropic call lost its explicit token budget."
    )
    assert "MAX_TOKENS_PER_STOCK" not in names, (
        "held-thesis update regressed to MAX_TOKENS_PER_STOCK (800). That "
        "budget deterministically truncates the bull_case + bear_case + "
        "catalysts output and hard-fails the run (2026-06-27 MDT incident). "
        "Keep it on MAX_TOKENS_STRATEGIC."
    )
    assert "MAX_TOKENS_STRATEGIC" in names, (
        f"held-thesis update must budget on MAX_TOKENS_STRATEGIC; found "
        f"{names}."
    )
