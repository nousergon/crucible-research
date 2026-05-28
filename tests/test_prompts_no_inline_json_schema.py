"""Locks audit finding F1: no production prompt may carry an inline
``Respond with ONLY a JSON object`` schema example.

Per the prompt audit at ``alpha-engine-docs/private/alpha-engine-research-
prompt-audit-260430.md`` § 2 / F1: every LLM call site uses
``with_structured_output(<PydanticModel>)`` or ``response_format=`` since
the typed-state hard-fail flip arc (PRs #62-#65). The inline JSON schema
example in each prompt body became redundant — and a drift surface
(PR #59/#60 caught a literal-vs-int drift between the prompt example and
the Pydantic schema).

PR B (2026-05-02) stripped these examples from all 10 prompts. This test
prevents resurrection: any future PR that re-introduces a literal JSON
schema in a prompt body will fail here, forcing the contributor to
re-affirm + extend the Pydantic schema instead.

Search paths mirror ``agents.prompt_loader._resolve_prompt_path``: sibling
clone, then ``$GITHUB_WORKSPACE`` (CI), then the Lambda-staged
``<repo>/config/prompts/`` directory.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from agents.prompt_loader import load_prompt

# The 10 production prompts shipped with the research repo. Mirrors the
# inventory in the audit doc § 1.
_PRODUCTION_PROMPTS = (
    "macro_agent",
    "macro_agent_critic",
    "quant_analyst_system",
    "quant_analyst_user",
    "qual_analyst_system",
    "qual_analyst_user",
    "peer_review_quant_addition",
    "peer_review_joint_selection",
    "peer_review_per_ticker_rationale",
    "sector_team_thesis_update",
    "ic_cio_evaluation",
)

# Patterns that indicate the prompt is re-stating its output schema in prose
# rather than relying on the SDK's structured-output enforcement. Each is
# case-insensitive. Hits trigger a hard-fail with the exact line for triage.
_INLINE_JSON_PATTERNS = (
    r"respond with only a json",
    r"output json only",
    r"end with a json block",
    r"respond with a json object containing your assessments",
    r"respond with your final ranked list as a json array",
    r"output the full refreshed report followed by the json block",
)


def _config_prompts_dir() -> Path | None:
    """Resolve the production prompts directory if available in this env.

    Returns ``None`` when neither the sibling-clone nor the
    ``$GITHUB_WORKSPACE`` checkout nor the Lambda-staged directory has the
    prompts (e.g. a contributor running pytest without staging the config
    repo). Test then ``pytest.skip``s rather than false-failing.
    """
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        Path.home() / "alpha-engine-config" / "research" / "prompts",
        repo_root.parent / "alpha-engine-config" / "research" / "prompts",
    ]
    ws = os.environ.get("GITHUB_WORKSPACE")
    if ws:
        candidates.append(
            Path(ws) / "alpha-engine-config" / "research" / "prompts"
        )
    candidates.append(repo_root / "config" / "prompts")
    for c in candidates:
        if c.exists() and any(c.glob("*.txt")):
            return c
    return None


@pytest.fixture(scope="module")
def prompts_dir() -> Path:
    p = _config_prompts_dir()
    if p is None:
        pytest.skip(
            "alpha-engine-config prompts not staged (no sibling clone, "
            "GITHUB_WORKSPACE, or Lambda staging). Skipping production-"
            "prompt content lock."
        )
    return p


@pytest.mark.parametrize("name", _PRODUCTION_PROMPTS)
def test_no_inline_json_schema(name: str, prompts_dir: Path) -> None:
    """Each production prompt must NOT contain an inline JSON schema example."""
    text = (prompts_dir / f"{name}.txt").read_text(encoding="utf-8")
    for pattern in _INLINE_JSON_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        assert match is None, (
            f"Prompt '{name}.txt' contains inline JSON-schema instruction "
            f"matching /{pattern}/ at offset {match.start()} — re-introduces "
            f"audit finding F1 (drift surface vs Pydantic schema). Remove "
            f"the prose example; rely on with_structured_output enforcement."
        )


@pytest.mark.parametrize("name", _PRODUCTION_PROMPTS)
def test_no_template_json_block(name: str, prompts_dir: Path) -> None:
    """No prompt should carry an example with multiple ``{{`` brace pairs.

    The previous schema examples used double-brace-escaped JSON for
    ``str.format()``. A handful of legitimate ``{{`` may appear in valid
    rendering contexts (extremely rare; we set a generous threshold). Three
    or more separate ``{{`` openings in one prompt strongly suggests an
    inline JSON example slipped back in.
    """
    text = (prompts_dir / f"{name}.txt").read_text(encoding="utf-8")
    open_braces = text.count("{{")
    assert open_braces < 3, (
        f"Prompt '{name}.txt' has {open_braces} ``{{{{`` escape sequences — "
        f"likely indicates an inline JSON-schema example. Audit finding F1 "
        f"removed these in PR B; re-introducing them resurrects the "
        f"two-sources-of-truth drift risk vs the Pydantic schema."
    )


@pytest.mark.parametrize("name", _PRODUCTION_PROMPTS)
def test_prompt_has_version_frontmatter(name: str, prompts_dir: Path) -> None:
    """Every production prompt must declare its version in frontmatter.

    Loader defaults to ``0.0.0`` when frontmatter is absent — that default
    is for legacy / test prompts only. Production prompts MUST stamp a
    real semver so LangSmith metadata propagation (PR D) and prompt-vs-
    prompt drift detection have a real version to attribute outputs to.
    """
    loaded = load_prompt(name)
    assert loaded.version != "0.0.0", (
        f"Prompt '{name}.txt' is missing ``# version: X.Y.Z`` frontmatter. "
        f"Add the frontmatter line; the loader's default-to-zero only "
        f"applies to legacy/test prompts."
    )


# ── Audit findings F2 + F3 (PR C, 2026-05-02) ────────────────────────────


@pytest.mark.parametrize("name", _PRODUCTION_PROMPTS)
def test_no_revised_date_stamp(name: str, prompts_dir: Path) -> None:
    """Closes audit finding F3: no prompt may carry inline ``(revised YYYY-MM-DD)``
    date stamps. Those are git-blame data leaking into the prompt body —
    operators get the same context from the prompt-version metadata in
    the LangSmith trace + the config repo's git history."""
    text = (prompts_dir / f"{name}.txt").read_text(encoding="utf-8")
    match = re.search(r"\(revised\s+\d{4}-\d{2}-\d{2}", text, re.IGNORECASE)
    assert match is None, (
        f"Prompt '{name}.txt' contains an inline ``(revised YYYY-MM-DD)`` "
        f"stamp at offset {match.start() if match else 'n/a'}. Strip it — "
        f"prompt versioning + LangSmith metadata propagation carry the "
        f"revision context now (audit finding F3)."
    )


def test_macro_agent_uses_canonical_sector_list_placeholder(prompts_dir: Path) -> None:
    """Closes audit finding F2: the canonical sector list must be injected
    into ``macro_agent.txt`` via the ``{sector_list_text}`` template
    placeholder, not enumerated verbatim in prompt prose. Single source
    of truth = ``config.ALL_SECTORS`` rendered at format() time."""
    text = (prompts_dir / "macro_agent.txt").read_text(encoding="utf-8")
    assert "{sector_list_text}" in text, (
        "macro_agent.txt must use ``{sector_list_text}`` placeholder so the "
        "canonical sector list is single-sourced from config.ALL_SECTORS. "
        "Remove any verbatim sector enumeration from the prompt body and "
        "inject via _PROMPT_TEMPLATE.format(sector_list_text=...)."
    )


def test_macro_agent_format_passes_canonical_sectors() -> None:
    """End-to-end: agents.macro_agent must pass ``ALL_SECTORS`` into the
    prompt's ``{sector_list_text}`` placeholder. Lock the wiring so a
    refactor that drops the kwarg surfaces here instead of crashing the
    LLM call at runtime."""
    from config import ALL_SECTORS
    from agents.macro_agent import _PROMPT_TEMPLATE

    rendered = _PROMPT_TEMPLATE.format(
        sector_list_text="\n".join(f"- {s}" for s in ALL_SECTORS),
        prior_date="2026-05-02",
        prior_report="NONE — initial report",
        # Stage C.1 (alpha-engine-config v1.3.0): macro_agent.txt has
        # a {regime_substrate} placeholder for the quant prior block.
        # Pass the canonical "not available this run" fallback string
        # so the format() call succeeds; the agent's
        # _format_regime_substrate(None) emits this same message.
        regime_substrate="QUANTITATIVE REGIME SUBSTRATE (Stage C): not available this run.",
        # Phase-2 prior-cycle scorecard placeholder. The graph wires this in
        # via ``research_graph::prior_cycle_scorecard_text`` (loaded from
        # ``evals/last_week_scorecard.py::load_latest_scorecard_text``);
        # empty string is the canonical missing-artifact fallback.
        prior_cycle_scorecard="",
        fed_funds="4.50",
        t2yr="3.90",
        t10yr="4.30",
        curve_slope="40",
        vix="18.0",
        spy_30d="2.0",
        qqq_30d="3.0",
        iwm_30d="1.0",
        oil="75.00",
        gold="2400",
        copper="4.20",
        cpi_yoy="2.8",
        unemployment="4.0",
        consumer_sentiment="72",
        initial_claims="220",
        hy_oas="320",
        pct_above_50d="60",
        pct_above_200d="65",
        adv_dec_ratio="1.20",
        upcoming_releases="See FRED calendar.",
    )
    for sector in ALL_SECTORS:
        assert f"- {sector}" in rendered, (
            f"Rendered macro_agent prompt missing canonical sector "
            f"{sector!r} — sector_list_text injection regressed."
        )
