"""Lint guard: no hardcoded ``max_tokens=N`` literals in agents/.

Closes the whack-a-mole pattern observed 2026-05-03:
  - PR #100 fixed peer_review._joint_finalization (800-cap truncation)
  - PR #101 (closed/superseded) was about to fix qual_analyst (4096-cap)
  - SF run 4 surfaced qual_analyst's 4096 truncation

Each subsequent fix touched a different hardcoded literal at a
different site. The consolidation (PR pairing config #25 with this
test) routes every site through ``MAX_TOKENS_PER_STOCK`` /
``MAX_TOKENS_STRATEGIC``. This lint enforces that no future drift
re-introduces hardcoded literals — the conversation that scoped the
two-tier taxonomy is the durable artifact, this test is just the
guard.

Allowlist: a small number of intentional small-cap calls in
``macro_agent.py`` (the critic) and elsewhere that produce
narrowly-scoped outputs and intentionally use a tighter budget than
either tier provides. Add to ``_ALLOWLIST`` only with a comment line
above the literal naming WHY the per-site override is appropriate.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_AGENTS_ROOT = _REPO_ROOT / "agents"
_GRAPH_ROOT = _REPO_ROOT / "graph"

# Sites where a hardcoded value is intentional. Every entry must have a
# comment in the source explaining WHY the per-site override is
# appropriate (audited 2026-05-03 during consolidation).
_ALLOWLIST: dict[str, set[int]] = {
    # macro_agent critic — small-output structured call (action + critique
    # + suggested_regime). 512 is intentional; both tier values are
    # oversized for this narrow case. Bumping would just slacken the
    # ceiling; leaving it at 512 keeps the failure-mode visible if the
    # critic ever generates verbose output.
    # Line moved 462 → 476 by the all-agents-strict rework (added the
    # langchain_utils import + an explicit max_retries=
    # SECTOR_TEAM_LLM_MAX_RETRIES kwarg on the critic ChatAnthropic),
    # then 476 → 562 by the drawdown-leg surface (added
    # _format_drawdown_leg + the continuous-statement reframe),
    # then 562 → 572 by the Phase 2.A.2 scorecard-kwarg arc (added
    # `prior_cycle_scorecard` to run_macro_agent signature + an inline
    # comment + kwarg passthrough in the _PROMPT_TEMPLATE.format call),
    # then 572 → 580 by the caution-regime retirement (v0.42.0,
    # 2026-05-28): added _LEGACY_REGIME_COERCION shim + WARN log +
    # docstring rework on _validate_regime, dropping the soft-override
    # → caution branches; net +8 lines above the critic call,
    # then 580 → 565 by retiring the now-dead _LEGACY_REGIME_COERCION
    # shim (2026-05-29, Phase 1B follow-on): the macro prompts dropped
    # the 4-class vocabulary so the coercion was unreachable on the LLM
    # path; net -15 lines above the critic call,
    # then 565 → 567 by the per-call LLM request-timeout guard (config#687,
    # 2026-06-26): added the SECTOR_TEAM_LLM_REQUEST_TIMEOUT_SECONDS import
    # (+1 line near the top) and a default_request_timeout= kwarg on the
    # strategic-tier ChatAnthropic above the critic (+1 line); net +2.
    # Same intentional 512-literal critic call, just relocated.
    # then 567 → 579 by the config#1753 rendered-prompt-capture fix:
    # ``run_macro_agent``'s return dict grew a ``"rendered_prompt"`` key
    # (+ explanatory comment, 12 lines total) above the critic function;
    # net +12. Same intentional 512-literal critic call, just relocated.
    "agents/macro_agent.py": {578},
}


_HARDCODED_PATTERN = re.compile(r"max_tokens\s*=\s*(\d+)")


def _scan_file(path: Path, repo_relative: str) -> list[tuple[int, str]]:
    """Return list of (line_no, line_text) for every hardcoded
    ``max_tokens=N`` literal in ``path`` that isn't allowlisted."""
    allowed_lines = _ALLOWLIST.get(repo_relative, set())
    findings: list[tuple[int, str]] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if line_no in allowed_lines:
            continue
        # Skip comments — references to past values in docstrings are fine.
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        m = _HARDCODED_PATTERN.search(line)
        if m:
            findings.append((line_no, line.rstrip()))
    return findings


def _all_python_files() -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    for root in (_AGENTS_ROOT, _GRAPH_ROOT):
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            out.append((path, str(path.relative_to(_REPO_ROOT))))
    return out


class TestMaxTokensLint:
    def test_no_hardcoded_max_tokens_literals(self):
        """Every call must use ``MAX_TOKENS_PER_STOCK`` (per-stock tier:
        single-ticker outputs) or ``MAX_TOKENS_STRATEGIC`` (synthesis
        tier: multi-item structured outputs).

        Hardcoded numeric literals re-introduce the truncation-bug
        whack-a-mole pattern (peer_review at 800 / qual_analyst at 4096
        / etc., each fixed in a separate PR). Use the named constants
        so a single config bump moves all sites at once.

        Add to ``_ALLOWLIST`` only with explicit justification — see
        the docstring at the top of this file.
        """
        violations: list[str] = []
        for path, repo_relative in _all_python_files():
            for line_no, line_text in _scan_file(path, repo_relative):
                violations.append(f"{repo_relative}:{line_no}: {line_text.strip()}")

        assert not violations, (
            "Hardcoded max_tokens=N literals found. Replace with "
            "MAX_TOKENS_PER_STOCK or MAX_TOKENS_STRATEGIC (from config), "
            "or add to _ALLOWLIST in tests/test_max_tokens_lint.py with "
            "a justifying source-code comment.\n\n"
            + "\n".join(f"  - {v}" for v in violations)
        )

    def test_allowlist_entries_still_exist(self):
        """Pin allowlisted lines so that if they disappear (refactor
        moves them, file deleted, etc.), the lint test fails loudly
        rather than silently accepting drift at the new line."""
        for repo_relative, line_nos in _ALLOWLIST.items():
            path = _REPO_ROOT / repo_relative
            assert path.exists(), f"Allowlist references missing file: {repo_relative}"
            file_lines = path.read_text().splitlines()
            for line_no in line_nos:
                assert 1 <= line_no <= len(file_lines), (
                    f"Allowlist line {repo_relative}:{line_no} out of range "
                    f"(file has {len(file_lines)} lines)"
                )
                line_text = file_lines[line_no - 1]
                assert _HARDCODED_PATTERN.search(line_text), (
                    f"Allowlist line {repo_relative}:{line_no} no longer contains "
                    f"a max_tokens=N literal — content is: {line_text!r}. "
                    f"Remove the allowlist entry or fix the line reference."
                )
