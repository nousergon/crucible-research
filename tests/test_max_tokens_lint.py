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

Allowlist: a small number of intentional small-cap calls that produce
narrowly-scoped outputs and deliberately use a tighter budget than
either tier provides. These are marked AT THE SITE with a trailing
``# max-tokens-allowlist`` pragma plus a comment naming WHY.

The allowlist used to be keyed on LINE NUMBERS. That key rotted on
every single edit above the site: the pin was re-pointed nine times
(462 -> 476 -> 562 -> 572 -> 579 -> 585 -> 582 -> 580 -> 565 -> 567),
each time carrying another paragraph of archaeology, and it broke
again under ruff-format on a change that never touched the call.
A line number is not a stable identifier for a line of code.

The pragma is stable under reformatting and moves WITH the code. To
keep the property that actually mattered -- a new suppression must be
a deliberate, reviewable act rather than a comment someone quietly
adds -- ``_EXPECTED_PRAGMA_SITES`` pins the per-file COUNT of
pragmas. Adding a site still requires editing this file; it just no
longer requires editing it every time an unrelated line moves.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_AGENTS_ROOT = _REPO_ROOT / "agents"
_GRAPH_ROOT = _REPO_ROOT / "graph"

# Files permitted to carry ``# max-tokens-allowlist`` pragmas, and how
# many. The COUNT is the gate: a new intentional override requires
# bumping it here, so the suppression stays reviewable. Line numbers are
# deliberately NOT recorded -- see the module docstring.
_EXPECTED_PRAGMA_SITES: dict[str, int] = {
    # macro_agent critic — small-output structured call.
    "agents/macro_agent.py": 1,
}

# Marks a deliberate per-site override. Must sit on the same line as the
# literal; a justifying comment belongs immediately above it.
_PRAGMA_PATTERN = re.compile(r"#\s*max-tokens-allowlist\b")

_HARDCODED_PATTERN = re.compile(r"max_tokens\s*=\s*(\d+)")


def _scan_file(path: Path, repo_relative: str) -> list[tuple[int, str]]:
    """Return (line_no, line_text) for every hardcoded ``max_tokens=N``
    literal in ``path`` not carrying an allowlist pragma."""
    findings: list[tuple[int, str]] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        # Skip comments — references to past values in docstrings are fine.
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        if not _HARDCODED_PATTERN.search(line):
            continue
        if _PRAGMA_PATTERN.search(line):
            continue
        findings.append((line_no, line.rstrip()))
    return findings


def _pragma_sites(path: Path) -> list[int]:
    """Line numbers carrying an allowlist pragma ON a max_tokens literal."""
    return [
        line_no
        for line_no, line in enumerate(path.read_text().splitlines(), start=1)
        if _PRAGMA_PATTERN.search(line) and not line.lstrip().startswith("#")
    ]


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
            "a justifying source-code comment.\n\n" + "\n".join(f"  - {v}" for v in violations)
        )

    def test_allowlist_pragma_inventory_matches(self):
        """Pin the NUMBER of deliberate overrides per file.

        Replaces the old line-number pin. Catches both directions:
        a pragma added without review (count up) and a pragma that
        silently disappeared in a refactor (count down), without
        breaking every time an unrelated line moves.
        """
        actual: dict[str, int] = {}
        for path, repo_relative in _all_python_files():
            n = len(_pragma_sites(path))
            if n:
                actual[repo_relative] = n

        assert actual == _EXPECTED_PRAGMA_SITES, (
            "max_tokens allowlist-pragma inventory changed.\n"
            f"  expected: {_EXPECTED_PRAGMA_SITES}\n"
            f"  actual:   {actual}\n"
            "If you added a deliberate per-site override, bump the count in "
            "_EXPECTED_PRAGMA_SITES and make sure the site carries a comment "
            "naming WHY the tier constants are wrong for it."
        )

    def test_every_pragma_actually_suppresses_something(self):
        """An orphaned pragma (one not sitting on a max_tokens literal)
        is dead weight that reads as a live suppression. Reject it."""
        orphans: list[str] = []
        for path, repo_relative in _all_python_files():
            for line_no in _pragma_sites(path):
                line = path.read_text().splitlines()[line_no - 1]
                if not _HARDCODED_PATTERN.search(line):
                    orphans.append(f"{repo_relative}:{line_no}: {line.strip()}")

        assert not orphans, (
            "max-tokens-allowlist pragma not attached to a max_tokens=N "
            "literal (it suppresses nothing and misleads readers):\n" + "\n".join(f"  - {o}" for o in orphans)
        )
