"""Cross-repo LLM-tier contract (config#3066).

The config#2678 pillar incident: a tier (``PILLAR_TIER = "pillar"``) was
referenced unconditionally in ``thinktank/analyst.py`` while only the legacy
``alpha-engine-config/research/thinktank.yaml`` carried the matching tier
entry — the *deployed* package source (``experiments/reference/research/
thinktank.yaml``) did not, so the 2026-07-18 weekly run KeyError'd at
runtime with no PR-time signal.

This test discovers every tier NAME the code references (statically, so it
never drifts out of sync with the code) and asserts each one exists in the
thinktank.yaml that actually resolves at runtime — the same
``resolve_experiment_config`` package-first/legacy-fallback path
``thinktank/settings.py`` uses. It replaces reliance on the self-disabling
``_check_superset_over_legacy`` bridge in alpha-engine-config's
``check_experiment_package_completeness.py`` (which stops checking the
moment the legacy ``research/`` dir is deleted) with a durable code<->config
invariant that survives config#1042's legacy-dir removal.
"""

from __future__ import annotations

import ast
from pathlib import Path

from thinktank.settings import load_settings

_THINKTANK_DIR = Path(__file__).resolve().parent.parent / "thinktank"


def _discover_referenced_tiers() -> set[str]:
    """Every module-level ``..._TIER = "name"`` / ``TIER = "name"`` string
    constant under ``thinktank/`` — the naming convention every tier
    reference in this package follows (e.g. ``thinktank/analyst.py``'s
    ``PILLAR_TIER``/``SWEEP_TIER``/``THESIS_TIER``, ``thinktank/themes.py``'s
    ``TIER``). Parsed via ``ast`` (no execution/import side effects) so a
    future tier constant is picked up automatically without editing this
    test.
    """
    tiers: set[str] = set()
    for path in _THINKTANK_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and (target.id == "TIER" or target.id.endswith("_TIER")):
                    tiers.add(node.value.value)
    return tiers


def test_every_code_referenced_tier_exists_in_deployed_config():
    referenced = _discover_referenced_tiers()
    # Sanity: fail loudly if the discovery pattern itself breaks (e.g. the
    # naming convention changes) rather than silently passing on zero tiers.
    assert referenced, (
        "static scan of thinktank/*.py found no '*_TIER = \"...\"' constants — "
        "the discovery pattern in this test is broken, not the config"
    )

    settings = load_settings()
    missing = referenced - set(settings.tiers)
    assert not missing, (
        f"thinktank code references LLM tier(s) {sorted(missing)} that do not "
        f"exist in the deployed thinktank.yaml (available: {sorted(settings.tiers)}) "
        "— this is exactly the config#2678 pillar KeyError class: a tier used by "
        "code with no matching deploy. Add the tier to alpha-engine-config's "
        "experiments/<ALPHA_ENGINE_EXPERIMENT_ID>/research/thinktank.yaml "
        "(default experiment: reference)."
    )
