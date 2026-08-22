"""Guard: no test file resolves a path outside this repo's root without a
documented, CI-provisioned fallback (alpha-engine-config-I7605 / I7619).
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
TESTS_DIR = Path(__file__).parent

_ALLOWLIST = {
    # "test_whatever.py": ("reason ..."),
}

_SIBLING_CHECKOUT_TELL = re.compile(
    r'Path\.home\(\)\s*/\s*["\']Development["\']'
    r'|os\.environ\[["\'][A-Z_]*_DIR["\']\]'
)

_THIS_FILE = Path(__file__).name


def _test_files():
    return sorted(
        p for p in TESTS_DIR.glob("test_*.py") if p.is_file() and p.name != _THIS_FILE
    )


def test_no_undocumented_sibling_checkout_path_resolution():
    offenders = []
    for path in _test_files():
        if path.name in _ALLOWLIST:
            continue
        text = path.read_text()
        if _SIBLING_CHECKOUT_TELL.search(text):
            offenders.append(path.name)
    assert not offenders, (
        f"test file(s) resolve a path outside this repo's root with no "
        f"documented CI-provisioned fallback: {offenders}. Either fix at the "
        f"contract layer (nousergon_lib.contracts), vendor the fixture "
        f"locally, or add a reviewed entry to _ALLOWLIST naming why not and "
        f"confirming the CI env-var override + hard-fail-on-CI guard are "
        f"both present."
    )


def test_allowlist_entries_still_exist_and_are_still_safe():
    for name, _reason in _ALLOWLIST.items():
        path = TESTS_DIR / name
        assert path.exists(), f"allowlisted {name} no longer exists — remove its entry"
        text = path.read_text()
        assert "_ON_CI" in text, f"{name} is allowlisted but lost its _ON_CI hard-fail guard."
        assert "_DIR" in text, f"{name} is allowlisted but lost its *_DIR env var override."
