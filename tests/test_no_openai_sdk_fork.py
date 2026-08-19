"""The fork must not grow back.

`thinktank/client.py` was a fork of `krepis.llm` — a private OpenAI SDK client
with its own retry classifier. It drifted: two OpenRouter failure classes were
hand-patched into it (`_NullChoicesError`, then `json.JSONDecodeError`) that the
shared chokepoint already handled, because a fix landing in krepis has no way to
reach a copy. alpha-engine-config#5223 / crucible-research#530 deleted it.

Deleting a fork is easy. Keeping it deleted is the part that needs a test: the
next agent that needs "just one OpenAI call" here will construct one, and
nothing else in CI would notice until the next incident.

Two invariants, both load-bearing and both previously expressed only in prose:

1. No module on the LLM path constructs the OpenAI SDK client directly — every
   call goes through `krepis.llm.LLMClient`, which is where retry
   classification, the null-choices guard, backoff, and cost attribution live.
2. The krepis floor stays at or above the release where that library became a
   SUPERSET of the fork it replaced. Below it, the migration is a regression —
   measured 2026-07-30 on 0.24.x, a null-choices body raised
   `TypeError: 'NoneType' object is not subscriptable` through the migrated
   client. requirements.txt says the floor is load-bearing; this asserts it.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Packages whose modules talk to LLM providers. A new one belongs here.
_LLM_PATH_PACKAGES = ("thinktank", "evals", "producers", "scoring", "graph", "lambda")

# LangChain's ChatOpenAI is a DIFFERENT surface (agents/langchain_utils.py) with
# its own chokepoint; this guard is about the raw SDK client the fork was built
# on. Named rather than pattern-excluded so adding to this set is a decision.
_ALLOWED_SYMBOLS = frozenset({"ChatOpenAI", "AsyncOpenAI_NOT_IN_USE"})

# krepis#93 added the null-choices guard at all five `choices[0]` reads plus
# bounded backoff between body-level retries. See requirements.txt.
_KREPIS_SUPERSET_FLOOR = (0, 25, 0)


def _python_files() -> list[Path]:
    files: list[Path] = []
    for package in _LLM_PATH_PACKAGES:
        root = _REPO_ROOT / package
        if not root.is_dir():
            continue
        files.extend(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    return files


def _openai_client_constructions(path: Path) -> list[str]:
    """Direct `openai.OpenAI(...)` construction or `from openai import OpenAI`.

    Parsed, not grepped: a docstring mentioning ``OpenAI(...)`` is documentation,
    and `thinktank/client.py` legitimately carries two such references
    explaining what it stopped doing.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "openai":
            for alias in node.names:
                if alias.name.endswith("OpenAI") and alias.name not in _ALLOWED_SYMBOLS:
                    offenders.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno} from openai import {alias.name}")
        elif isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "attr", None) or getattr(func, "id", None)
            if name in ("OpenAI", "AsyncOpenAI") and name not in _ALLOWED_SYMBOLS:
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno} {name}(...)")
    return offenders


def test_no_module_on_the_llm_path_constructs_the_openai_sdk_directly():
    offenders = [o for path in _python_files() for o in _openai_client_constructions(path)]
    assert not offenders, (
        "These construct the OpenAI SDK client directly, re-forking "
        "krepis.llm:\n  " + "\n  ".join(offenders) + "\n\n"
        "Every provider call goes through krepis.llm.LLMClient — that is where "
        "retry classification, the null-choices guard, backoff and cost "
        "attribution live. A local client gets none of them, and a fix landing "
        "in krepis has no way to reach a copy: that is exactly how the "
        "thinktank fork accumulated two hand-patched failure classes krepis "
        "already handled (alpha-engine-config#5223)."
    )


def _krepis_floor() -> tuple[int, int, int]:
    """Parse the resolved krepis version out of requirements.txt.

    Accepts either `krepis>=X.Y.Z` (a floor) or `krepis==X.Y.Z` (an exact
    pin, per §139 — first-party/fast-moving deps are pinned, never
    floored). The property under test is "the krepis this image resolves
    is at or above X" — an exact pin establishes that property MORE
    strongly than a floor (it IS the resolved version, not a lower bound
    on it), so the comparator itself is not what matters here
    (alpha-engine-config-I7635).
    """
    text = (_REPO_ROOT / "requirements.txt").read_text()
    match = re.search(r"^krepis\s*(?:>=|==)\s*(\d+)\.(\d+)\.(\d+)", text, re.MULTILINE)
    assert match, "requirements.txt must pin a krepis floor or exact pin"
    return tuple(int(g) for g in match.groups())  # type: ignore[return-value]


def test_krepis_floor_is_at_or_above_the_superset_release():
    """Below 0.25.0 the migration onto krepis.llm is a REGRESSION, not a port.

    Measured 2026-07-30: on 0.24.x a null-choices body through the migrated
    ThinktankClient raised ``TypeError: 'NoneType' object is not subscriptable``
    — the bug tests/test_null_choices_provider_error.py exists to prevent,
    reintroduced by adopting shared code that lacked the fork's guard.
    """
    assert _krepis_floor() >= _KREPIS_SUPERSET_FLOOR, (
        f"krepis floor {_krepis_floor()} is below {_KREPIS_SUPERSET_FLOOR}, the "
        "first release where krepis.llm is a superset of the fork it replaced "
        "(krepis#93: null-choices guard at all five choices[0] reads, plus "
        "backoff between body-level retries)."
    )


def test_krepis_floor_regex_parses_both_floor_and_exact_pin_forms():
    """Regression guard for the exact defect I7635 fixed: converting the
    floor to an exact pin must not make this guard report the requirement
    as ABSENT."""
    pattern = re.compile(r"^krepis\s*(?:>=|==)\s*(\d+)\.(\d+)\.(\d+)", re.MULTILINE)
    floor_match = pattern.search("krepis >= 0.59.16")
    exact_match = pattern.search("krepis==0.59.16")
    assert floor_match is not None
    assert exact_match is not None
    assert floor_match.groups() == exact_match.groups() == ("0", "59", "16")


@pytest.mark.parametrize("package", _LLM_PATH_PACKAGES)
def test_the_guard_actually_scans_something(package):
    """A guard over an empty file set passes forever and proves nothing —
    the failure mode this whole test module exists to catch, one level up."""
    root = _REPO_ROOT / package
    if not root.is_dir():
        pytest.skip(f"{package}/ not present")
    assert any(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
