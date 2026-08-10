"""Guard: this repo has exactly ONE trading-day normalizer, and it is the lib's.

config-I6667. `nousergon_lib.dates.resolve_trading_day` is the fleet chokepoint
for "the most recent NYSE trading day on or before this date" — its own
docstring records that it was lifted there on the second adoption, replacing
`pipeline_common.resolve_trading_day` (backtester) and
`grading.handler._to_trading_day` (evaluator).

`crucible-research` carried two further reimplementations until I6667:
an inline block in `lambda/scanner_handler.py` and
`lambda/handler.py::most_recent_trading_day`. Three implementations of one
question is the condition that makes "did this call site remember to
normalize?" something anyone has to ask — and config-I6653 was exactly a call
site that had not, silently keying `signals/` by the calendar date while the
Scanner in the same execution keyed by the trading day.

The two contracts also differ: the lib helper is DEFENSIVE (a parse failure
returns the input unchanged with a WARNING, so a normalization miss cannot
abort the caller), while the inline block raised out of `date.fromisoformat`.
Which contract a call site got was decided by which copy it happened to use.

This guard fails when a fourth copy arrives. It is deliberately a source scan
rather than a behavioural test: the defect class is *duplication*, and two
copies that agree today still fail this test — which is the point.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO = Path(__file__).parent.parent
_SCANNED_DIRS = ("lambda", "scoring")

# The on-or-before construction is what makes something a normalizer. The lib
# helper is the only permitted way to answer it, so no module under the scanned
# directories may call the calendar's rewind primitive directly.
_REWIND_CALL = re.compile(r"\bprevious_trading_day\s*\(")

# A local `def` whose name reads as this operation is the other shape the
# duplication takes (the removed `most_recent_trading_day`).
_LOCAL_NORMALIZER_DEF = re.compile(
    r"^\s*def\s+_?(most_recent_trading_day|resolve_trading_day|to_trading_day|"
    r"_to_trading_day|normalize_trading_day)\s*\(",
    re.MULTILINE,
)


def _scanned_files() -> list[Path]:
    files: list[Path] = []
    for d in _SCANNED_DIRS:
        files.extend(sorted((_REPO / d).rglob("*.py")))
    return files


def test_scanned_dirs_are_non_empty():
    """A guard that scans nothing passes for the wrong reason."""
    files = _scanned_files()
    assert len(files) > 5, f"expected to scan lambda/ and scoring/, found {files}"


@pytest.mark.parametrize("path", _scanned_files(), ids=lambda p: str(p.name))
def test_no_local_trading_day_normalizer(path: Path):
    src = path.read_text(encoding="utf-8")

    rewinds = [
        line
        for line in src.splitlines()
        if _REWIND_CALL.search(line) and not line.lstrip().startswith("#")
    ]
    assert not rewinds, (
        f"{path.relative_to(_REPO)} calls previous_trading_day() directly — that is "
        f"the on-or-before rewind primitive, and building the normalizer out of it "
        f"here re-introduces the duplicate config-I6667 removed. Use "
        f"`from nousergon_lib.dates import resolve_trading_day` instead.\n"
        f"  offending: {rewinds}"
    )

    local_defs = _LOCAL_NORMALIZER_DEF.findall(src)
    assert not local_defs, (
        f"{path.relative_to(_REPO)} defines a local trading-day normalizer "
        f"{local_defs} — the fleet chokepoint is "
        f"`nousergon_lib.dates.resolve_trading_day` (config-I6667). Convert types "
        f"at the call site rather than wrapping the lib helper, or the wrapper "
        f"becomes the next copy that drifts."
    )


def test_the_permitted_import_is_actually_used():
    """The guard must not pass by everyone having stopped normalizing at all.

    I6653's defect was a MISSING normalizer, not a wrong one. If the two known
    normalization sites stop resolving through the lib, this repo has regressed
    to the state that produced the calendar-vs-trading-day key split.
    """
    required = {
        "lambda/scanner_handler.py",
        "lambda/signals_envelope_handler.py",
        "lambda/handler.py",
    }
    for rel in sorted(required):
        src = (_REPO / rel).read_text(encoding="utf-8")
        assert "from nousergon_lib.dates import resolve_trading_day" in src, (
            f"{rel} no longer imports the shared normalizer — every entry point "
            f"that keys an artifact must resolve its run_date through "
            f"`nousergon_lib.dates.resolve_trading_day` (config-I6653, I6667)."
        )
