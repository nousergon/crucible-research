"""Producer-side contract: a coverage verdict is keyed by the TRADING day.

alpha-engine-config-I10171 / I10194. Mirrors
``nousergon-data/tests/test_weekly_partition_family_contract.py`` on the
producer side of the same boundary: that repo asserts the Step Function
DECLARES the partition family each date-touching state uses; this file
asserts the handlers this repo deploys actually KEY their
``_stage_coverage`` verdict by it.

The defect it exists for: five weekly stages derived their own key from
``event["end_time_iso"]`` (the raw ``$$.Execution.StartTime``) and wrote
every verdict into the CALENDAR partition. On 2026-09-05 the reader's
dual-partition fallback expired by design and all five began reading
**absent** — indistinguishable from a stage that never ran.

Population is DERIVED, not listed (this file names none of the five). It is
the union of:

  * ``test_stage_coverage_assertions._WEEKLY_SF_STAGE_TO_HANDLER_FILE``'s
    values — the repo's declared weekly-SF stage → handler map, itself held
    against the SF by the totality tests in that module; and
  * every ``.py`` under ``lambda/`` and ``evals/`` whose source contains an
    ``assert_stage_coverage(`` call.

The second term is what makes a SIXTH handler added tomorrow subject to the
same rule without anyone remembering to edit a list here. A hand-written
list of five names would pass forever while the class re-opened next to it.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent

_RESOLVER = "resolve_stage_run_date"


def _load_repo_module(rel_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, _REPO_ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _declared_handler_files() -> set[str]:
    """The declared weekly-SF stage → handler map's values."""
    mod = _load_repo_module(
        "tests/test_stage_coverage_assertions.py",
        "_i10171_declared_stage_map",
    )
    return set(mod._WEEKLY_SF_STAGE_TO_HANDLER_FILE.values())


def _asserting_source_files() -> set[str]:
    """Every repo file that calls ``assert_stage_coverage(``.

    Scanned rather than listed — this is the term that catches a handler
    added after this test was written.
    """
    found: set[str] = set()
    for directory in ("lambda", "evals"):
        for path in sorted((_REPO_ROOT / directory).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            if "assert_stage_coverage(" in path.read_text():
                found.add(str(path.relative_to(_REPO_ROOT)))
    return found


def _coverage_asserting_files() -> list[str]:
    return sorted(_declared_handler_files() | _asserting_source_files())


def _call_args(source: str, needle: str) -> list[str]:
    """Paren-matched argument text of every ``needle(...)`` call."""
    calls: list[str] = []
    idx = source.find(needle + "(")
    while idx != -1:
        i = idx + len(needle) + 1
        depth = 1
        while i < len(source) and depth:
            if source[i] == "(":
                depth += 1
            elif source[i] == ")":
                depth -= 1
            i += 1
        calls.append(source[idx + len(needle) + 1 : i - 1])
        idx = source.find(needle + "(", i)
    return calls


class TestPopulationIsDerived:
    def test_the_scan_finds_the_handlers(self):
        """A scan that silently matched nothing would make every test below
        vacuously green — the exact blindness this file exists to close."""
        files = _coverage_asserting_files()
        assert len(files) >= 5, files

    def test_the_scan_is_not_merely_the_declared_map(self):
        """The derived term must contribute at least one file the map does
        not name, or the population is a hand-written list wearing a scan's
        clothes."""
        assert _asserting_source_files(), "no file in lambda/ or evals/ calls assert_stage_coverage"


class TestEveryAssertingFileUsesTheResolver:
    @pytest.mark.parametrize("filename", _coverage_asserting_files())
    def test_file_resolves_its_run_date_through_the_shared_resolver(self, filename):
        source = (_REPO_ROOT / filename).read_text()
        assert _RESOLVER in source, (
            f"{filename} asserts stage coverage but never calls "
            f"{_RESOLVER} — its partition key is derived locally and can "
            "silently be the CALENDAR date (alpha-engine-config-I10171)"
        )

    @pytest.mark.parametrize("filename", _coverage_asserting_files())
    def test_no_file_keys_a_verdict_on_a_bare_execution_start_time(self, filename):
        """``end_time_iso`` / ``$$.Execution.StartTime`` derivations may
        still exist — as the resolver's NAMED ``fallback=``, never as the
        value handed straight to ``assert_stage_coverage``."""
        source = (_REPO_ROOT / filename).read_text()
        for args in _call_args(source, "assert_stage_coverage"):
            assert "end_time" not in args, (
                f"{filename} passes an end_time-derived value directly to "
                f"assert_stage_coverage: {args!r}. Route it through "
                f"{_RESOLVER}(event, stage=..., fallback=...) so the "
                "SF-threaded trading day wins and the fallback is recorded "
                "(alpha-engine-config-I10171)."
            )

    @pytest.mark.parametrize("filename", _coverage_asserting_files())
    def test_every_resolver_call_names_its_fallback_source(self, filename):
        source = (_REPO_ROOT / filename).read_text()
        for args in _call_args(source, _RESOLVER):
            if "fallback=" in args:
                assert "fallback_source=" in args, (
                    f"{filename} calls {_RESOLVER} with an unnamed fallback: "
                    f"{args!r} — an unnamed fallback is an unrecorded one"
                )
            assert "stage=" in args, (
                f"{filename} calls {_RESOLVER} without a stage= name: {args!r}"
            )


class TestResolverPrefersTheTradingDay:
    @pytest.fixture
    def resolver(self):
        return _load_repo_module(
            "stage_coverage_run_date.py", "_i10171_stage_coverage_run_date"
        )

    def test_event_run_date_wins_over_the_fallback(self, resolver):
        """The whole fix in one assertion: a Saturday execution whose
        calendar date is 2026-09-05 keys its verdict on the trading day
        2026-09-04 that the SF threaded."""
        run_date, prov = resolver.resolve_stage_run_date(
            {"run_date": "2026-09-04", "end_time_iso": "2026-09-05T06:00:00Z"},
            stage="EvalRollingMean",
            fallback="2026-09-05",
            fallback_source="event.end_time_iso",
        )
        assert run_date == "2026-09-04"
        assert prov == {"run_date_source": "event.run_date"}
        assert "run_date_fallback_reason" not in prov

    def test_fallback_is_used_only_when_run_date_is_absent(self, resolver):
        run_date, prov = resolver.resolve_stage_run_date(
            {"end_time_iso": "2026-09-05T06:00:00Z"},
            stage="EvalRollingMean",
            fallback="2026-09-05",
            fallback_source="event.end_time_iso",
        )
        assert run_date == "2026-09-05"
        assert prov["run_date_source"] == "fallback:event.end_time_iso"

    def test_the_fallback_is_recorded_not_silent(self, resolver, caplog):
        with caplog.at_level(logging.WARNING):
            _, prov = resolver.resolve_stage_run_date(
                {"end_time_iso": "2026-09-05T06:00:00Z"},
                stage="RationaleClustering",
                fallback="2026-09-05",
                fallback_source="event.end_time_iso",
            )
        assert "run_date_fallback_reason" in prov, (
            "a fallback nobody can see afterwards reproduces the defect"
        )
        assert "event.end_time_iso" in prov["run_date_fallback_reason"]
        assert any(r.levelno >= logging.WARNING for r in caplog.records), (
            "the fallback must log at WARNING or worse — fail loud, no "
            "silent swallows"
        )

    def test_a_blank_run_date_is_not_preferred(self, resolver):
        run_date, prov = resolver.resolve_stage_run_date(
            {"run_date": "   ", "end_time_iso": "2026-09-05T06:00:00Z"},
            stage="Counterfactual",
            fallback="2026-09-05",
            fallback_source="event.end_time_iso",
        )
        assert run_date == "2026-09-05"
        assert prov["run_date_source"] == "fallback:event.end_time_iso"

    def test_a_timestamp_run_date_is_truncated_to_its_date_part(self, resolver):
        run_date, _ = resolver.resolve_stage_run_date(
            {"run_date": "2026-09-04T00:00:00Z"}, stage="SignalsEnvelope",
        )
        assert run_date == "2026-09-04"

    def test_nothing_usable_returns_none_and_never_fabricates(self, resolver, caplog):
        with caplog.at_level(logging.ERROR):
            run_date, prov = resolver.resolve_stage_run_date(
                {}, stage="EvalJudgeSubmitFirstSaturday",
            )
        assert run_date is None, "must never substitute today's date (config-I8155)"
        assert prov == {"run_date_source": "none"}
        assert any(r.levelno >= logging.ERROR for r in caplog.records)

    def test_a_non_mapping_event_is_not_a_crash(self, resolver):
        run_date, prov = resolver.resolve_stage_run_date(
            None, stage="EvalRollingMean", fallback="2026-09-05",
            fallback_source="event.end_time_iso",
        )
        assert run_date == "2026-09-05"
        assert prov["run_date_source"] == "fallback:event.end_time_iso"

    def test_an_unnamed_fallback_raises_at_the_call_site(self, resolver):
        with pytest.raises(ValueError, match="fallback_source"):
            resolver.resolve_stage_run_date(
                {}, stage="EvalRollingMean", fallback="2026-09-05",
            )
