"""Tests for the `challengers_only` operator mode + the fail-hard post-step
wiring (config#1683).

The mode re-emits the challenger shadow cohort for the LATEST weekly run
(recovery path for a missed cohort) by reconstructing the prior population
membership-exactly (current population minus rows entered on run_date), and
refuses any non-latest date fail-loud.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

_HANDLER_PATH = Path(__file__).parent.parent / "lambda" / "handler.py"


def _import_handler():
    """lambda/ collides with the keyword — load via importlib."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "research_handler_challengers_only_test", _HANDLER_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _archive_stub(latest_date: str, population: list[dict]) -> MagicMock:
    am = MagicMock()
    am.bucket = "test-bucket"
    body = MagicMock()
    body.read.return_value = json.dumps({"date": latest_date}).encode()
    am.s3.get_object.return_value = {"Body": body}
    am.load_population.return_value = population
    return am


def test_challengers_only_reconstructs_prior_population_and_runs():
    mod = _import_handler()
    population = [
        {"ticker": "HELD1", "entry_date": "2026-05-01"},
        {"ticker": "HELD2", "entry_date": "2026-06-12"},
        {"ticker": "NEW1", "entry_date": "2026-07-02"},
    ]
    am = _archive_stub("2026-07-02", population)

    with patch("archive.manager.ArchiveManager", return_value=am), \
         patch("producers.runner.run_challengers",
               return_value={"written": {"no_agent_quant": "k1"}, "errors": {}}) as rc:
        res = mod._run_challengers_only(
            {"mode": "challengers_only", "date": "2026-07-02"}
        )

    assert res["status"] == "OK" and res["written"] == {"no_agent_quant": "k1"}
    # Prior population = membership minus the rows this run entered.
    passed_pop = rc.call_args.kwargs["population"]
    assert [p["ticker"] for p in passed_pop] == ["HELD1", "HELD2"]
    am.download_db.assert_called_once()
    am.close.assert_called_once()  # closed even on success (finally)


def test_challengers_only_refuses_non_latest_date():
    """The prior-population reconstruction is only membership-exact against
    the most recent population commit — any other date must fail loud."""
    mod = _import_handler()
    am = _archive_stub("2026-07-02", [])

    with patch("archive.manager.ArchiveManager", return_value=am), \
         patch("producers.runner.run_challengers") as rc:
        with pytest.raises(ValueError, match="only valid for the latest run"):
            mod._run_challengers_only({"date": "2026-06-26"})
    rc.assert_not_called()
    am.close.assert_called_once()  # finally still closes the archive


def test_challengers_only_requires_date():
    mod = _import_handler()
    with pytest.raises(ValueError, match="requires event\\['date'\\]"):
        mod._run_challengers_only({"mode": "challengers_only"})


def test_handler_wiring_failhard_and_mode_dispatch():
    """Source-level pins:
    1. the challengers_only dispatch sits BEFORE the time/trading gates,
    2. the old fail-soft swallow around the challenger post-step is GONE —
       run_challengers is invoked bare so a gap fails the run (config#1683).
    """
    src = _HANDLER_PATH.read_text()
    dispatch = src.index('event.get("mode") == "challengers_only"')
    gates = src.index('force = event.get("force", False)')
    assert dispatch < gates, (
        "challengers_only must dispatch before the run-time/trading-day gates"
    )
    assert "shadow mode, non-fatal" not in src, (
        "the fail-soft swallow around the challenger post-step was "
        "re-introduced — config#1683 made experiment producers FAIL-HARD"
    )
    assert "from producers.runner import run_challengers" in src
