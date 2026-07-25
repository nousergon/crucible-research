"""Regression guard: the research runner's ``dry_run_llm`` deploy-canary mode
must be a DETERMINISTIC, side-effect-free boot validation.

Root cause of the 2026-07-21 Deploy red: ``dry_run_llm`` only short-circuited
AFTER the wall-clock time gate, preflight, ``download_db()`` and the
scorecard / team-accuracy S3 emitters. A deploy landing inside the
5:40-5:55am PT weekday gate window (so the time gate did NOT return SKIPPED)
therefore ran real S3/DB work in the canary; a transient failure there
returned ``status=ERROR`` and tripped a spurious auto-rollback.

The fix hoists the ``dry_run_llm`` boot validation ABOVE the time gate and
all pre-graph I/O, matching the fleet convention (thinktank /
aggregate_costs / rationale_clustering / eval_judge dry paths). These tests
lock that in: the dry canary must

  1. never consult the wall clock (``_is_scheduled_run_time`` not called),
  2. never touch S3/DB (``download_db`` not called),
  3. still exercise the boot surface (``build_graph`` + ``create_initial_state``),
  4. return ``{"status": "OK", "phase": "boot_validation"}``.
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

_HANDLER_PATH = Path(__file__).parent.parent / "lambda" / "handler.py"


def _import_handler():
    """lambda/ collides with the keyword — load via importlib."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "research_handler_dry_run_canary_test", _HANDLER_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dry_run_llm_boots_without_clock_or_s3():
    mod = _import_handler()
    am = MagicMock()
    restore = MagicMock()

    with patch.object(mod, "_ensure_init"), \
         patch.object(mod, "_is_scheduled_run_time") as sched, \
         patch.object(mod, "most_recent_trading_day",
                      return_value=datetime.date(2026, 7, 20)), \
         patch("archive.manager.ArchiveManager", return_value=am), \
         patch("dry_run.install_dry_run_stubs", return_value=restore) as stubs, \
         patch("graph.research_graph.build_graph") as build_graph, \
         patch("graph.research_graph.create_initial_state") as create_state:
        res = mod.handler({"dry_run_llm": True}, None)

    assert res["status"] == "OK"
    assert res["phase"] == "boot_validation"
    assert res["dry_run_llm"] is True
    # (1) never consults the wall clock — deterministic regardless of when a
    #     deploy lands (this is the crux of the 2026-07-21 fix).
    sched.assert_not_called()
    # (2) no S3 / DB side effects in the canary.
    am.download_db.assert_not_called()
    # (3) still validates the boot surface (node wiring + imports + state).
    build_graph.assert_called_once()
    create_state.assert_called_once()
    # stubs installed + restored around the boot validation.
    stubs.assert_called_once()
    restore.assert_called_once()
