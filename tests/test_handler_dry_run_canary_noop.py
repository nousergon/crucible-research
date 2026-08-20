"""Regression guard: the research runner's ``dry_run_llm`` deploy-canary mode
must be a DETERMINISTIC, side-effect-free boot validation OF THE LIVE PATH.

Two root causes are locked in here.

**2026-07-21 (determinism).** ``dry_run_llm`` only short-circuited AFTER the
wall-clock time gate, preflight, ``download_db()`` and the scorecard /
team-accuracy S3 emitters. A deploy landing inside the 5:40-5:55am PT weekday
gate window (so the time gate did NOT return SKIPPED) therefore ran real S3/DB
work in the canary; a transient failure there returned ``status=ERROR`` and
tripped a spurious auto-rollback. The boot validation was hoisted above all
pre-run I/O.

**2026-08-20 (aim — alpha-engine-config-I7827).** The hoisted check was still
calling ``graph.research_graph.build_graph()``: a producer retired 2026-07-12,
with no invoker in production since the weekly SF dropped its ``Research``
state on 2026-07-14. The deploy's own safety check was smoke-testing dead code
and NOT smoke-testing ``producers/``, so a deploy that broke the live producers
passed the canary green. It now runs
``producers.boot_check.run_live_producer_boot_check``.

The canary must:

  1. never consult the wall clock (no ``_is_scheduled_run_time`` — the symbol
     no longer exists, and the trading-day/time gates went with the champion
     pass),
  2. never touch S3/DB (``download_db`` not called),
  3. exercise the LIVE producer boot surface,
  4. return ``{"status": "OK", "phase": "boot_validation"}`` naming the
     producers it covered,
  5. install the dry-run stubs around it and restore them,
  6. propagate a boot-check failure as a canary failure (so the deploy's
     rollback path fires) rather than swallowing it.
"""

from __future__ import annotations

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
        "research_handler_dry_run_canary_test", _HANDLER_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dry_run_llm_boots_the_live_producer_path_without_clock_or_s3():
    mod = _import_handler()
    am = MagicMock()
    restore = MagicMock()

    with patch.object(mod, "_ensure_init"), \
         patch("nousergon_lib.dates.resolve_trading_day",
               return_value="2026-07-20"), \
         patch("archive.manager.ArchiveManager", return_value=am), \
         patch("dry_run.install_dry_run_stubs", return_value=restore) as stubs, \
         patch("producers.boot_check.run_live_producer_boot_check",
               return_value={"producers": ["no_agent_quant"],
                             "modules": ["producers.no_agent"],
                             "payload_keys": ["signals"]}) as boot_check:
        res = mod.handler({"dry_run_llm": True}, None)

    assert res["status"] == "OK"
    assert res["phase"] == "boot_validation"
    assert res["dry_run_llm"] is True
    # (4) says WHAT it covered — a check reporting only "OK" cannot be told
    #     apart from a check that silently covered nothing.
    assert res["producers"] == ["no_agent_quant"]
    # (1) the handler no longer has a wall-clock gate at all.
    assert not hasattr(mod, "_is_scheduled_run_time")
    # (2) no S3 / DB side effects in the canary.
    am.download_db.assert_not_called()
    # (3) the LIVE producer path is what got exercised.
    boot_check.assert_called_once()
    # (5) stubs installed + restored around the boot validation.
    stubs.assert_called_once()
    restore.assert_called_once()


def test_dry_run_llm_never_reaches_the_retired_graph():
    """The canary must not import the deleted module under any name."""
    mod = _import_handler()
    src = _HANDLER_PATH.read_text()
    # The only permitted mentions are the explanatory comments recording WHY
    # it was removed; there must be no import statement.
    assert "from graph.research_graph import" not in src
    assert "import graph.research_graph" not in src
    assert not hasattr(mod, "build_graph")
    assert not hasattr(mod, "create_initial_state")
    # The module is gone from the tree entirely — a re-added file would make
    # the import assertions above meaningful again rather than vacuous.
    assert not (_HANDLER_PATH.parent.parent / "graph" / "research_graph.py").exists()


def test_a_broken_live_producer_path_fails_the_canary():
    """(6) The boot check raising must surface as a canary failure, not be
    swallowed — the deploy's rollback path reads this return."""
    mod = _import_handler()
    am = MagicMock()
    restore = MagicMock()

    from producers.boot_check import LiveProducerBootCheckError

    with patch.object(mod, "_ensure_init"), \
         patch("nousergon_lib.dates.resolve_trading_day",
               return_value="2026-07-20"), \
         patch("archive.manager.ArchiveManager", return_value=am), \
         patch("dry_run.install_dry_run_stubs", return_value=restore), \
         patch("producers.boot_check.run_live_producer_boot_check",
               side_effect=LiveProducerBootCheckError("producers/ is broken")):
        with pytest.raises(LiveProducerBootCheckError):
            mod.handler({"dry_run_llm": True}, None)

    # Stubs restored even on the failure path — a Lambda container is reused.
    restore.assert_called_once()
