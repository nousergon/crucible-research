"""Every Lambda handler flushes the cost sink before returning (config-I7423).

An AWS Lambda container is FROZEN between invocations, not exited, so
``krepis.cost_sink.S3JsonlCostSink``'s ``atexit`` hook never runs. The sink
buffers to 200 records per ``(date, callsite_id)`` group, so a handler that
finishes below the threshold writes NOTHING and the container may be reclaimed
hours later without ever reaching interpreter shutdown.

Measured 2026-08-15 on weekly-SF execution ``watch-rerun-2026-08-15-2``:
``AggregateCosts`` failed the whole run with ``2 stage(s) ran and emitted no
cost record: replay-concordance, single-agent-quant ... Observed producers:
(none)``. The env wiring was correct (config-I7179), the sink was constructed,
the records were priced and accepted, and every one of them died in memory.

This file is the cheap structural guard the issue asks for: a NEW LLM-calling
handler cannot ship without the flush, and it does not depend on anyone
correctly predicting which handlers reach a model.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_LAMBDA_DIR = Path(__file__).resolve().parent.parent / "lambda"
_HANDLERS = sorted(_LAMBDA_DIR.glob("*.py"))


def test_handler_directory_is_not_empty():
    """Guards the parametrisation itself: an empty glob passes vacuously."""
    assert len(_HANDLERS) >= 10, f"only found {len(_HANDLERS)} handlers"


@pytest.mark.parametrize("path", _HANDLERS, ids=lambda p: p.name)
def test_handler_flushes_the_cost_sink_in_a_finally(path: Path):
    tree = ast.parse(path.read_text())
    handler = next(
        (n for n in tree.body
         if isinstance(n, ast.FunctionDef) and n.name == "handler"),
        None,
    )
    assert handler is not None, f"{path.name} defines no module-level `handler`"

    tries = [n for n in ast.walk(handler) if isinstance(n, ast.Try) and n.finalbody]
    assert tries, (
        f"{path.name}: handler has no try/finally. The flush must be in a "
        f"`finally` — every handler here has multiple return paths, and a "
        f"flush on one of them is exactly the defect config-I7423 fixed."
    )

    flushed = any(
        isinstance(node, ast.Name) and node.id == "flush_default_sink"
        or isinstance(node, ast.Attribute) and node.attr == "flush_default_sink"
        or isinstance(node, ast.alias) and node.name == "flush_default_sink"
        for t in tries for stmt in t.finalbody for node in ast.walk(stmt)
    )
    assert flushed, (
        f"{path.name}: handler's `finally` does not call "
        f"krepis.cost_sink.flush_default_sink() — cost records for this "
        f"Lambda die in the in-process buffer (config-I7423)"
    )


@pytest.mark.parametrize("path", _HANDLERS, ids=lambda p: p.name)
def test_flush_failure_cannot_break_the_handler(path: Path):
    """The flush is import-guarded: telemetry must not take down the work.

    ``flush_default_sink`` itself never raises, but the import can fail on an
    image whose krepis pin predates it.
    """
    tree = ast.parse(path.read_text())
    handler = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "handler"
    )
    guarded = any(
        isinstance(n, ast.Try)
        and any(
            isinstance(node, ast.alias) and node.name == "flush_default_sink"
            for node in ast.walk(n)
        )
        and n.handlers
        for n in ast.walk(handler)
    )
    assert guarded, (
        f"{path.name}: the flush import is not exception-guarded — an image "
        f"with krepis <0.59.8 would fail the handler on a telemetry concern"
    )
