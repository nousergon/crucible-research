"""The scanner sizes ArcticDB's S3 I/O pool before arcticdb loads (alpha-engine-config-I12271).

Two weekly-freshness rehearsals (10-14, 10-15) died at the Scanner's 440 s
budget. Every scanner read is a ``read_batch`` over ~900 symbols and is bound
by S3 round trips, and the weekday ``daily_append`` adds one segment per
symbol per session until the Saturday collection rewrites it: 1 segment per
symbol at every successful run, 5-6 at the two failed rehearsals. A
CPU-sized I/O pool turned that into 3x slower reads. These tests pin the
mechanism, not the timing.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_HANDLER_PATH = _REPO_ROOT / "lambda" / "scanner_handler.py"
_ENV_KEY = "ARCTICDB_VersionStore_NumIOThreads_int"


def _module_body() -> list[ast.stmt]:
    return ast.parse(_HANDLER_PATH.read_text(encoding="utf-8")).body


def _is_io_threads_setdefault(node: ast.stmt) -> bool:
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    call = node.value
    return (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "setdefault"
        and bool(call.args)
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == _ENV_KEY
    )


def test_io_threads_are_set_before_any_first_party_or_third_party_import():
    """ArcticDB reads ARCTICDB_* env config once, at import.

    Setting it after anything that could import arcticdb (graph.*,
    nousergon_lib.*, data.*) would leave the pool at its CPU-sized default
    with no error anywhere, which is the state that timed out.
    """
    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    body = _module_body()
    idx = next((i for i, n in enumerate(body) if _is_io_threads_setdefault(n)), None)
    assert idx is not None, f"scanner_handler.py must os.environ.setdefault({_ENV_KEY!r}, ...)"
    for node in body[:idx]:
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.module or "").split(".")[0]]
        else:
            continue
        for name in names:
            assert name in stdlib, f"{name!r} is imported before the ArcticDB I/O pool is sized"


def test_io_thread_default_is_well_above_the_cpu_sized_pool():
    """32 measured 8-10x faster than 3 on a 150-symbol live read."""
    node = next(n for n in _module_body() if _is_io_threads_setdefault(n))
    value = node.value.args[1]  # type: ignore[attr-defined]
    assert isinstance(value, ast.Constant) and isinstance(value.value, str)
    assert int(value.value) >= 16


def test_env_key_is_the_one_arcticdb_honours():
    """A misspelled key is silently ignored, so prove arcticdb reads it."""
    pytest.importorskip("arcticdb")
    code = (
        "import arcticdb; from arcticdb_ext import get_config_int; print(get_config_int('VersionStore.NumIOThreads'))"
    )
    env = {**os.environ, _ENV_KEY: "32"}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "32"
