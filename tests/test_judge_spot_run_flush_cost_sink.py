"""``evals.judge_spot_run`` flushes the cost sink on exit (config-I7423).

The spot substrate replaced the Lambda Process handler but inherited the same
``default_sink_from_env`` wiring. Without an explicit flush, a full judge
corpus (~114 records) stays below the 200-record buffer threshold and
``evaljudge-sync`` emits nothing — ``AggregateCosts`` then names
``EvalJudgeProcess`` (measured watch-rerun-2026-08-28-8, 2026-08-30).
"""

from __future__ import annotations

import ast
from pathlib import Path


def test_judge_spot_run_main_flushes_cost_sink_in_finally():
    path = Path(__file__).resolve().parent.parent / "evals" / "judge_spot_run.py"
    tree = ast.parse(path.read_text())
    main_fn = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    tries = [n for n in ast.walk(main_fn) if isinstance(n, ast.Try) and n.finalbody]
    assert tries, "main() must wrap the run body in try/finally for cost flush"
    flushed = any(
        isinstance(node, ast.Name) and node.id == "flush_default_sink"
        or isinstance(node, ast.Attribute) and node.attr == "flush_default_sink"
        for t in tries for stmt in t.finalbody for node in ast.walk(stmt)
    )
    assert flushed, (
        "main()'s finally must call krepis.cost_sink.flush_default_sink()"
    )
