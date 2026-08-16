"""No module may construct its OWN cost sink (alpha-engine-config-I7423).

`krepis.cost_sink.flush_default_sink()` — which every handler in `lambda/`
now calls in a `finally` — flushes the PROCESS-DEFAULT sink and nothing else.
A module that constructs a private `S3JsonlCostSink` shadows it: the records go
into an instance nothing else holds a reference to, whose only exit path is
`register_atexit=True`. An AWS Lambda container is FROZEN between invocations,
not exited, so `atexit` never runs and every record below the 200-per-group
buffer threshold dies in memory.

Measured 2026-08-16 on weekly-SF execution `watch-rerun-2026-08-16-1`, AFTER
the handler-level flush had shipped and deployed: `ChallengerShadow` made a
real DeepSeek call (`input_tokens=4296 output_tokens=7906`), wrote both shadow
signal sets, and `AggregateCosts` still failed the run with

    cost fan-in coverage BREACH — 1 stage(s) ran and emitted no cost record:
    single-agent-quant. Observed producers: replay-concordance.

`replay-concordance` — which passes `cost_sink=None` and lets `LLMClient`
resolve the default — appeared. `single-agent-quant`, which built its own,
did not. The difference is entirely this.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent

# thinktank/run.py is the ONE deliberate exception. It runs on a spot box in a
# process that genuinely exits, so its atexit hook fires, and it threads an
# explicit `run_id` into the sink that the env-resolved default would replace —
# changing the `_cost_raw/{date}/{run_id}/` key layout its historical
# partitions already use. It is also the only callsite whose records have ever
# reliably appeared. Moving it is a migration, not a bug fix.
_EXEMPT = {"thinktank/run.py"}

_SOURCE_DIRS = ("producers", "evals", "lambda", "graph", "agents", "scoring")


def _python_files() -> list[Path]:
    out: list[Path] = []
    for d in _SOURCE_DIRS:
        root = _REPO_ROOT / d
        if root.is_dir():
            out.extend(sorted(root.rglob("*.py")))
    return out


def test_the_scan_covers_something():
    """An empty file list would make every assertion below vacuous."""
    assert len(_python_files()) >= 20, len(_python_files())


@pytest.mark.parametrize(
    "path", _python_files(), ids=lambda p: str(p.relative_to(_REPO_ROOT))
)
def test_no_module_constructs_its_own_cost_sink(path: Path):
    rel = str(path.relative_to(_REPO_ROOT))
    if rel in _EXEMPT:
        pytest.skip(f"{rel} is a declared exception — see _EXEMPT")
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "S3JsonlCostSink"
        ):
            raise AssertionError(
                f"{rel}:{node.lineno} constructs a private S3JsonlCostSink. Use "
                f"krepis.cost_sink.default_sink_from_env(), or pass cost_sink=None "
                f"and let LLMClient resolve it — a private instance is invisible "
                f"to flush_default_sink() and its records die in a frozen Lambda "
                f"(config-I7423)."
            )


def test_the_exemption_list_still_describes_reality():
    """A stale exemption silently re-permits the defect."""
    for rel in _EXEMPT:
        path = _REPO_ROOT / rel
        assert path.exists(), f"exempt path {rel} no longer exists — drop it"
        assert "S3JsonlCostSink(" in path.read_text(), (
            f"{rel} no longer constructs its own sink — remove it from _EXEMPT "
            f"so the guard covers it"
        )


def test_single_agent_lets_the_client_resolve_the_sink():
    """The exact callsite that failed watch-rerun-2026-08-16-1."""
    tree = ast.parse((_REPO_ROOT / "producers" / "single_agent.py").read_text())
    constructions = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "S3JsonlCostSink"
    ]
    assert not constructions, (
        "single_agent constructs a private sink again — flush_default_sink() "
        "cannot see it and single-agent-quant returns to emitting nothing"
    )


def test_judge_returns_the_env_resolved_default():
    src = (_REPO_ROOT / "evals" / "judge.py").read_text()
    assert "default_sink_from_env" in src, (
        "the judge chain must resolve the process default, not memoise its own"
    )
