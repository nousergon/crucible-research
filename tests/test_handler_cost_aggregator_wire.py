"""Locks the cost-aggregator PACKAGING invariants.

Per ROADMAP P2 "SF-wire the aggregate_costs.py CLI" (closed 2026-05-02):
the aggregator must run automatically at the end of every successful
Research Lambda invocation so the Backtester evaluator email's
``## LLM cost report`` section has data to render — no manual CLI step
between Research and Backtester.

Source-text invariants (vs spinning up the heavy handler module) — same
shape as test_decoupled_structured_extraction's architectural-lock tests.
"""

from __future__ import annotations

from pathlib import Path

_HANDLER_PATH = Path(__file__).parent.parent / "lambda" / "handler.py"


def _strip_comments_and_strings(src: str) -> str:
    """Drop comments + triple-quoted strings so a forbidden-pattern check
    isn't tripped by mention of the pattern in a comment."""
    import re
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    src = re.sub(r"'''[\s\S]*?'''", "", src)
    src = re.sub(r"(?m)^\s*#.*$", "", src)
    src = re.sub(r"(?m)\s*#[^\n]*$", "", src)
    return src


# ── Three tests DELETED with the champion graph (alpha-engine-config-I7827) ──
# `test_handler_invokes_aggregate_day_on_success`,
# `test_aggregator_call_is_gated_on_email_sent` and
# `test_aggregator_failure_is_non_fatal` asserted source-text invariants over
# `lambda/handler.py`'s INLINE cost-aggregation tail: the `aggregate_day(...)`
# call gated on `final_state["email_sent"]` at the end of the champion pass.
# That whole tail was deleted with the retired graph, and the capability did
# not go with it — `lambda/aggregate_costs_handler.py` is the live invoker
# (the weekly SF's AggregateCosts stage, ROADMAP L1146), with its own tests.
# Keeping the three would have been testing a retired element.
#
# The two below are NOT about the retired handler: they lock the packaging
# invariants (`scripts/__init__.py`, the Dockerfile COPY) that
# `aggregate_costs_handler.py` still depends on to resolve
# `from scripts.aggregate_costs import ...` inside the image.


def test_scripts_package_is_importable():
    """Locks ``scripts/__init__.py`` existing — without it, ``from
    scripts.aggregate_costs import aggregate_day`` raises
    ``ModuleNotFoundError`` inside the Lambda image (caught 2026-05-02
    on the post-PR-D validation invoke against v92).

    Implicit namespace packages would work in some environments but the
    explicit marker keeps the import contract visible AND survives any
    aggressive Docker COPY filter that strips empty directories."""
    pkg_init = _HANDLER_PATH.parent.parent / "scripts" / "__init__.py"
    assert pkg_init.exists(), (
        "scripts/__init__.py must exist as the explicit package marker. "
        "Without it the Lambda runtime can hit ModuleNotFoundError on "
        "``from scripts.aggregate_costs import aggregate_day``."
    )


def test_dockerfile_copies_scripts_directory():
    """Locks the Dockerfile ``COPY scripts/`` line. Without it the Lambda
    image is missing the aggregate_costs module entirely and every run
    logs ``[cost_aggregator] aggregation failed: No module named 'scripts'``
    (caught 2026-05-02 on the post-PR-D validation invoke)."""
    dockerfile = _HANDLER_PATH.parent.parent / "Dockerfile"
    src = dockerfile.read_text()
    assert "COPY scripts/" in src, (
        "Dockerfile must include ``COPY scripts/ ${LAMBDA_TASK_ROOT}/scripts/`` "
        "so the cost-aggregator wire-up at lambda/handler.py can resolve "
        "``from scripts.aggregate_costs import aggregate_day`` at runtime."
    )
