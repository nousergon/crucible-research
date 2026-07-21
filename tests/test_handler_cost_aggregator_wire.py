"""Locks the cost-aggregator wire-up in lambda/handler.py.

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


def test_handler_invokes_aggregate_day_on_success():
    """Locks the canonical wire-up: ``from scripts.aggregate_costs import
    aggregate_day`` followed by an ``aggregate_day(...)`` call at the
    success-return path. Removing this resurrects the pre-2026-05-02
    manual-CLI step."""
    src = _strip_comments_and_strings(_HANDLER_PATH.read_text())
    assert "from scripts.aggregate_costs import aggregate_day" in src, (
        "lambda/handler.py must import aggregate_day — without it, the "
        "Backtester email's cost section renders empty between Research "
        "and Backtester (manual CLI step required)."
    )
    assert "aggregate_day(" in src


def test_aggregator_call_is_gated_on_email_sent():
    """The aggregator only runs when Research actually succeeded enough
    to send the email (email_sent=True). On dry_run / early failure /
    skipped-by-time the aggregator should NOT fire — there's no captured
    data to aggregate, and a spurious WARN log per skipped invocation
    would be noise."""
    src = _HANDLER_PATH.read_text()
    # Locate the aggregator call block.
    block_start = src.find("aggregate_day(")
    assert block_start != -1, "aggregator call site not found"
    # Walk backwards to the nearest ``if `` / ``try`` to confirm gating.
    preamble = src[max(0, block_start - 500):block_start]
    assert "email_sent" in preamble, (
        "aggregator must be gated on email_sent — without the gate it "
        "fires on dry-runs and skipped invocations and emits noise WARNs."
    )


def test_aggregator_failure_is_non_fatal():
    """Aggregator failure must NOT propagate. Research already succeeded
    by this point and the Backtester gracefully renders an empty cost
    section if the parquet is absent. A failed aggregation should surface
    via a WARN log, not crash the Research Lambda return path.

    Locks the canonical ``except _agg_exc`` name so the wrap doesn't get
    accidentally removed in a refactor — the unique variable name flags
    this as the aggregator-specific catch."""
    src = _HANDLER_PATH.read_text()
    assert "_agg_exc" in src, (
        "aggregator must be wrapped in ``try: ... except Exception as "
        "_agg_exc:`` — without the catch a transient S3 hiccup or pandas "
        "issue would crash a successful Research run."
    )
    # And the catch must surface a WARN, not silently swallow.
    assert "[cost_aggregator] aggregation failed" in src, (
        "aggregator catch must log a WARN with the canonical "
        "``[cost_aggregator] aggregation failed`` prefix so a recurring "
        "failure is greppable in CloudWatch."
    )


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
