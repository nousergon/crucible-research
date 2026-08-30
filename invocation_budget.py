"""Bounded execution of SECONDARY work inside a Lambda invocation (alpha-engine-config-I9102).

A stage that finished its primary deliverable must never be reported DEGRADED
because optional work hung. On 2026-08-28 the weekly
``alpha-engine-research-eval-rolling-mean:live`` invocation emitted its rolling
mean in **1.6 seconds** and then sat for another **298 seconds** inside a
secondary aggregation until the 300s function ceiling killed it. The Step
Functions state raised ``States.Timeout``, the research/predictor branch
fail-opened and the whole weekly run terminated ``FAILED`` — for a stage whose
actual deliverable had succeeded five minutes earlier.

MEASURED, not assumed. From that invocation's own log stream
(``2026/08/28/[379]2195e7f6733c410eae3c42e205dc3e59``):

    22:25:53.652  [eval_rolling_mean_handler] control bands status=OK ...
    22:30:51.912  END / REPORT Duration: 300000.00 ms  Status: timeout

Nothing between them. The next statement after that log line enters
``scripts.build_agent_quality.build_agent_quality`` →
``evals.judge_outcome_ic.open_research_db``, which ``download_file``s the
**356 MB** ``s3://alpha-engine-research/research.db`` snapshot into a 512 MB
function's ``/tmp``. The handler never returned; nothing held it open *after*
it returned. (The issue's stated hypothesis — non-daemon flow-doctor notifier
threads keeping the runtime alive past ``END`` — is disproven by the same
logs: on the healthy 2026-08-21 and 2026-08-22 runs ``END`` lands 1.3–1.5s
after the last handler log line, not 20–35s.)

The class this module fixes is not "that one slow download". It is: **a
handler that hangs the primary deliverable's report on how long optional work
decides to take.** Four secondary aggregations are bolted onto the eval
rolling-mean stage, two of them unbounded network walks, and one more will be
bolted on next quarter. Whatever their individual cost, none of them may be
able to spend the stage's entire budget.

Two mechanisms, both required:

``InvocationBudget``
    Mirrors ``crucible-evaluator/director/budget.py`` — the fleet's declared
    pattern for this (``alpha-engine-config#6904``): "work that cannot fit is
    skipped with a reason rather than started and killed mid-flight, because a
    Lambda killed at the wall writes nothing and logs no cause." Outside Lambda
    (tests, local runs) there is no context, the budget is unbounded, and every
    block runs exactly as before. Second adoption of that pattern in a second
    repo — lift into ``nousergon-lib`` is tracked, not done here, so this P0
    does not depend on a library publish + pin lockstep landing first.

``run_bounded``
    A pre-check alone is NOT a guarantee: it can only decline to *start* a
    block, and the block that killed this run would have passed any plausible
    pre-check (its predecessor took 0.2s the week before, because a separate
    bug made it fail instantly). So each block also runs under a hard
    watchdog — a **daemon** thread joined with a timeout. Daemon is the load-
    bearing word: the Lambda runtime posts the invocation response as soon as
    the handler returns, and a daemon thread neither delays that nor survives
    a container reclaim.

Abandoning a thread is a real cost and is accepted deliberately: an abandoned
block may resume during a LATER invocation and complete an S3 write keyed to
the earlier cycle's date. Every block wrapped here is an idempotent artifact
write, so the worst case is a late-written artifact under its own correct key
— strictly better than the current worst case, which is the entire weekly
pipeline reported FAILED.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# Held back from the last block's quote for the work that is NOT a secondary
# aggregation: the stage-coverage self-assertion, the return-value assembly and
# the cost-sink flush in the handler's ``finally``. Measured across the healthy
# 2026-08-21/22 invocations, that tail runs in 1.3-1.5s; 20s carries it with an
# order of magnitude of headroom without materially shrinking any block.
DEFAULT_RESERVE_S = 20.0

# A block quoted less than this cannot plausibly reach its first network round
# trip, so starting it only guarantees a timeout inside a timeout. Below the
# floor the block is declined with a reason instead.
MIN_VIABLE_BLOCK_S = 5.0


class InvocationBudget:
    """What this invocation can still afford, in seconds.

    ``remaining_ms`` is normally ``context.get_remaining_time_in_millis`` —
    passed as the callable and never as its value, because the answer changes
    as the invocation proceeds and every quote must be taken fresh.
    """

    def __init__(
        self,
        remaining_ms: Callable[[], float] | None = None,
        *,
        reserve_s: float = DEFAULT_RESERVE_S,
    ) -> None:
        self._remaining_ms = remaining_ms
        self._reserve_s = reserve_s

    @classmethod
    def from_context(cls, context: Any, *, reserve_s: float = DEFAULT_RESERVE_S) -> InvocationBudget:
        """Build from a Lambda context; unbounded when there is none."""
        getter = getattr(context, "get_remaining_time_in_millis", None)
        return cls(getter if callable(getter) else None, reserve_s=reserve_s)

    @property
    def bounded(self) -> bool:
        return self._remaining_ms is not None

    def remaining(self) -> float:
        """Seconds left for secondary work, after the reserve. ``inf`` unbounded."""
        if self._remaining_ms is None:
            return float("inf")
        return max(0.0, self._remaining_ms() / 1000.0 - self._reserve_s)

    def quote(self, ceiling: float) -> float:
        """Seconds to allow a block: the smaller of its ceiling and what is left.

        Returns ``ceiling`` unchanged when unbounded, and ``0.0`` when the
        invocation can no longer afford a viable attempt — the caller declines
        rather than starting work it cannot finish.

        The viability floor is checked against what the INVOCATION has left,
        never against the quote. A block whose own declared ceiling is below
        the floor is a deliberately tiny block, not an unaffordable one, and
        declining it would silently disable it.
        """
        if ceiling <= 0:
            raise ValueError(f"ceiling must be > 0, got {ceiling}")
        remaining = self.remaining()
        if remaining < MIN_VIABLE_BLOCK_S:
            return 0.0
        return min(ceiling, remaining)


class BlockTimeout(Exception):
    """A secondary block exceeded the time this invocation could give it."""

    def __init__(self, name: str, seconds: float) -> None:
        super().__init__(
            f"{name} exceeded its {seconds:.0f}s budget and was abandoned so the "
            f"stage could return its primary deliverable"
        )
        self.name = name
        self.seconds = seconds


class NoBudget(Exception):
    """The invocation had too little time left to start a secondary block."""

    def __init__(self, name: str, remaining_s: float) -> None:
        super().__init__(
            f"{name} not started — only {remaining_s:.1f}s of invocation budget "
            f"left, below the {MIN_VIABLE_BLOCK_S:.0f}s viability floor"
        )
        self.name = name
        self.remaining_s = remaining_s


def run_bounded(
    fn: Callable[[], Any],
    *,
    name: str,
    ceiling_s: float,
    budget: InvocationBudget,
) -> Any:
    """Run ``fn`` under a hard wall-clock bound; raise rather than overrun.

    Raises :class:`NoBudget` when the invocation cannot afford a viable attempt,
    :class:`BlockTimeout` when the attempt exceeds its quote, and re-raises
    whatever ``fn`` raised otherwise — so a caller's existing ``except``
    branches keep recording real failures exactly as they do today.
    """
    allowed = budget.quote(ceiling_s)
    if allowed == 0.0:
        raise NoBudget(name, budget.remaining())

    box: dict[str, Any] = {}

    def _target() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 — re-raised on the caller's thread
            box["error"] = exc

    # daemon=True is load-bearing: an abandoned block must not delay the
    # invocation response nor keep the runtime alive. See the module docstring.
    thread = threading.Thread(target=_target, name=f"bounded:{name}", daemon=True)
    thread.start()
    thread.join(timeout=allowed)

    if thread.is_alive():
        logger.error(
            "[invocation_budget] %s exceeded its %.0fs budget — abandoned so the "
            "stage returns its primary deliverable instead of hitting the "
            "function ceiling (alpha-engine-config-I9102)",
            name, allowed,
        )
        raise BlockTimeout(name, allowed)

    if "error" in box:
        raise box["error"]
    return box.get("value")
