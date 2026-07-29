"""thinktank_box_runner.py — run the daily Think Tank pass on a spot box.

WHY THIS FILE EXISTS AT ALL (config-I5208 / nous-ergon-ops-I162): the daily
Think Tank ran as a Lambda and hit the 900s ceiling every day from 2026-07-17,
dying mid-loop before any of its terminal writes. ARCHITECTURE §47 puts a
long-running agent loop on owned compute, so the run moves to a self-
terminating EC2 spot box. This module is the on-box entry point; it invokes
the SAME ``lambda/thinktank_handler.py::handler`` orchestration the Lambda
drives, so the run's contract (fail-loud, manifest shape, terminal-write
ordering) is preserved by construction.

THE DEADLINE IS THE WHOLE POINT — DO NOT CALL ``handler(event, None)``.
``thinktank_handler.handler`` derives its deadline exclusively from a Lambda
``context`` (``handler.py`` — ``seconds_remaining`` stays ``None`` when
``context is None``), and ``thinktank.run._out_of_time`` treats ``None`` as
"no deadline, never truncate". So the obvious box-side call —
``handler(event, None)`` — silently DISARMS the crucible-research#516 deadline
guard. The run would then execute unbounded until something external kills it:
the SSM ``execution_timeout``, or a spot reclaim. Either one lands mid-LLM-loop
and discards every terminal write — i.e. it reproduces the exact 2026-07-17
failure at a longer timescale, on hardware chosen to fix it.

This module therefore supplies a REAL clock from two independent sources:

1. **Run budget** (``THINKTANK_RUN_BUDGET_SECONDS``). A wall-clock deadline set
   below the dispatcher's SSM ``execution_timeout`` so the run always reaches
   its terminal writes before SSM guillotines the command. The margin is the
   reserve ``thinktank.run`` already enforces (``_TERMINAL_WRITE_RESERVE_S``,
   pinned >= 120s by its own test).

2. **Spot interruption notice** (IMDSv2 ``/latest/meta-data/spot/instance-action``).
   EC2 gives ~120s of notice. A background poller collapses the reported
   remaining time to zero the moment a notice appears, which drives
   ``_out_of_time`` true on the next check — the run stops taking new work and
   proceeds to persist. This is what makes spot SAFE for this workload rather
   than merely bigger: without it, a reclaim is indistinguishable from the
   Lambda timeout it replaced.

Fail-loud: the handler raises on any failure and this module does not catch —
the non-zero exit propagates through the bootstrap to the SSM command status,
which is the dispatcher's and sf-watch's only honest failure signal. A caught
exception here would make a dead run report success (the no-silent-fails rule).
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
import urllib.error
import urllib.request

# Repo root on sys.path so ``lambda/thinktank_handler.py`` and ``thinktank.*``
# resolve from a bare clone, mirroring the Lambda's task layout.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "lambda"))

logger = logging.getLogger("thinktank_box_runner")

# S105 is a false positive here: flake8-bandit flags the NAME (it contains
# "TOKEN"), but the value is the IMDSv2 session-token ENDPOINT, not a secret.
# The token itself is fetched at runtime and never persisted.
_IMDS_TOKEN_URL = "http://169.254.169.254/latest/api/token"  # noqa: S105
_IMDS_ACTION_URL = "http://169.254.169.254/latest/meta-data/spot/instance-action"
_IMDS_POLL_SECONDS = 5.0
_IMDS_TIMEOUT_SECONDS = 2.0

# Default budget for a daily pass. Must stay strictly BELOW the dispatcher's
# SSM execution_timeout_seconds; the dispatcher owns that coupling and passes
# this value in explicitly.
#
# DERIVED FROM MEASURED RUNS, not guessed. The daily run is INCREMENTAL — it
# does not rebuild coverage. Two live manifests bracket the #464 step change:
#
#   2026-07-16 (pre-#464, last complete run): 443s total for 8 theses
#     (5 new + 3 event-driven) AND a 70-name sweep. ~55s/thesis.
#   2026-07-29 (post-#464, deadline-truncated): 801s for 5 theses and NO
#     sweep. ~160s/thesis — the pillar/moat call roughly tripled per-thesis
#     wall-clock, which is what pushed the run past the 900s ceiling.
#
# The sweep itself is cheap and was never the problem: it chunks at
# sweep_chunk_size=25, so 135 covered names is 6 LLM calls (3 calls / $0.0007
# on 07-16). Steady-state daily work is therefore ~8 theses at ~160s plus the
# sweep chunks -- call it ~25 min, i.e. a shade over 2x the Lambda ceiling.
#
# 90 min gives ~3.5x headroom over that measurement, which covers the two
# known growth vectors: intake continues at daily_new_names=5 until coverage
# reaches rank_ceiling=150 (135 today, so ~3 more days), and staleness refresh
# (stale_after_days=30) begins firing in early August for the theses written
# when coverage started. Re-derive from manifests rather than raising blind
# if runs start truncating again.
_DEFAULT_BUDGET_SECONDS = 5400  # 90 min


class SpotInterruptionWatcher:
    """Polls IMDSv2 for a spot interruption notice and latches on the first one.

    Latch-only, never un-latches: EC2 does not retract a notice, and a
    transient IMDS failure after a real notice must not resurrect the run's
    remaining time. IMDS errors are logged and treated as "no notice yet" —
    the run budget is the independent second clock, so a broken IMDS degrades
    to budget-only rather than to no deadline at all (ARCHITECTURE §132:
    unverified is not false).
    """

    def __init__(self) -> None:
        self._interrupted = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def interrupted(self) -> bool:
        return self._interrupted.is_set()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._poll_loop, name="spot-interruption-watcher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _poll_loop(self) -> None:
        while not self._stop.is_set() and not self._interrupted.is_set():
            if self._notice_present():
                logger.error(
                    "[thinktank_box_runner] SPOT INTERRUPTION NOTICE — collapsing "
                    "remaining time to 0 so the run stops taking new work and "
                    "persists its terminal writes inside the ~120s window"
                )
                self._interrupted.set()
                return
            self._stop.wait(_IMDS_POLL_SECONDS)

    def _notice_present(self) -> bool:
        # S310 (audit URL open for permitted schemes) is suppressed on each
        # call below: both URLs are module-level http:// literals pointing at
        # the EC2 link-local IMDS address. No caller-supplied input reaches
        # these requests, so the scheme-confusion class the rule guards
        # against cannot occur here.
        try:
            token_req = urllib.request.Request(  # noqa: S310
                _IMDS_TOKEN_URL,
                method="PUT",
                headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
            )
            with urllib.request.urlopen(token_req, timeout=_IMDS_TIMEOUT_SECONDS) as resp:  # noqa: S310
                token = resp.read().decode()
            action_req = urllib.request.Request(  # noqa: S310
                _IMDS_ACTION_URL, headers={"X-aws-ec2-metadata-token": token}
            )
            with urllib.request.urlopen(action_req, timeout=_IMDS_TIMEOUT_SECONDS) as resp:  # noqa: S310
                return resp.status == 200
        except urllib.error.HTTPError as exc:
            # 404 is the steady state: no interruption scheduled.
            if exc.code == 404:
                return False
            logger.warning("[thinktank_box_runner] IMDS probe HTTP %s", exc.code)
            return False
        except Exception as exc:  # noqa: BLE001 - IMDS is best-effort; budget is the floor
            # Swallowed deliberately (no-silent-fails carve-out): the failure
            # mode is "spot notice not seen", the primary deliverable survives
            # via the independent run-budget clock, and it is recorded here at
            # WARNING on every poll so the degradation is visible in the box log.
            logger.warning("[thinktank_box_runner] IMDS probe failed: %s", exc)
            return False


class BoxContext:
    """Minimal stand-in for the Lambda context object.

    ``thinktank_handler.handler`` only ever calls
    ``get_remaining_time_in_millis()`` (it duck-types via ``hasattr``), so this
    is the entire surface needed to arm the deadline guard on a box.
    """

    def __init__(self, budget_seconds: float, watcher: SpotInterruptionWatcher) -> None:
        self._deadline = time.monotonic() + budget_seconds
        self._watcher = watcher

    def get_remaining_time_in_millis(self) -> float:
        if self._watcher.interrupted:
            return 0.0
        return max(0.0, (self._deadline - time.monotonic()) * 1000.0)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    budget = float(os.environ.get("THINKTANK_RUN_BUDGET_SECONDS", _DEFAULT_BUDGET_SECONDS))
    if budget <= 0:
        raise ValueError(
            f"THINKTANK_RUN_BUDGET_SECONDS must be > 0, got {budget!r} — an "
            "absent/zero budget disarms the deadline guard, which is the "
            "failure this migration exists to fix"
        )

    watcher = SpotInterruptionWatcher()
    watcher.start()
    context = BoxContext(budget, watcher)

    from thinktank_handler import handler  # noqa: PLC0415 - after sys.path setup

    logger.info(
        "[thinktank_box_runner] starting daily run: budget=%.0fs, terminal-write reserve enforced by thinktank.run",
        budget,
    )
    try:
        result = handler({}, context)
    finally:
        watcher.stop()
    logger.info("[thinktank_box_runner] run complete: status=%s", result.get("status"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
