"""The Think Tank's availability-retry budget must outlast a router restart.

`alpha-engine-config-I8351`. The LiteLLM router is restarted by ordinary
merges — `router-config-deploy.yml` fires on any `alpha-engine-config`
registry merge, at whatever hour that merge lands — and it takes about fifty
seconds to come back (config generation, git pull, prisma migrate). During
that window nginx answers `502 Bad Gateway`.

The OpenAI SDK's backoff is 0.5s, 1s, 2s, 4s, then 8s per attempt, so
`max_retries=3` spans **3.2 seconds**. Measured 2026-08-25, two daily runs
died that way seventeen minutes apart, each after real work:

    deploy 17:02:35 -> router stop 17:04:38 -> 502 -> ABORTED, 4 theses
    deploy 17:18:03 -> router stop 17:21:03 -> 502 -> ABORTED, 6 theses

Both gave up with roughly 46 seconds of the outage still to run. A daily run
is ~50 minutes, is NOT resumable, is spot-priced, and withholds
`challenger_selection/latest.json` on abort — so waiting a minute costs a
minute, and not waiting costs all of that.

A SOURCE-TEXT invariant rather than a construction test, deliberately:
building a `ThinktankClient` reaches `krepis`' DLP pre-scan, which shells out
to `gitleaks` and fails on a developer laptop that has the binary but not the
repo's scanner config. A guard that only runs in CI is a guard that stops
being read. Same shape as `test_thinktank_arm_registration`'s architectural
locks.
"""

from __future__ import annotations

import re
from pathlib import Path

CLIENT_PATH = Path(__file__).parent.parent / "thinktank" / "client.py"

#: Seconds the SDK's schedule spends across N retries: 0.5, 1, 2, 4, then 8
#: capped. Kept as an explicit table rather than a formula so a change to the
#: SDK's backoff shows up as a disagreement here rather than silently.
_SDK_BACKOFF_SECONDS = [0.5, 1.0, 2.0, 4.0] + [8.0] * 20

#: The measured time the router is unreachable across a restart, 2026-08-25.
MEASURED_RESTART_SECONDS = 50.0


def _retry_span_seconds(max_retries: int) -> float:
    return sum(_SDK_BACKOFF_SECONDS[:max_retries])


def test_the_source_declares_a_max_retries_for_the_llm_client():
    src = CLIENT_PATH.read_text()
    assert "max_retries=" in src, (
        "thinktank/client.py no longer passes max_retries to LLMClient — the "
        "SDK default would then decide how long this run survives a router "
        "restart, which is exactly the thing I8351 made explicit"
    )


def test_the_retry_budget_outlasts_a_measured_router_restart():
    src = CLIENT_PATH.read_text()
    m = re.search(r"^\s*max_retries=(\d+),\s*$", src, re.MULTILINE)
    assert m, "could not find the LLMClient max_retries literal"
    span = _retry_span_seconds(int(m.group(1)))
    assert span >= MEASURED_RESTART_SECONDS, (
        f"max_retries={m.group(1)} spans only {span:.1f}s of backoff, but the "
        f"LiteLLM router is unreachable for ~{MEASURED_RESTART_SECONDS:.0f}s "
        f"across a restart, and it is restarted by ordinary registry merges. "
        f"A ~50-minute non-resumable run must not give up inside that window "
        f"(alpha-engine-config-I8351)."
    )


def test_the_reason_is_written_down_where_the_number_lives():
    """The number is only defensible with what it is bounding beside it.

    Every previous instance of this class in the fleet was a literal whose
    rationale lived in a PR body nobody reads at the call site.
    """
    src = CLIENT_PATH.read_text()
    assert "I8351" in src, "name the issue that establishes the restart duration"
    assert "restart" in src.lower(), (
        "say WHAT is being waited out — a retry count with no named dependency "
        "reads as a rate-limit budget, which is how it was sized before"
    )


def test_the_schema_corrective_budget_is_not_conflated_with_this_one():
    """`_STRUCTURED_ATTEMPTS` bounds body/schema retries, not availability.

    They are different failures with different right answers, and merging them
    would make a malformed-JSON response cost a minute of backoff.
    """
    src = CLIENT_PATH.read_text()
    assert "_STRUCTURED_ATTEMPTS = 3" in src, (
        "the schema-corrective budget changed — check it was not conflated "
        "with the availability budget this file's max_retries owns"
    )
