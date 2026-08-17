"""A failure path must not delete the evidence its own message points at.

**The 2026-08-15 weekly-SF failure (alpha-engine-config-I7396 → I7442).** The
`PredictorBacktest` stage (a sibling launcher, `crucible-backtester`) died and
printed:

    ERROR: SSM step 'predictor-backtest' terminal status=Failed …
      — full remote log: s3://alpha-engine-research/tmp/spot_predictor-backtest/
        20260815T123311Z-i-08a4371deec28ef07/ssm-output/

Four lines later, the same exit path printed *"Instance terminated; S3 staging
cleaned."* The prefix the error named was **empty** by the time anyone read
it, and so was its parent. `run_ssm`'s `--output-key-prefix` points SSM's
`OutputS3KeyPrefix` at `${S3_STAGING_PREFIX}/ssm-output` — the SSM agent's own
upload of the FULL remote stdout/stderr, the only copy not subject to
`GetCommandInvocation`'s 24KB inline cap. Every spot launcher's EXIT trap ran
`aws s3 rm "$S3_STAGING" --recursive` while handling a failure, which destroys
that upload — this repo's `spot_research_weekly.sh` carried the same unguarded
delete even though it was not the stage that died on 2026-08-15, because the
defect is a CLASS, not one call site.

**What changed at I7442.** Teardown now runs through
`krepis.spot_evidence teardown` (krepis branch
`fix/i7442-resource-kill-classifier`), which copies the staging prefix to
`_spot_evidence/<slug>/<date>/<run-id>/` FIRST and deletes staging only if
that copy succeeded — so the ordering is a property of that module's call
graph rather than of a branch reimplemented per launcher, and it holds for
every launcher family at once. Reference implementation:
`crucible-backtester#675`'s `infrastructure/_spot_common.sh::
spot_common_teardown_staging`. This repo has no shared `_spot_common.sh`, so
the equivalent lives inline in `spot_research_weekly.sh` as `_teardown_staging`.

These tests therefore pin the CLASS property: no launcher in this repo may
carry an unguarded staging delete, and the teardown must degrade to retention
rather than to deletion.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

INFRA = Path(__file__).resolve().parent.parent / "infrastructure"
LAUNCHER = INFRA / "spot_research_weekly.sh"


@pytest.fixture(scope="module")
def body() -> str:
    assert LAUNCHER.is_file(), f"{LAUNCHER} missing"
    return LAUNCHER.read_text(encoding="utf-8")


def _fn(body: str, name: str) -> str:
    start = body.index(f"{name}() {{")
    end = body.index("\n}\n", start)
    return body[start:end]


class TestNoLauncherDeletesItsOwnStaging:
    """The class guard. Fixing one call site of a systemic defect is not a fix."""

    def test_no_shell_script_in_infrastructure_removes_S3_STAGING(self):
        offenders = []
        for path in sorted(INFRA.glob("*.sh")):
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if "aws s3 rm" in stripped and "S3_STAGING" in stripped:
                    offenders.append(f"{path.name}:{n}: {stripped}")
        assert not offenders, (
            "an unguarded staging delete is back — this is the "
            "alpha-engine-config-I7442 defect, and it destroys the only "
            "un-truncated copy of a failure's output:\n  "
            + "\n  ".join(offenders)
        )

    def test_every_launcher_that_stages_also_tears_down_through_the_chokepoint(self):
        for path in sorted(INFRA.glob("spot_*.sh")):
            text = path.read_text(encoding="utf-8")
            if "S3_STAGING=" not in text and "S3_STAGING_PREFIX=" not in text:
                continue
            assert "krepis.spot_evidence teardown" in text, (
                f"{path.name} provisions its own staging prefix but never "
                "hands it to the teardown chokepoint"
            )


class TestTeardownGoesThroughTheChokepoint:
    def test_cleanup_calls_the_shared_teardown(self, body):
        cleanup = _fn(body, "cleanup")
        assert '_teardown_staging "$exit_code"' in cleanup

    def test_on_exit_passes_the_captured_exit_code_to_cleanup(self, body):
        on_exit = _fn(body, "on_exit")
        assert 'cleanup "$rc"' in on_exit, (
            "the workload's exit status is what decides whether evidence is "
            "preserved; without it every run looks like a success"
        )

    def test_the_teardown_helper_invokes_krepis(self, body):
        fn = _fn(body, "_teardown_staging")
        assert "krepis.spot_evidence teardown" in fn
        assert "--exit-code" in fn
        assert "--staging" in fn and "--slug" in fn

    def test_an_unavailable_chokepoint_degrades_to_RETENTION_not_deletion(self, body):
        """The merge-order safety property, and the fail-safe direction.

        A box whose krepis pin predates `spot_evidence` must keep the
        evidence, never fall back to the delete this whole change exists to
        remove.
        """
        fn = _fn(body, "_teardown_staging")
        assert "RETAINED" in fn
        code = "\n".join(
            line for line in fn.splitlines() if not line.strip().startswith("#")
        )
        assert "aws s3 rm" not in code

    def test_the_teardown_never_aborts_the_exit_path(self, body):
        fn = _fn(body, "_teardown_staging")
        assert fn.rstrip().endswith("return 0"), (
            "a janitor that can change the trap's exit status masks the "
            "workload's own failure"
        )


class TestTheOperatorIsToldWhereToLook:
    def test_the_resource_limit_reaches_the_dispatcher(self, body):
        """sf-pipeline-policy §3 obligation 3 — name the limit, not only the
        classification. The dispatcher knows its own executionTimeout; only
        the launcher knows which instance types the stage was allowed."""
        assert re.search(r"--resource-limit\s+\"instance-types=", body)
