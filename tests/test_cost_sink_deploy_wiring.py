"""The cost sink is a DEPLOYMENT fact, and this asserts the deploy sets it.

alpha-engine-config-I7179. `krepis.llm.LLMClient` used to emit a cost record
only when a call site remembered to pass `cost_sink=`, so coverage equalled
the set of authors who thought about it — one process, measured 2026-08-13.
krepis 0.57.0 inverts that: a client with no `cost_sink` argument resolves
one from `KREPIS_COST_SINK_BUCKET` + `KREPIS_COST_SINK_PREFIX`.

Which makes `infrastructure/deploy.sh` the load-bearing artifact, and a
deploy-path defect is never visible in the file that has the bug — so it is
asserted here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEPLOY = _REPO_ROOT / "infrastructure" / "deploy.sh"

#: The literals must be exactly these. A prefix that looks right and points
#: one directory sideways makes every row invisible to the aggregator —
#: which is what `_cost` vs `_cost_raw` already did to the Think Tank once
#: (see the comment in thinktank/run.py).
_BUCKET = "alpha-engine-research"
_PREFIX = "decision_artifacts/_cost_raw"


@pytest.fixture(scope="module")
def deploy() -> str:
    return _DEPLOY.read_text()


class TestTheHelperExists:
    def test_helper_is_defined(self, deploy):
        assert "_apply_cost_sink_env() {" in deploy

    def test_it_sets_both_variables_with_the_exact_literals(self, deploy):
        body = deploy.split("_apply_cost_sink_env() {", 1)[1].split("\n}", 1)[0]
        assert f"--set KREPIS_COST_SINK_PREFIX={_PREFIX}" in body
        assert '--set KREPIS_COST_SINK_BUCKET="$BUCKET"' in body
        assert re.search(rf'^BUCKET="{re.escape(_BUCKET)}"$', deploy, re.M)

    def test_it_merges_rather_than_replaces(self, deploy):
        """`update-function-configuration --environment` REPLACES the whole
        map, and these functions carry provider keys and LangSmith config
        that exist only on the live function and are codified nowhere."""
        body = deploy.split("_apply_cost_sink_env() {", 1)[1].split("\n}", 1)[0]
        assert "krepis.aws merge-lambda-env" in body
        assert "--environment" not in body


class TestEveryPublishedFunctionGetsIt:
    """Applied to every function this script publishes, not a hand-picked
    subset — picking the subset by hand is the same judgement that produced
    the gap."""

    @pytest.mark.parametrize(
        "target",
        [
            '"$FUNCTION_MAIN"',
            '"$FUNCTION_EVAL_JUDGE"',
            '"$FUNCTION_EVAL_ROLLING_MEAN"',
            '"$FUNCTION_RATIONALE_CLUSTERING"',
            '"$fn_name"',  # the shared-image path: submit / poll / process /
            # aggregate_costs / scanner / signals_envelope /
            # openrouter_shadow / perturbation_battery
        ],
    )
    def test_target_is_covered(self, deploy, target):
        assert f"_apply_cost_sink_env {target}" in deploy

    def test_every_publish_version_is_preceded_by_it(self, deploy):
        """ORDERING IS THE WHOLE POINT. A published Lambda version snapshots
        the environment and the `live` alias points at a published version,
        so a merge that runs AFTER publish-version leaves the alias serving
        a version without the variables — while `get-function-configuration`
        on $LATEST shows them set. A deploy-path defect that is invisible in
        every file, and green in every inspection.
        """
        chunks = deploy.split("aws lambda publish-version")
        # chunks[0] is the preamble before the first publish; every
        # subsequent boundary must have an application in the text before it.
        for i, chunk in enumerate(chunks[:-1]):
            assert "_apply_cost_sink_env" in chunk, (
                f"publish-version #{i + 1} is not preceded by a cost-sink "
                f"environment merge — the published version, and therefore "
                f"the live alias, would emit no cost record"
            )


class TestTheSinkStaysAnEnvironmentFact:
    """A later 'helpful' retrofit at a call site is the regression to
    prevent: it restores coverage for exactly as long as it takes to add
    the next call site, which is how this started."""

    def test_the_krepis_floor_is_pinned(self):
        """The resolved version must be AT LEAST 0.57.0 — asserted as a
        version comparison, not as a literal string.

        A literal `"krepis>=0.57.0" in req` passes only while the floor is
        exactly that value, so the next legitimate raise (0.59.7 shipped the
        cost-record contract columns, config-I7393) fails a guard that has
        nothing to say about the raise. A guard keyed on a constant it does
        not own reports every correct change as a defect.

        Accepts either `krepis>=X.Y.Z` (a floor) or `krepis==X.Y.Z` (an
        exact pin, per §139 — first-party/fast-moving deps are pinned, never
        floored). The property under test is "the krepis this image
        resolves is at or above X" — an exact pin establishes that property
        MORE strongly than a floor, so the comparator is not what matters
        here (alpha-engine-config-I7635).
        """
        import re

        req = (_REPO_ROOT / "requirements.txt").read_text()
        m = re.search(r"^krepis(?:>=|==)(\d+)\.(\d+)\.(\d+)", req, re.MULTILINE)
        assert m, "requirements.txt declares no krepis floor or exact pin at all"
        floor = tuple(int(g) for g in m.groups())
        assert floor >= (0, 57, 0), (
            f"krepis floor is {'.'.join(map(str, floor))}; below 0.57.0 there "
            "is no environment-resolved cost sink and no merge-lambda-env "
            "subcommand — the deploy would fail loudly, which is right, but "
            "the floor is what says so"
        )

    def test_krepis_pin_regex_parses_both_floor_and_exact_forms(self):
        """Regression guard for the exact defect I7635 fixed: converting
        the floor to an exact pin must not make this guard report the
        requirement as ABSENT."""
        import re

        pattern = re.compile(r"^krepis(?:>=|==)(\d+)\.(\d+)\.(\d+)", re.MULTILINE)
        floor_match = pattern.search("krepis>=0.59.16")
        exact_match = pattern.search("krepis==0.59.16")
        assert floor_match is not None
        assert exact_match is not None
        assert floor_match.groups() == exact_match.groups() == ("0", "59", "16")
