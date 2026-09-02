"""``scripts/run_experiment.py`` — the single-experiment R-slot CLI.

WHAT THESE GUARD. Not the scoring arithmetic (that is the live path's, and is
covered by ``tests/test_leaderboard_scoring.py`` / the long-horizon suite). Four
properties of the ENTRY POINT itself:

    an arm resolves                  from the register, by name
    an unknown arm is REFUSED loudly a typo must never fall through to an
                                     empty table, which reads identically to a
                                     real arm that produced nothing
    nothing is written without --write  both production prefixes it touches
    ONE arm grades ALONE             the load-bearing one — a verdict for the
                                     arm under test must not require every
                                     other registered challenger to have built

That last property is the reason the script exists.
``producers.runner.run_challengers`` raises ``ChallengerShadowGapError`` when
ANY always-on buildable challenger fails, so one broken unrelated arm starves
the whole cohort. The test below reproduces exactly that state — a cohort where
only the champion and ONE challenger ever wrote a shadow — and asserts the CLI
still returns a graded verdict for that challenger, while ``run_challengers``
over the same registry would have raised.
"""

from __future__ import annotations

import json
import types

import boto3
import pytest
from moto import mock_aws

from scoring.leaderboard_scoring import COMPARISON_NO_COMMON_COHORT
from scripts.run_experiment import (
    ExperimentError,
    build_parser,
    grade_arm,
    produce_arm,
    render_verdict,
    resolve_arm,
)

_BUCKET = "alpha-engine-research"
_CHAMPION = "scanner_predictor_direct"
_ARM_UNDER_TEST = "no_agent_quant"
_AS_OF = "2026-08-17"


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=_BUCKET)
        yield client


def _put_json(client, key, obj):
    client.put_object(Bucket=_BUCKET, Key=key, Body=json.dumps(obj).encode())


def _session_calendar(n: int, start: str = "2025-02-03") -> list[str]:
    """``n`` synthetic TRADING sessions — business days only, which is what the
    real panel holds. Mirrors ``tests/test_leaderboard_long_horizon_and_
    confidence.py::_session_calendar``."""
    import pandas as pd

    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start=start, periods=n)]


class _Panel:
    """``{date: {ticker: close}}`` handed back as a ``closes_panel_loader``, so
    no test here reaches ArcticDB. Same shape the live loader returns."""

    def __init__(self) -> None:
        self.panel: dict[str, dict[str, float]] = {}

    def put(self, date_str: str, closes: dict) -> _Panel:
        self.panel.setdefault(date_str, {}).update({t: float(c) for t, c in closes.items()})
        return self

    def loader(self):
        return lambda bucket, entry_dates, horizon_days, symbols=None: self.panel


def _matured_panel() -> tuple[_Panel, list[str]]:
    cal = _session_calendar(400)
    panel = _Panel()
    for i, d in enumerate(cal):
        panel.put(d, {"A": 100 + i, "B": 100 + 0.5 * i, "C": 100 - 0.2 * i, "SPY": 100 + 0.3 * i})
    return panel, [cal[250], cal[260]]


def _seed_two_arms_only(client, entries: list[str]) -> None:
    """A cohort in which ONLY the champion and ``_ARM_UNDER_TEST`` ever wrote a
    shadow. Every other registered challenger is absent — the state a
    ``ChallengerShadowGapError`` leaves behind."""
    _put_json(client, "config/producer_champion.json", {"schema_version": 1, "champion": _CHAMPION})
    for d in entries:
        _put_json(
            client,
            f"signals_shadow/{_CHAMPION}/{d}/signals.json",
            {"signals": {t: {"signal": "ENTER", "score": s} for t, s in [("A", 0.9), ("C", 0.1)]}},
        )
        # A membership that DIFFERS from the champion's — a challenger
        # resolving the champion's exact set is a vacuous comparison
        # (champion-challenger-policy.md §4) and the board alerts on it.
        _put_json(
            client,
            f"signals_shadow/{_ARM_UNDER_TEST}/{d}/signals.json",
            {"signals": {t: {"signal": "ENTER", "score": s} for t, s in [("A", 0.9), ("B", 0.5)]}},
        )


# ── The arm resolves ─────────────────────────────────────────────────────────


class TestArmResolution:
    def test_a_registered_arm_resolves_to_its_spec(self):
        spec = resolve_arm(_ARM_UNDER_TEST)
        assert spec.name == _ARM_UNDER_TEST
        assert spec.kind == "challenger"

    def test_an_unknown_arm_is_refused_loudly_and_names_the_alternatives(self):
        """A typo must RAISE, not return an empty verdict.

        "no rows for that arm" and "a real arm that produced nothing" render
        identically, which is the conflation champion-challenger-policy.md §7.2
        forbids — so the refusal happens before any S3 read."""
        with pytest.raises(ExperimentError) as exc:
            resolve_arm("no_agent_qaunt")
        msg = str(exc.value)
        assert "no_agent_qaunt" in msg
        assert _ARM_UNDER_TEST in msg, "the refusal must name the registered arms"

    def test_an_arm_absent_from_the_board_raises_rather_than_printing_blank(self, s3):
        panel, entries = _matured_panel()
        _seed_two_arms_only(s3, entries)
        with pytest.raises(ExperimentError, match="no row on the producer leaderboard"):
            grade_arm(
                s3, _BUCKET, "never_registered_arm", _AS_OF,
                top_n=2, write=False, closes_panel_loader=panel.loader(),
            )


# ── Nothing is written without --write ───────────────────────────────────────


class TestWritePosture:
    def test_write_defaults_to_false_on_the_parser(self):
        args = build_parser().parse_args(["--arm", _ARM_UNDER_TEST, "--date", _AS_OF])
        assert args.write is False, "the default posture is a DRY RUN"
        assert args.produce is False

    def test_grading_without_write_leaves_the_leaderboard_prefix_untouched(self, s3):
        panel, entries = _matured_panel()
        _seed_two_arms_only(s3, entries)
        verdict = grade_arm(
            s3, _BUCKET, _ARM_UNDER_TEST, _AS_OF,
            top_n=2, write=False, closes_panel_loader=panel.loader(),
        )
        assert verdict["board_key"] is None
        listing = s3.list_objects_v2(Bucket=_BUCKET, Prefix="research/producer_leaderboard/")
        assert not listing.get("Contents"), (
            "a dry run must not write research/producer_leaderboard/ — that "
            "artifact is read by crucible-backtester's promotion engine"
        )

    def test_write_persists_the_leaderboard(self, s3):
        panel, entries = _matured_panel()
        _seed_two_arms_only(s3, entries)
        verdict = grade_arm(
            s3, _BUCKET, _ARM_UNDER_TEST, _AS_OF,
            top_n=2, write=True, closes_panel_loader=panel.loader(),
        )
        assert verdict["board_key"] == f"research/producer_leaderboard/{_AS_OF}.json"
        body = json.loads(
            s3.get_object(Bucket=_BUCKET, Key=verdict["board_key"])["Body"].read()
        )
        assert body["leaderboard_id"] == "producer"

    def test_produce_without_write_does_not_put_the_shadow(self, s3):
        """The produce stage's own dry-run posture, checked separately: a
        verdict prefix and a shadow prefix are two different production
        surfaces and one flag governs both."""
        built = {}

        def _build(run_date, archive_manager, **ctx):
            built["called"] = run_date
            return {"signals": {"A": {"signal": "ENTER", "score": 0.9}}}

        spec = types.SimpleNamespace(name=_ARM_UNDER_TEST, build=_build)
        manager = types.SimpleNamespace(bucket=_BUCKET, s3=s3)
        rec = produce_arm(spec, manager, _AS_OF, write=False)

        assert built["called"] == _AS_OF, "the build still RUNS in a dry run"
        assert rec["written"] is False
        assert rec["n_enter"] == 1
        assert not s3.list_objects_v2(Bucket=_BUCKET, Prefix="signals_shadow/").get("Contents")

    def test_producing_an_externally_built_arm_is_refused(self, s3):
        """``build=None`` means the arm writes its own shadow on its own
        cadence (e.g. ``thinktank_coverage``). Calling it would raise
        TypeError deep inside; refuse it at the boundary with the reason."""
        spec = types.SimpleNamespace(name="thinktank_coverage", build=None)
        with pytest.raises(ExperimentError, match="build=None"):
            produce_arm(spec, types.SimpleNamespace(bucket=_BUCKET, s3=s3), _AS_OF, write=True)


# ── THE LOAD-BEARING ONE: one arm grades alone ───────────────────────────────


class TestOneArmGradesWithoutTheCohort:
    def test_a_verdict_is_produced_for_one_arm_with_the_cohort_incomplete(self, s3):
        """A graded verdict for ``_ARM_UNDER_TEST`` on a cohort where every
        OTHER registered challenger wrote no shadow at all.

        This is the state ``ChallengerShadowGapError`` leaves the bucket in.
        The weekly path cannot proceed through it; this entry point must."""
        panel, entries = _matured_panel()
        _seed_two_arms_only(s3, entries)

        verdict = grade_arm(
            s3, _BUCKET, _ARM_UNDER_TEST, _AS_OF,
            top_n=2, write=False, closes_panel_loader=panel.loader(),
        )

        assert verdict["champion"] == _CHAMPION
        assert verdict["board_status"] == "ok"
        by_h = {h["horizon_days"]: h for h in verdict["horizons"]}
        assert 21 in by_h, "the primary horizon must carry a row for the arm"
        row = by_h[21]["row"]
        assert row["name"] == _ARM_UNDER_TEST
        assert row["n_dates_scored"] == len(entries), (
            "the arm scored on every cohort date it wrote a shadow for, with "
            "no other challenger present"
        )
        # The arm's OWN history is intact — champion-challenger-policy.md §3.
        assert row["topn_alpha_vs_benchmark"] is not None
        assert row["realized_rank_ic"] is not None

        # And the paired champion comparison — the statistic the board cannot
        # give here (see TestBoardIntersectionIsAllArms) — IS produced, over
        # the dates this arm and the champion share.
        pw = verdict["pairwise_vs_champion"][21]
        assert pw["n_dates"] == len(entries)
        assert pw["metric"] is not None
        assert pw["metric"]["mean"] is not None
        assert [pw["first"], pw["last"]] == [entries[0], entries[-1]]


class TestBoardIntersectionIsAllArms:
    """MEASURED 2026-09-01, and the reason ``pairwise_vs_champion`` exists.

    ``leaderboard_scoring.apply_cohort_intersection`` narrows every row's
    ``topn_alpha_vs_champion`` to ``strict_cohort_intersection(ALL arms)``.
    Correct for a ranked table under champion-challenger-policy.md §4 — and it
    means ONE registered arm with no shadow at all empties the intersection and
    nulls the paired figure for every other arm, including the one under test.

    That is the same "an unrelated arm starves this verdict" class
    ``ChallengerShadowGapError`` creates at the produce stage, reappearing at
    the grading stage. This test pins it so the CLI's second pass cannot
    quietly become unnecessary work nobody notices, or be deleted as such."""

    def test_one_armless_registered_arm_nulls_the_boards_paired_figure(self, s3):
        panel, entries = _matured_panel()
        _seed_two_arms_only(s3, entries)
        verdict = grade_arm(
            s3, _BUCKET, _ARM_UNDER_TEST, _AS_OF,
            top_n=2, write=False, closes_panel_loader=panel.loader(),
        )
        row = {h["horizon_days"]: h["row"] for h in verdict["horizons"]}[21]
        assert row["topn_alpha_vs_champion"] is None
        assert row["comparison_status"] == COMPARISON_NO_COMMON_COHORT
        block = {h["horizon_days"]: h for h in verdict["horizons"]}[21]
        assert block["arms_with_no_cohort"], (
            "the arms that emptied the intersection must be NAMED on the "
            "verdict — 'no_common_cohort' with no culprit is undiagnosable"
        )
        assert _ARM_UNDER_TEST not in block["arms_with_no_cohort"], (
            "the arm under test DID score at this horizon"
        )
        assert "single_agent_quant" in block["arms_with_no_cohort"]

    def test_the_rendering_names_the_arms_that_emptied_the_intersection(self, s3):
        panel, entries = _matured_panel()
        _seed_two_arms_only(s3, entries)
        out = render_verdict(
            grade_arm(
                s3, _BUCKET, _ARM_UNDER_TEST, _AS_OF,
                top_n=2, write=False, closes_panel_loader=panel.loader(),
            )
        )
        assert "pairwise vs champ" in out
        assert COMPARISON_NO_COMMON_COHORT in out
        assert "single_agent_quant" in out, "the culprit arm is named in the rendering"

    def test_the_cohort_this_grades_would_fail_the_weekly_completeness_gate(self, s3):
        """The counterpart assertion, so the test above cannot silently stop
        being about anything. Over the same registry, the weekly builder's gate
        RAISES — every buildable challenger is expected to emit, and in this
        fixture they do not."""
        from producers.runner import ChallengerShadowGapError, run_challengers

        class _FailingManager:
            bucket = _BUCKET

            def write_shadow_signals_json(self, *a, **k):  # pragma: no cover - not reached
                raise AssertionError("no build should get this far in this fixture")

        def _boom(*a, **k):
            raise RuntimeError("one unrelated arm is broken")

        import producers.runner as runner_mod

        specs = [
            types.SimpleNamespace(name=n, build=_boom)
            for n in (_ARM_UNDER_TEST, "single_agent_quant")
        ]
        orig = runner_mod.buildable_challenger_producers
        runner_mod.buildable_challenger_producers = lambda: specs
        try:
            with pytest.raises(ChallengerShadowGapError):
                run_challengers(_FailingManager(), _AS_OF)
        finally:
            runner_mod.buildable_challenger_producers = orig


# ── The verdict a human reads ────────────────────────────────────────────────


class TestVerdictRendering:
    def test_the_rendering_surfaces_the_scorer_s_own_evidence_floor(self, s3):
        """No new significance rule: the floor shown is the slot's registered
        ``min_dates_for_inference`` and the status shown is the scorer's own
        per-row ``confidence``."""
        from scoring.leaderboard_scoring import slot_spec

        panel, entries = _matured_panel()
        _seed_two_arms_only(s3, entries)
        verdict = grade_arm(
            s3, _BUCKET, _ARM_UNDER_TEST, _AS_OF,
            top_n=2, write=False, closes_panel_loader=panel.loader(),
        )
        floor = slot_spec("producer").min_dates_for_inference
        assert verdict["min_dates_for_inference"] == floor

        out = render_verdict(verdict)
        assert _ARM_UNDER_TEST in out
        assert _CHAMPION in out
        assert f"floor = {floor}" in out
        assert "alpha vs champion" in out
        assert "alpha vs benchmark" in out
        assert "n_dates_scored" in out
        # Two cohort dates against a floor of MIN_DATES_FOR_INFERENCE: the
        # rendering must SAY the sample is below the floor, not print the mean
        # as though it were a result.
        assert "BELOW the evidence floor" in out
        for h in (21, 126, 252):
            assert f"── {h} sessions" in out, "every registered horizon is rendered"
