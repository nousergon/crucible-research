"""Wiring the weekly ledger into the scanner run (alpha-engine-config-I8264).

`crucible-research-PR731` shipped `scoring/weekly_ledger` — the module — and
nothing wrote it. These are the guards for the wiring: which week a run
records, what it prices that week against, what it refuses to do twice, and
what happens when a week closes and cannot be measured.

RED on origin/main at c38c2350: `scoring.weekly_ledger.record_completed_week`
does not exist, and `lambda/scanner_handler.py` never mentions the ledger.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring.universe_membership import SLOT_ARMS  # noqa: E402
from scoring.weekly_ledger import (  # noqa: E402
    LEDGER_KEY,
    read_ledger,
    record_completed_week,
    resolve_boundary,
)

WEEK_START = "2026-08-13"
WEEK_END = "2026-08-20"
PRIOR_GENERATED_AT = "2026-08-13T09:00:00+00:00"

_ARMS = list(SLOT_ARMS)
CHAMPION = _ARMS[0]
POPULATION = [f"T{i:03d}" for i in range(12)]


# ── Fakes ────────────────────────────────────────────────────────────────────


class FakeS3:
    """An S3 with real keys, and a record of everything it was asked for.

    It counts LIST calls because "record one week" and "walk the history" are
    different programs, and the difference is not visible from the ledger
    afterwards — a backfilled week and a forward-accumulated one look identical
    once written, which is exactly why the prohibition has to be tested at the
    call, not at the artifact (alpha-engine-config-I8255 / I8262).
    """

    def __init__(self, objects: dict[str, bytes] | None = None):
        self.objects = dict(objects or {})
        self.gets: list[str] = []
        self.puts: list[str] = []
        self.lists = 0

    def get_object(self, Bucket, Key):  # noqa: N803
        self.gets.append(Key)
        if Key not in self.objects:
            raise RuntimeError(f"NoSuchKey: {Key}")
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, Bucket, Key, Body, ContentType=None):  # noqa: N803
        self.puts.append(Key)
        self.objects[Key] = Body

    def list_objects_v2(self, **kwargs):
        self.lists += 1
        return {"Contents": []}

    def get_paginator(self, *a, **k):
        self.lists += 1
        raise AssertionError("the ledger wiring must never page a prefix")


def _membership(
    *,
    run_date: str,
    generated_at: str,
    cut_effective_date: str,
    cuts: dict[str, list[str]],
    turnover: dict | None,
) -> dict:
    return {
        "run_date": run_date,
        "generated_at": generated_at,
        "cut_effective_date": cut_effective_date,
        "cuts": {name: {"tickers": t} for name, t in cuts.items()},
        "ranks": {t: {"rank": i} for i, t in enumerate(POPULATION)},
        "turnover": turnover,
    }


def _prior_cuts() -> dict[str, list[str]]:
    """Four of the five arms hold names; the fifth holds none.

    That asymmetry is the live state as of 2026-08-24 (`attractiveness_hard3_
    top_60` has never emitted a cut) and it is the case §3 governs: an arm that
    produced no output is recorded as a MISS, never omitted.
    """
    return {
        _ARMS[0]: ["T000", "T001", "T002"],
        _ARMS[1]: ["T003", "T004"],
        _ARMS[2]: ["T000", "T005"],
        _ARMS[3]: ["T006", "T007"],
    }


def _turnover_block(*, retention: float = 75.0, retained: int = 3) -> dict:
    return {
        "prior_run_date": "2026-08-06",
        "prior_generated_at": "2026-08-06T09:00:00+00:00",
        "per_cut": {
            arm: {"size": 3, "retained": retained, "added": 1, "dropped": 1,
                  "retention_pct": retention}
            for arm in _ARMS
        },
    }


def _prior_doc(**over) -> dict:
    base = {
        "run_date": WEEK_START,
        "generated_at": PRIOR_GENERATED_AT,
        "cut_effective_date": WEEK_START,
        "cuts": _prior_cuts(),
        "turnover": _turnover_block(),
    }
    base.update(over)
    return _membership(**base)


def _current_doc(*, cut_effective_date: str = WEEK_END, prior_generated_at: str = PRIOR_GENERATED_AT) -> dict:
    return _membership(
        run_date=WEEK_END,
        generated_at="2026-08-20T09:00:00+00:00",
        cut_effective_date=cut_effective_date,
        cuts={arm: ["T008", "T009"] for arm in _ARMS},
        turnover={
            "prior_run_date": WEEK_START,
            "prior_generated_at": prior_generated_at,
            "per_cut": {arm: {"retention_pct": 50.0, "retained": 1} for arm in _ARMS},
        },
    )


def _runs_key(run_date: str, generated_at: str) -> str:
    from scoring.universe_membership import run_stamp

    return f"universe_membership/{run_date}/runs/{run_stamp(generated_at)}.json"


def _s3_with_prior(prior: dict | None = None, *, also_dated: dict | None = None) -> FakeS3:
    prior = prior if prior is not None else _prior_doc()
    objects = {
        _runs_key(prior["run_date"], prior["generated_at"]): json.dumps(prior).encode(),
    }
    if also_dated is not None:
        objects[f"universe_membership/{prior['run_date']}/membership.json"] = json.dumps(also_dated).encode()
    return FakeS3(objects)


def _loader(prices: dict[str, dict[str, float]] | None = None):
    """A stand-in for ``load_universe_ohlcv``: ``{date: {ticker: close}}`` in,
    ``{ticker: DataFrame}`` out, in the TITLE-case shape ArcticDB stores."""
    import pandas as pd

    default = {
        WEEK_START: dict.fromkeys([*POPULATION, "SPY"], 100.0),
        WEEK_END: dict.fromkeys([*POPULATION, "SPY"], 110.0),
    }
    panel = prices if prices is not None else default

    def load(bucket, *, symbols=None, lookback_days=None, end=None, columns=None):
        wanted = set(symbols or [])
        frames: dict[str, pd.DataFrame] = {}
        for date, row in panel.items():
            for ticker, close in row.items():
                if wanted and ticker not in wanted:
                    continue
                frames.setdefault(ticker, {})[date] = close
        return {
            t: pd.DataFrame({"Close": list(v.values())}, index=pd.to_datetime(list(v.keys())))
            for t, v in frames.items()
        }

    load.calls = []  # type: ignore[attr-defined]
    return load


def _record(s3: FakeS3, current: dict | None = None, **over):
    kwargs = {
        "bucket": "b", "s3_client": s3, "ohlcv_loader": _loader(),
        "market_regime": "bull", "written_at": "2026-08-20T12:00:00+00:00",
    }
    kwargs.update(over)
    return record_completed_week(current or _current_doc(), **kwargs)


# ── Which week a run records ─────────────────────────────────────────────────


class TestWhichWeekIsRecorded:
    def test_a_run_records_the_week_its_new_cut_just_ENDED(self):
        """Not the week it is starting. A week's return is realized at the
        moment the next cut replaces the current one, which is now."""
        status = _record(_s3_with_prior())
        assert status["status"] == "ok"
        assert status["week"] == [WEEK_START, WEEK_END]

    def test_every_arm_of_the_slot_gets_a_row(self):
        s3 = _s3_with_prior()
        _record(s3)
        df = read_ledger(bucket="b", s3_client=s3)
        assert sorted(df["arm"]) == sorted(_ARMS)

    def test_an_arm_that_held_no_cut_is_a_recorded_MISS_not_an_omission(self):
        """champion-challenger-policy.md §3: a cycle where an arm produced no
        output is recorded as a miss. Omitting it makes a dead arm and a
        scored-and-flat arm render identically."""
        s3 = _s3_with_prior()
        status = _record(s3)
        missing = _ARMS[4]
        assert status["arms_missing"] == [missing]
        df = read_ledger(bucket="b", s3_client=s3)
        row = df[df["arm"] == missing].iloc[0]
        assert pd.isna(row["gross_log_return"])
        assert int(row["n_names"]) == 0

    def test_an_unfinished_week_writes_NOTHING(self):
        """The cut was carried forward — no week closed. Writing a partial week
        would put a number in the series a later run would want to change."""
        s3 = _s3_with_prior()
        status = _record(s3, _current_doc(cut_effective_date=WEEK_START))
        assert status["status"] == "skipped"
        assert status["reason"] == "no_week_closed"
        assert LEDGER_KEY not in s3.puts

    def test_a_first_run_with_no_prior_cut_is_skipped_not_failed(self):
        s3 = FakeS3()
        status = _record(s3, _membership(
            run_date=WEEK_END, generated_at="2026-08-20T09:00:00+00:00",
            cut_effective_date=WEEK_END, cuts={}, turnover=None,
        ))
        assert status["status"] == "skipped"
        assert status["reason"] == "no_recoverable_prior_cut"
        assert s3.puts == []

    def test_it_never_walks_history(self):
        """A run records at most the ONE week that closed. Backfilling from
        archived membership artifacts would mix restatement-exposed weeks into
        an append-only store whose only property is that nothing was restated
        (alpha-engine-config-I8255; I8262 owns any deliberate prior)."""
        s3 = _s3_with_prior()
        _record(s3)
        assert s3.lists == 0
        df = read_ledger(bucket="b", s3_client=s3)
        assert set(df["week_start"]) == {WEEK_START}


# ── What the week is priced against ──────────────────────────────────────────


class TestTheThreeBenchmarkLegs:
    def test_the_champion_leg_is_present_on_every_challenger_and_absent_on_the_champion(self):
        s3 = _s3_with_prior()
        _record(s3)
        df = read_ledger(bucket="b", s3_client=s3).set_index("arm")
        assert pd.isna(df.loc[CHAMPION, "champion_log_return"])
        for arm in _ARMS[1:4]:
            assert df.loc[arm, "champion_log_return"] == pytest.approx(
                df.loc[CHAMPION, "gross_log_return"]
            )

    def test_the_serving_arm_is_resolved_from_the_pointer_not_a_literal(self):
        """champion-challenger-policy.md §7.5. A hardcoded champion name is the
        artifact asserting something false about its own origin the moment the
        pointer moves."""
        s3 = _s3_with_prior()
        with patch("scoring.universe_membership.live_cut_champion", return_value=_ARMS[1]):
            _record(s3)
        df = read_ledger(bucket="b", s3_client=s3).set_index("arm")
        assert bool(df.loc[_ARMS[1], "is_champion"]) is True
        assert bool(df.loc[CHAMPION, "is_champion"]) is False

    def test_the_population_leg_is_the_whole_scanned_universe(self):
        s3 = _s3_with_prior()
        prices = {
            WEEK_START: dict.fromkeys([*POPULATION, "SPY"], 100.0),
            WEEK_END: {**dict.fromkeys(POPULATION, 100.0), "SPY": 100.0, "T000": 200.0},
        }
        _record(s3, ohlcv_loader=_loader(prices))
        df = read_ledger(bucket="b", s3_client=s3).set_index("arm")
        # One of twelve population names doubled; the arm holding it (3 names)
        # must show a much larger move than the population leg.
        assert df.loc[CHAMPION, "population_log_return"] == pytest.approx(0.05776, rel=1e-3)
        assert df.loc[CHAMPION, "gross_log_return"] == pytest.approx(0.23105, rel=1e-3)

    def test_the_benchmark_leg_is_SPY(self):
        s3 = _s3_with_prior()
        prices = {
            WEEK_START: dict.fromkeys([*POPULATION, "SPY"], 100.0),
            WEEK_END: {**dict.fromkeys(POPULATION, 100.0), "SPY": 110.0},
        }
        _record(s3, ohlcv_loader=_loader(prices))
        df = read_ledger(bucket="b", s3_client=s3).set_index("arm")
        assert df.loc[CHAMPION, "benchmark_log_return"] == pytest.approx(0.09531, rel=1e-3)
        assert df.loc[CHAMPION, "gross_log_return"] == pytest.approx(0.0)


# ── The prior basket, and where turnover comes from ──────────────────────────


class TestTheBasketThatWasHeld:
    def test_the_week_is_priced_against_the_IMMUTABLE_prior_copy(self):
        """A second same-day run replaces the dated pointer. Reading it would
        price the week against a basket that was never held for it — the
        clobber defect (I6785) entering through the reader."""
        clobbered = _prior_doc(
            generated_at="2026-08-13T18:00:00+00:00",
            cuts={arm: ["T009"] * 1 for arm in _ARMS},
        )
        s3 = _s3_with_prior(also_dated=clobbered)
        s3.objects[f"universe_membership/{WEEK_START}/membership.json"] = json.dumps(clobbered).encode()
        _record(s3)
        df = read_ledger(bucket="b", s3_client=s3).set_index("arm")
        assert int(df.loc[CHAMPION, "n_names"]) == 3  # the runs/ copy, not the clobber

    def test_an_unrecoverable_prior_is_never_priced_against_whatever_is_there(self):
        s3 = FakeS3({
            f"universe_membership/{WEEK_START}/membership.json": json.dumps(
                _prior_doc(generated_at="2026-08-13T18:00:00+00:00")
            ).encode(),
        })
        status = _record(s3)
        assert status["status"] == "skipped"
        assert status["reason"] == "no_recoverable_prior_cut"
        assert s3.puts == []

    def test_turnover_comes_from_the_artifact_not_a_recomputation(self):
        """`compute_turnover` already writes per-cut retention on every run
        (I6785). A second derivation here is the multi-writer drift the
        membership artifact exists to end — so the cost must follow the
        RECORDED number even when the two cuts' actual overlap differs."""
        s3 = _s3_with_prior(_prior_doc(turnover=_turnover_block(retention=40.0)))
        _record(s3)
        loose = read_ledger(bucket="b", s3_client=s3).set_index("arm")
        s3b = _s3_with_prior(_prior_doc(turnover=_turnover_block(retention=90.0)))
        _record(s3b)
        tight = read_ledger(bucket="b", s3_client=s3b).set_index("arm")
        # Identical baskets and identical prices; only the RECORDED retention
        # differs, and the cost must move with it.
        assert loose.loc[CHAMPION, "gross_log_return"] == pytest.approx(
            tight.loc[CHAMPION, "gross_log_return"]
        )
        assert loose.loc[CHAMPION, "turnover_frac"] == pytest.approx(0.60)
        assert tight.loc[CHAMPION, "turnover_frac"] == pytest.approx(0.10)
        assert loose.loc[CHAMPION, "cost_bps"] > tight.loc[CHAMPION, "cost_bps"]
        assert loose.loc[CHAMPION, "net_log_return"] < tight.loc[CHAMPION, "net_log_return"]

    def test_a_prior_with_no_turnover_block_leaves_net_null_never_gross(self):
        s3 = _s3_with_prior(_prior_doc(turnover=None))
        _record(s3)
        row = read_ledger(bucket="b", s3_client=s3).set_index("arm").loc[CHAMPION]
        assert row["gross_log_return"] is not None
        assert pd.isna(row["net_log_return"])  # never gross under a "net" name
        assert row["net_unavailable_reason"] == "turnover_unknown_so_cost_uncomputable"


# ── Immutability, end to end ─────────────────────────────────────────────────


class TestRerunningTheSameDate:
    def test_a_rerun_writes_nothing_new_and_reports_every_row_as_skipped(self):
        s3 = _s3_with_prior()
        first = _record(s3)
        assert len(first["report"]["written"]) == len(_ARMS)
        before = s3.objects[LEDGER_KEY]
        second = _record(s3)
        assert second["report"]["written"] == []
        assert len(second["report"]["skipped_immutable"]) == len(_ARMS)
        assert s3.objects[LEDGER_KEY] == before

    def test_a_rerun_does_not_silently_no_op(self):
        """"Already written" and "failed to write" must never look the same to
        the caller — that indistinguishability is the whole reason `append_week`
        returns a report rather than a bool."""
        s3 = _s3_with_prior()
        _record(s3)
        assert _record(s3)["status"] == "ok"


# ── A week that closed and could not be measured ─────────────────────────────


class TestUnmeasurable:
    def test_unpriced_boundaries_are_UNMEASURABLE_not_an_empty_success(self):
        """champion-challenger-policy.md §7.2 — the fleet's dominant bug class
        is a record asserting an action that never happened."""
        s3 = _s3_with_prior()
        stale = {"2026-07-01": dict.fromkeys([*POPULATION, "SPY"], 100.0)}
        status = _record(s3, ohlcv_loader=_loader(stale))
        assert status["status"] == "unmeasurable"
        assert status["reason"] == "week_boundaries_unpriced"
        assert LEDGER_KEY not in s3.puts

    def test_a_closes_read_failure_is_unmeasurable_and_says_why(self):
        def boom(*a, **k):
            raise RuntimeError("arcticdb unreachable")

        status = _record(_s3_with_prior(), ohlcv_loader=boom)
        assert status["status"] == "unmeasurable"
        assert status["reason"] == "closes_read_failed"
        assert "arcticdb unreachable" in status["detail"]

    def test_a_boundary_resolved_to_a_nearby_session_is_RECORDED_not_assumed(self):
        """A holiday re-cut, or a close that has not landed yet, moves the
        priced span. A week priced over a shorter span than its label claims is
        a wrong number wearing a right name."""
        s3 = _s3_with_prior()
        shifted = {
            "2026-08-12": dict.fromkeys([*POPULATION, "SPY"], 100.0),
            "2026-08-19": dict.fromkeys([*POPULATION, "SPY"], 110.0),
        }
        status = _record(s3, ohlcv_loader=_loader(shifted))
        assert status["status"] == "ok"
        assert status["priced_span"] == ["2026-08-12", "2026-08-19"]
        row = read_ledger(bucket="b", s3_client=s3).set_index("arm").loc[CHAMPION]
        assert row["week_start"] == WEEK_START and row["priced_from"] == "2026-08-12"
        assert row["week_end"] == WEEK_END and row["priced_to"] == "2026-08-19"

    def test_a_boundary_beyond_the_slip_is_refused_rather_than_stretched(self):
        assert resolve_boundary({"2026-08-01": {}}, WEEK_START) is None
        assert resolve_boundary({"2026-08-11": {}}, WEEK_START) == "2026-08-11"


# ── The scanner run itself ───────────────────────────────────────────────────


_HANDLER_PATH = Path(__file__).resolve().parent.parent / "lambda" / "scanner_handler.py"


def _load_handler_module():
    spec = importlib.util.spec_from_file_location("lambda_scanner_handler_ledger", _HANDLER_PATH)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["lambda_scanner_handler_ledger"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    mod._init_done = False
    return mod


def _artifact() -> dict:
    return {
        "run_date": "2026-08-22",
        "scanner_version": "v1.0",
        "generated_at": "2026-08-22T09:00:00+00:00",
        "population_tickers": ["AAPL", "GOOG"],
        "scanner_tickers": ["AMD", "BNY", "SN"],
        "agent_input_set": ["AAPL", "GOOG", "AMD"],
        "filters_applied": {},
        "stats": {
            "universe_size": 903, "post_scanner": 3, "population_size": 2,
            "agent_input_size": 3, "feature_store_enriched": 903,
            "feature_store_missing": 0, "new_vs_prior_cycle": [],
            "dropped_vs_prior_cycle": [], "prior_run_date": "2026-08-15",
            "baseline_missing": False,
        },
    }


def _run_handler(ledger_return, *, order: list | None = None):
    mod = _load_handler_module()

    def _ledger(*a, **k):
        if order is not None:
            order.append("ledger")
        return ledger_return

    def _promote(*a, **k):
        if order is not None:
            order.append("promotion")
        return {"decision": "hold", "champion": CHAMPION, "reason_code": "cooldown"}

    with (
        patch.object(mod, "_ensure_init"),
        patch("data.scanner_orchestrator.build_candidates_artifact", return_value=_artifact()),
        patch("data.scanner_orchestrator.write_candidates_artifact",
              return_value="candidates/2026-08-21/candidates.json"),
        patch("data.scanner_orchestrator.write_universe_board_for_scanner_run",
              return_value="scanner/universe/2026-08-21/universe.json"),
        patch("scoring.universe_membership.compute_and_write_universe_membership",
              return_value="universe_membership/2026-08-21/membership.json"),
        patch("scoring.universe_membership.read_latest_membership", return_value=_current_doc()),
        patch("data.fetchers.feature_store_reader.read_latest_factor_loadings", return_value={}),
        patch("scoring.weekly_ledger.record_completed_week", side_effect=_ledger),
        patch("scoring.leaderboard_producers.build_cuts_leaderboard",
              return_value={"status": "ok", "key": "research/cuts_leaderboard/2026-08-21.json"}),
        patch("scoring.cut_promotion.run_cut_promotion", side_effect=_promote),
        patch("observe_alerts.publish_observe_alert") as alert,
        patch("boto3.client", return_value=MagicMock()),
    ):
        result = mod.handler({"run_date": "2026-08-22"}, context=None)
    return result, alert


class TestTheScannerRunWiring:
    def test_the_scanner_run_records_the_week_and_surfaces_the_report(self):
        ok = {
            "status": "ok", "week": [WEEK_START, WEEK_END],
            "report": {"written": [[a, WEEK_START] for a in _ARMS],
                       "skipped_immutable": [], "restated": []},
            "arms_priced": _ARMS[:4], "arms_missing": [_ARMS[4]],
        }
        result, alert = _run_handler(ok)
        assert result["status"] == "OK"
        ledger = result["summary"]["weekly_ledger"]
        assert ledger["status"] == "ok"
        assert ledger["week"] == [WEEK_START, WEEK_END]
        # The per-row report is carried verbatim, not collapsed to a count —
        # "written", "skipped_immutable" and "restated" are three different
        # claims about an append-only store.
        assert len(ledger["report"]["written"]) == len(_ARMS)
        assert "weekly_ledger" in result["summary"]["boards"]["completed"]
        alert.assert_not_called()

    def test_an_unmeasurable_week_is_ALERTED_not_left_as_a_WARN(self):
        """A write that silently does not happen is indistinguishable from a
        week in which nothing was measured, and an append-only series can never
        recover it later (ARCHITECTURE §61(a); §7.2)."""
        result, alert = _run_handler({
            "status": "unmeasurable", "week": [WEEK_START, WEEK_END],
            "reason": "week_boundaries_unpriced", "detail": "no priced session",
        })
        assert result["status"] == "OK"  # observe-only never reds the run
        alert.assert_called_once()
        kwargs = alert.call_args.kwargs
        assert kwargs["severity"] == "ERROR"
        assert kwargs["source"] == "research:weekly_ledger"
        assert "week_boundaries_unpriced" in kwargs["message"]

    def test_a_week_that_did_not_close_does_not_alert(self):
        """Silence and a correct nothing-to-do must not render identically —
        but a nothing-to-do must not page either."""
        result, alert = _run_handler({"status": "skipped", "reason": "no_week_closed"})
        alert.assert_not_called()
        assert result["summary"]["weekly_ledger"]["status"] == "skipped"

    def test_the_ledger_runs_BEFORE_the_promotion_engine_can_move_the_pointer(self):
        """The ledger resolves the serving arm from the live pointer (§7.5),
        and the promotion engine can move that pointer inside this same
        invocation. Recording afterwards would attribute the incoming champion
        to a week the outgoing one served."""
        order: list[str] = []
        _run_handler({"status": "skipped", "reason": "no_week_closed"}, order=order)
        assert order == ["ledger", "promotion"]

    def test_a_ledger_failure_never_reds_the_scanner_run(self):
        mod = _load_handler_module()
        with (
            patch.object(mod, "_ensure_init"),
            patch("data.scanner_orchestrator.build_candidates_artifact", return_value=_artifact()),
            patch("data.scanner_orchestrator.write_candidates_artifact",
                  return_value="candidates/2026-08-21/candidates.json"),
            patch("scoring.universe_membership.compute_and_write_universe_membership",
                  return_value="universe_membership/2026-08-21/membership.json"),
            patch("data.fetchers.feature_store_reader.read_latest_factor_loadings", return_value={}),
            patch("scoring.weekly_ledger.record_completed_week",
                  side_effect=RuntimeError("ledger exploded")),
            patch("observe_alerts.publish_observe_alert") as alert,
            patch("boto3.client", return_value=MagicMock()),
        ):
            result = mod.handler({"run_date": "2026-08-22"}, context=None)
        assert result["status"] == "OK"
        assert result["summary"]["weekly_ledger"]["status"] == "error"
        # The promotion block alerts too under a MagicMock S3; what matters is
        # that the LEDGER's own failure reached the alarmed surface.
        assert [c for c in alert.call_args_list
                if c.kwargs["source"] == "research:weekly_ledger"]
