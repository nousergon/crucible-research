"""The universe-cut slot's append-only weekly performance ledger
(alpha-engine-config-I8261, Brian's ruling 2026-08-24 — option (d)).

Brian: *"shouldn't we just be tracking performance weekly?"*

RED on origin/main at a8952d43: `scoring/weekly_ledger` does not exist.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring.weekly_ledger import (  # noqa: E402
    LEDGER_COLS,
    LEDGER_VERSION,
    REFERENCE_NOTIONAL_USD,
    WeeklyLedgerError,
    append_week,
    build_week_row,
    chained_log_return,
    equal_weight_log_return,
    holding_period,
    net_from_gross,
    paired_weekly_differences,
    turnover_cost_bps,
)

WRITTEN_AT = "2026-08-24T12:00:00+00:00"


# ── The holding period ───────────────────────────────────────────────────────


class TestHoldingPeriod:
    def test_a_week_runs_until_the_next_cut_replaces_it(self):
        assert holding_period("2026-08-20", "2026-08-27") == ("2026-08-20", "2026-08-27")

    def test_an_unfinished_week_is_not_written(self):
        """The current week is still being held; its return is not realized.

        Writing a partial week would put a number in the series that a later
        run would want to change — exactly what the append-only guarantee
        forbids. An unfinished week is not a missing week.
        """
        assert holding_period("2026-08-20", None) is None

    def test_the_period_is_not_a_fixed_five_sessions(self):
        """A missed run, a holiday, or an off-cadence re-cut moves the real end
        date. A fixed window would attribute returns to an arm that had already
        been replaced, or drop returns it genuinely earned."""
        assert holding_period("2026-08-20", "2026-09-10") == ("2026-08-20", "2026-09-10")

    def test_a_non_advancing_next_cut_yields_no_period(self):
        assert holding_period("2026-08-27", "2026-08-20") is None
        assert holding_period("2026-08-20", "2026-08-20") is None


# ── The basket return ────────────────────────────────────────────────────────


class TestEqualWeightLogReturn:
    def test_log_domain_equal_weight(self):
        import math

        r, n = equal_weight_log_return(
            ["A", "B"], {"A": 100.0, "B": 50.0}, {"A": 110.0, "B": 55.0},
        )
        assert n == 2
        assert r == pytest.approx(math.log(1.1))

    def test_a_missing_name_is_dropped_not_treated_as_flat(self):
        """A delisting or a data gap is NOT a 0% return. Averaging a fabricated
        zero into the basket biases every arm toward the mean by an amount
        proportional to its own data quality — an arm with worse coverage would
        look artificially safe."""
        import math

        r, n = equal_weight_log_return(
            ["A", "GONE"], {"A": 100.0}, {"A": 110.0},
        )
        assert n == 1
        assert r == pytest.approx(math.log(1.1))

    def test_no_usable_name_is_none_not_zero(self):
        r, n = equal_weight_log_return(["X"], {}, {})
        assert r is None and n == 0

    def test_a_nonpositive_price_is_dropped(self):
        r, n = equal_weight_log_return(["A"], {"A": 0.0}, {"A": 10.0})
        assert r is None and n == 0


# ── Cost ─────────────────────────────────────────────────────────────────────


class TestTurnoverCost:
    def test_cost_scales_with_the_replaced_share(self):
        """The reason this column exists. An arm retaining 76% pays on 24% of
        the basket; one retaining 42% pays on 58%. Measured 2026-07-27, those
        are the live retention figures for two arms of this very slot."""
        low = turnover_cost_bps(0.24)
        high = turnover_cost_bps(0.58)
        assert low is not None and high is not None
        assert high > low
        assert high / low == pytest.approx(0.58 / 0.24, rel=1e-6)

    def test_unknown_turnover_is_none_never_zero(self):
        """Zero is a real and different claim — an arm that changed nothing.
        A slot whose arms differ mainly in churn must never render 'no churn'
        and 'churn unknown' identically."""
        assert turnover_cost_bps(None) is None

    def test_zero_turnover_costs_nothing_and_is_not_none(self):
        assert turnover_cost_bps(0.0) == pytest.approx(0.0)

    def test_it_uses_the_fleet_cost_engine_not_a_local_constant(self):
        """One shared square-root-impact model (shared-code-policy). A flat bps
        literal here would be a second cost model that drifts from the one the
        board and the backtester use."""
        import inspect

        import scoring.weekly_ledger as wl

        src = inspect.getsource(wl.turnover_cost_bps)
        assert "TransactionCostModel" in src

    def test_a_full_turnover_is_capped_at_the_whole_basket(self):
        assert turnover_cost_bps(1.5) == pytest.approx(turnover_cost_bps(1.0))


class TestNetFromGross:
    def test_cost_is_subtracted(self):
        net, reason = net_from_gross(0.01, 50.0)
        assert reason is None
        assert net == pytest.approx(0.01 - 0.005)

    def test_an_uncomputable_cost_yields_none_never_gross(self):
        """Returning gross under a column named `net` is the exact defect shape
        this arc has been about: a number that is plausible, close to right,
        and answers a different question than its name claims."""
        net, reason = net_from_gross(0.01, None)
        assert net is None
        assert reason == "turnover_unknown_so_cost_uncomputable"

    def test_a_missing_gross_yields_none_with_its_own_reason(self):
        net, reason = net_from_gross(None, 50.0)
        assert net is None
        assert reason == "gross_return_unavailable"


# ── The row ──────────────────────────────────────────────────────────────────


def _row(**over):
    base = {
        "arm": "attractiveness_top_60",
        "week_start": "2026-08-20", "week_end": "2026-08-27",
        "tickers": ["A", "B"],
        "closes_start": {"A": 100.0, "B": 50.0, "SPY": 500.0, "C": 10.0},
        "closes_end": {"A": 110.0, "B": 55.0, "SPY": 505.0, "C": 10.5},
        "turnover_frac": 0.25,
        "population_tickers": ["A", "B", "C"],
        "champion_tickers": ["A", "C"],
        "market_regime": "bull",
        "written_at": WRITTEN_AT,
    }
    base.update(over)
    return build_week_row(**base)


class TestRow:
    def test_row_carries_every_declared_column(self):
        assert set(_row()) == set(LEDGER_COLS)

    def test_the_row_is_a_pure_function_of_its_inputs(self):
        """`written_at` is an ARGUMENT, not a `utcnow()` call. A module that
        stamps its own clock cannot be tested for the idempotence its own
        append-only guarantee depends on."""
        assert _row() == _row()

    def test_all_three_benchmark_legs_are_recorded(self):
        r = _row()
        assert r["benchmark_log_return"] is not None
        assert r["population_log_return"] is not None
        assert r["champion_log_return"] is not None

    def test_the_row_stores_legs_not_differences(self):
        """A difference is one subtraction a consumer can always do. A leg it
        was never given is gone for good — and a stored difference cannot be
        re-based when the champion changes."""
        assert not [c for c in LEDGER_COLS if "alpha" in c or "excess" in c or "diff" in c]

    def test_the_champion_has_no_champion_leg(self):
        """An arm has no paired difference against itself, and a zero there
        would read as a measured tie."""
        r = _row(is_champion=True)
        assert r["champion_log_return"] is None
        assert r["is_champion"] is True

    def test_net_differs_from_gross_when_turnover_is_known(self):
        r = _row()
        assert r["net_log_return"] < r["gross_log_return"]
        assert r["net_unavailable_reason"] is None

    def test_unknown_turnover_leaves_net_null_and_says_why(self):
        r = _row(turnover_frac=None)
        assert r["gross_log_return"] is not None
        assert r["net_log_return"] is None
        assert r["net_unavailable_reason"] == "turnover_unknown_so_cost_uncomputable"

    def test_provenance_is_stamped_on_every_row(self):
        """Which implementation produced this number. Without it a later
        version's rows are indistinguishable from an earlier version's, and the
        whole point of not restating history is lost."""
        r = _row()
        assert r["ledger_version"] == LEDGER_VERSION
        assert r["code_sha"]
        assert r["written_at"] == WRITTEN_AT

    def test_regime_is_recorded_at_write_time(self):
        """Cheap now, unrecoverable later. At ~52 observations a year any read
        worth having will want to condition on regime rather than pool across
        it."""
        assert _row()["market_regime"] == "bull"

    def test_n_names_counts_contributors_not_the_requested_list(self):
        r = _row(tickers=["A", "B", "MISSING"])
        assert r["n_names"] == 2


# ── The store: immutability is the contract ──────────────────────────────────


class _S3:
    def __init__(self):
        self.obj: bytes | None = None
        self.puts = 0

    def get_object(self, Bucket, Key):  # noqa: N803
        if self.obj is None:
            raise RuntimeError("NoSuchKey")
        import io as _io

        return {"Body": _io.BytesIO(self.obj)}

    def put_object(self, Bucket, Key, Body, ContentType):  # noqa: N803
        self.obj = Body
        self.puts += 1


class TestAppendOnly:
    def test_first_write_lands(self):
        s3 = _S3()
        rep = append_week([_row()], s3_client=s3)
        assert rep["written"] == [("attractiveness_top_60", "2026-08-20")]
        assert not rep["skipped_immutable"]

    def test_rewriting_a_week_is_refused_and_REPORTED(self):
        """The whole contract. The leaderboard recomputes its cohort every run,
        so a scoring change silently restates numbers nobody can audit
        afterward. Here a week is written once and stands — and 'already
        written' is a recorded outcome, never a silent no-op, because it must
        not look the same to the caller as 'failed to write'."""
        s3 = _S3()
        append_week([_row()], s3_client=s3)
        rep = append_week([_row(turnover_frac=0.99)], s3_client=s3)
        assert rep["written"] == []
        assert rep["skipped_immutable"] == [("attractiveness_top_60", "2026-08-20")]

    def test_a_refused_row_does_not_change_the_stored_number(self):
        """The assertion that makes the one above mean something."""
        import pandas as pd

        from scoring.weekly_ledger import read_ledger

        s3 = _S3()
        append_week([_row()], s3_client=s3)
        before = read_ledger(s3_client=s3)["net_log_return"].iloc[0]
        append_week([_row(turnover_frac=0.99)], s3_client=s3)
        after = read_ledger(s3_client=s3)["net_log_return"].iloc[0]
        assert pd.notna(before) and before == after

    def test_a_new_week_appends_alongside_the_old_one(self):
        from scoring.weekly_ledger import read_ledger

        s3 = _S3()
        append_week([_row()], s3_client=s3)
        append_week([_row(week_start="2026-08-27", week_end="2026-09-03")], s3_client=s3)
        df = read_ledger(s3_client=s3)
        assert len(df) == 2
        assert sorted(df["week_start"]) == ["2026-08-20", "2026-08-27"]

    def test_restating_is_possible_but_must_be_asked_for(self):
        """One legitimate case: inputs later found to be wrong (a contaminated
        close, a mis-stamped cut). Deliberate and logged — not a default, and
        not how a code change reaches the history."""
        from scoring.weekly_ledger import read_ledger

        s3 = _S3()
        append_week([_row()], s3_client=s3)
        rep = append_week([_row(turnover_frac=0.99)], s3_client=s3, allow_restate=True)
        assert rep["restated"] == [("attractiveness_top_60", "2026-08-20")]
        assert read_ledger(s3_client=s3)["turnover_frac"].iloc[0] == pytest.approx(0.99)

    def test_two_rows_claiming_one_key_in_ONE_batch_is_an_error(self):
        """Last-write-wins inside a batch would make which row landed depend on
        row order, and neither would be recorded as skipped."""
        s3 = _S3()
        with pytest.raises(WeeklyLedgerError, match="same \\(arm, week_start\\)"):
            append_week([_row(), _row(turnover_frac=0.5)], s3_client=s3)

    def test_a_row_without_its_primary_key_is_an_error(self):
        """A row that cannot be identified cannot be protected from
        restatement, which is the only guarantee this store makes."""
        s3 = _S3()
        bad = _row()
        bad["arm"] = None
        with pytest.raises(WeeklyLedgerError, match="primary key"):
            append_week([bad], s3_client=s3)

    def test_an_absent_ledger_reads_as_none_not_empty(self):
        """'No ledger exists' and 'the ledger is empty' are different states,
        and a consumer that cannot tell them apart reports a healthy empty
        series for a store that was never created."""
        from scoring.weekly_ledger import read_ledger

        assert read_ledger(s3_client=_S3()) is None

    def test_an_empty_append_writes_nothing(self):
        s3 = _S3()
        assert append_week([], s3_client=s3) == {
            "written": [], "skipped_immutable": [], "restated": [],
        }
        assert s3.puts == 0


# ── Reading the series back ──────────────────────────────────────────────────


class TestChaining:
    def test_log_returns_add_across_weeks(self):
        """Why the ledger stores logs: a six-month read is the sum of 26 rows,
        computed by a reader, with no re-derivation from prices and no
        dependence on the scoring code that produced any single week."""
        assert chained_log_return([0.01, 0.02, -0.005]) == pytest.approx(0.025)

    def test_a_gap_makes_the_span_none_rather_than_skipping_it(self):
        """Skipping would silently report a 25-week return under a 26-week
        label. A gap in a compounding series is not a zero — it is a span that
        cannot be stated."""
        assert chained_log_return([0.01, None, 0.02]) is None

    def test_an_empty_span_is_none(self):
        assert chained_log_return([]) is None


class TestPairedDifferences:
    def test_it_differences_the_net_column_by_default(self):
        """The wrong default for a slot whose arms differ mainly in churn would
        be gross."""
        rows = [
            {"net_log_return": 0.02, "gross_log_return": 0.03, "champion_log_return": 0.01},
        ]
        assert paired_weekly_differences(rows) == [pytest.approx(0.01)]

    def test_gross_is_available_by_asking(self):
        rows = [
            {"net_log_return": 0.02, "gross_log_return": 0.03, "champion_log_return": 0.01},
        ]
        got = paired_weekly_differences(rows, column="gross_log_return")
        assert got == [pytest.approx(0.02)]

    def test_a_week_missing_either_leg_is_dropped_not_zeroed(self):
        """Substituting a zero for an absent champion leg would manufacture a
        week in which the arm exactly matched the champion."""
        rows = [
            {"net_log_return": 0.02, "champion_log_return": None},
            {"net_log_return": None, "champion_log_return": 0.01},
            {"net_log_return": 0.02, "champion_log_return": 0.01},
        ]
        assert paired_weekly_differences(rows) == [pytest.approx(0.01)]


class TestSotaProperties:
    def test_weekly_observations_do_not_overlap(self):
        """The statistical payoff, asserted against the overlap machinery
        itself. Successive weekly holding periods ABUT rather than overlap, so
        `date_clustered_stats`'s iid SE — written for exactly this shape and
        applied until now to 21-session forward windows — becomes correct
        rather than assumed (alpha-engine-config-I8263)."""
        from scoring.leaderboard_scoring import overlap_lags_for

        assert overlap_lags_for(5, 5) == 0

    def test_the_reference_notional_is_not_a_live_nav(self):
        """Tying an arm's cost to a live account NAV would make its HISTORICAL
        record move when the account did — the restatement this store exists to
        prevent, entering through the cost column."""
        assert isinstance(REFERENCE_NOTIONAL_USD, float)
        assert REFERENCE_NOTIONAL_USD > 0
