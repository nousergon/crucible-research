"""Coverage-ledger intake policy + budget guard."""

from __future__ import annotations

from datetime import date

import boto3
import pytest
from moto import mock_aws

from tests._feed_helpers import feed_window_for
from thinktank.costs import BudgetExceededError, BudgetGuard
from thinktank.ledger import (
    load_ledger,
    record_thesis_write,
    save_ledger,
    select_intake,
)
from thinktank.schemas import CoverageLedger
from thinktank.settings import ThinktankSettings
from thinktank.storage import ThinktankStore


def _si(ledger, board, **kw):
    """``select_intake`` with the FeedWindow the scanner would have published.

    The window is no longer derived from the board by the code under test
    (alpha-engine-config-I7842); the fixture supplies it.
    """
    return select_intake(ledger, board, feed_window_for(board), **kw)


def _board(n: int = 10, none_scores: set[int] | None = None) -> dict:
    stocks = []
    for i in range(n):
        score = None if (none_scores and i in none_scores) else 100 - i
        stocks.append({"ticker": f"T{i}", "sector": "Tech", "attractiveness_score": score})
    return {"schema_version": 3, "stocks": stocks}


def _settings(**over) -> ThinktankSettings:
    base = {
        "bucket": "alpha-engine-research",
        "daily_new_names": 5,
        "rank_ceiling": 150,
        "sweep_chunk_size": 25,
        "stale_after_days": 30,
        "monthly_budget_usd_default": 25.0,
        "budget_ssm_param": "/thinktank/monthly_budget_usd",
        "providers": {},
        "tiers": {},
    }
    base.update(over)
    return ThinktankSettings(**base)


def test_intake_takes_top_uncovered_by_attractiveness():
    ledger = CoverageLedger()
    new_rows, refresh = _si(ledger, _board(), daily_new_names=5, rank_ceiling=150)
    assert [r["ticker"] for r in new_rows] == ["T0", "T1", "T2", "T3", "T4"]
    assert [r["_attractiveness_rank"] for r in new_rows] == [1, 2, 3, 4, 5]
    assert refresh == []


def test_intake_skips_covered_and_null_scores():
    ledger = CoverageLedger()
    record_thesis_write(ledger, ticker="T0", trading_day="2026-07-01", thesis_version=1)
    new_rows, _ = _si(ledger, _board(none_scores={1}), daily_new_names=3, rank_ceiling=150)
    # T0 covered, T1 has no score → next are T2,T3,T4
    assert [r["ticker"] for r in new_rows] == ["T2", "T3", "T4"]


def test_rank_ceiling_bounds_intake_and_stalest_refresh_fills_slots():
    ledger = CoverageLedger()
    record_thesis_write(ledger, ticker="T0", trading_day="2026-06-01", thesis_version=1)
    record_thesis_write(ledger, ticker="T1", trading_day="2026-06-20", thesis_version=1)
    record_thesis_write(ledger, ticker="T2", trading_day="2026-06-10", thesis_version=1)
    # ceiling 3 → only ranks 1-3 (T0,T1,T2) eligible, all covered → 0 new
    new_rows, refresh = _si(ledger, _board(), daily_new_names=2, rank_ceiling=3)
    assert new_rows == []
    # slots refresh the STALEST theses first
    assert refresh == ["T0", "T2"]


def test_skip_stale_refill_returns_new_only_even_with_slots_left():
    """Gap-fill mode (Saturday SF): staleness refresh is the daily job's
    role. skip_stale_refill=True must return an EMPTY refresh list even
    when there's remaining budget and stale covered names exist — padding
    with stale-refill picks would silently do daily's job for it."""
    ledger = CoverageLedger()
    record_thesis_write(ledger, ticker="T0", trading_day="2026-06-01", thesis_version=1)
    new_rows, refresh = _si(ledger, _board(), daily_new_names=5, rank_ceiling=150, skip_stale_refill=True)
    assert [r["ticker"] for r in new_rows] == ["T1", "T2", "T3", "T4", "T5"]
    assert refresh == []


def test_skip_stale_refill_zero_gap_returns_nothing():
    """The gap-fill caller sizes daily_new_names to the EXACT measured gap
    (coverage_gap.uncovered_count) — when that's 0 (fully covered), no new
    names should be selected at all."""
    ledger = CoverageLedger()
    for i in range(10):
        record_thesis_write(ledger, ticker=f"T{i}", trading_day="2026-07-01", thesis_version=1)
    new_rows, refresh = _si(ledger, _board(), daily_new_names=0, rank_ceiling=150, skip_stale_refill=True)
    assert new_rows == []
    assert refresh == []


def test_stale_entries_are_refreshed_regardless_of_new_names_slot_fill():
    """TT-2.1-staleness-sla-is-actionable / alpha-engine-config-I6478: the
    ``stale_after_days`` horizon must be an actionable SLA, not merely
    informational. Before this fix, the stale-refill fallback in
    ``select_intake`` only fired when the new-names intake slot was
    otherwise UNFILLED (``slots_left > 0``) — on a day with enough fresh
    uncovered names to fill ``daily_new_names`` on their own (coverage
    "already full" for the day), a covered name past its staleness horizon
    produced no refresh and no signal at all.

    This asserts the invariant directly rather than a hand-picked literal:
    independently compute every covered ticker whose thesis age (relative
    to ``trading_day``) is >= ``stale_after_days`` — the ledger's own
    breach set, mirroring test_thinktank_tier_contract.py's shape of
    deriving the expected set from the data/code itself — and require
    every one of them to appear in ``select_intake``'s ``refresh``, even
    though the board has ample uncovered names to fill every
    ``daily_new_names`` slot with brand-new coverage (the exact scenario
    the pre-fix fallback could never reach)."""
    trading_day = "2026-08-04"
    stale_after_days = 30
    ledger = CoverageLedger()
    record_thesis_write(ledger, ticker="T0", trading_day="2026-06-01", thesis_version=1)  # 64d old — breach
    record_thesis_write(ledger, ticker="T1", trading_day="2026-06-20", thesis_version=1)  # 45d old — breach
    record_thesis_write(ledger, ticker="T2", trading_day="2026-08-01", thesis_version=1)  # 3d old — fresh

    # T0-T2 are covered; the board's uncovered pool (T3..) is large enough
    # that ordinary new-name intake fills every daily_new_names slot on its
    # own — the "coverage already full" condition under which the
    # discretionary graceful-refresh path never fires (slots_left <= 0).
    board = _board(n=20)
    new_rows, refresh = _si(
        ledger,
        board,
        daily_new_names=5,
        rank_ceiling=150,
        stale_after_days=stale_after_days,
        trading_day=trading_day,
    )

    assert len(new_rows) == 5
    assert all(r["ticker"] not in {"T0", "T1", "T2"} for r in new_rows)

    breached = {
        e.ticker
        for e in ledger.entries.values()
        if (date.fromisoformat(trading_day) - date.fromisoformat(e.thesis_updated_on)).days >= stale_after_days
    }
    assert breached == {"T0", "T1"}
    assert breached <= set(refresh), (
        f"stale_after_days={stale_after_days} breach set {breached} not fully "
        f"covered by refresh={refresh} — the staleness horizon produced no "
        "action on a day coverage was already full "
        "(TT-2.1-staleness-sla-is-actionable)"
    )
    assert "T2" not in refresh  # fresh entry must not be force-refreshed


def test_stale_breach_is_mandatory_even_under_skip_stale_refill():
    """The SLA floor (breach-forced refresh) is independent of
    ``skip_stale_refill`` — that flag suppresses only the DISCRETIONARY
    graceful-refill path (gap_fill's "staleness is daily's job, not mine"),
    never the mandatory breach floor. Only exercised here at the function
    level: the gap_fill caller (``gap_fill_fanout.plan_gap_fill``)
    intentionally never passes ``stale_after_days``/``trading_day``, so in
    production this floor activates only on the daily cadence — see
    ``thinktank/run.py``."""
    trading_day = "2026-08-04"
    ledger = CoverageLedger()
    record_thesis_write(ledger, ticker="T0", trading_day="2026-06-01", thesis_version=1)  # 64d old — breach

    new_rows, refresh = _si(
        ledger,
        _board(),
        daily_new_names=5,
        rank_ceiling=150,
        skip_stale_refill=True,
        stale_after_days=30,
        trading_day=trading_day,
    )
    assert [r["ticker"] for r in new_rows] == ["T1", "T2", "T3", "T4", "T5"]
    assert refresh == ["T0"]


def test_record_thesis_write_updates_existing_entry():
    ledger = CoverageLedger()
    record_thesis_write(
        ledger,
        ticker="T9",
        trading_day="2026-07-01",
        thesis_version=1,
        sector="Tech",
        attractiveness_rank=4,
    )
    record_thesis_write(ledger, ticker="T9", trading_day="2026-07-02", thesis_version=2)
    entry = ledger.entries["T9"]
    assert entry.thesis_version == 2
    assert entry.thesis_updated_on == "2026-07-02"
    assert entry.covered_since == "2026-07-01"
    assert entry.attractiveness_rank_at_entry == 4


def test_ledger_round_trips_through_s3():
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="alpha-engine-research")
        store = ThinktankStore("alpha-engine-research", s3)
        ledger = CoverageLedger()
        record_thesis_write(ledger, ticker="T5", trading_day="2026-07-02", thesis_version=1)
        save_ledger(store, ledger)
        again = load_ledger(store)
        assert again.entries["T5"].thesis_version == 1
        assert again.updated_at


def test_budget_guard_refuses_at_cap(monkeypatch):
    monkeypatch.setenv("THINKTANK_MONTHLY_BUDGET_USD", "10.0")
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="alpha-engine-research")
        store = ThinktankStore("alpha-engine-research", s3)
        guard = BudgetGuard(store, _settings())

        spent, limit = guard.check("2026-07-02")
        assert (spent, limit) == (0.0, 10.0)

        guard.record_run("2026-07-02", run_id="r1", trading_day="2026-07-02", cost_usd=9.5)
        guard.check("2026-07-02")  # still under
        guard.record_run("2026-07-02", run_id="r2", trading_day="2026-07-02", cost_usd=0.6)
        with pytest.raises(BudgetExceededError):
            guard.check("2026-07-02")
        # a new month starts a fresh ledger
        spent, _ = guard.check("2026-08-01")
        assert spent == 0.0


def test_budget_limit_falls_closed_onto_default_when_ssm_unreadable(monkeypatch):
    monkeypatch.delenv("THINKTANK_MONTHLY_BUDGET_USD", raising=False)

    class _BrokenSSM:
        def get_parameter(self, **kw):
            raise RuntimeError("ssm down")

    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="alpha-engine-research")
        store = ThinktankStore("alpha-engine-research", s3)
        guard = BudgetGuard(store, _settings(), ssm_client=_BrokenSSM())
        assert guard.limit_usd() == 25.0  # YAML default cap still enforced
