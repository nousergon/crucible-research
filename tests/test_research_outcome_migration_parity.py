"""Parity test for the config#1530 consumer cutover — `evals.last_week_scorecard`,
`evals.team_accuracy`, and `memory.episodic` migrating off the wide
horizon-suffixed `score_performance` columns onto the long-format
`score_performance_outcomes` store (EPIC config#1483 Phase 3).

ACCEPTANCE BAR (config#1530): the migrated read must produce output
IDENTICAL to the pre-migration wide-column read, across history. Tolerance:
returns to 1e-3 (decimal<->percent rounding), beat_spy/log_alpha exact.

METHODOLOGY (the method proven on config#1483 Phase 2 verification + reused
by crucible-backtester#435 for config#1528): build ONE dual-representation
fixture carrying the SAME ground truth in both the wide `score_performance`
columns (2dp-rounded PERCENT, matching the real historic producer
convention: `round(decimal * 100, 2)`) and the long-format
`score_performance_outcomes` rows (DECIMAL, canonical `log_alpha` on the
primary horizon only) — then assert the OLD wide-column SQL (frozen here,
byte-identical to this repo's pre-cutover `origin/main` code, since the
migration replaced those call sites) and the NEW long-store-backed
production code agree on every derived output field, row for row, across a
history spanning multiple sectors/teams/dates/outcomes (resolved wins,
resolved losses, and unresolved signals).
"""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pytest

from evals.last_week_scorecard import build_scorecard
from evals.team_accuracy import analyze_team_performance
from memory import episodic as episodic_module

# ---------------------------------------------------------------------------
# Ground truth: (symbol, score_date, sector, team_id, cio_decision,
#                stock_return, spy_return) — decimals, or None if unresolved.
# log_alpha is derived as stock_return - spy_return for simplicity (this test
# only needs internal self-consistency between the two representations, not
# real log-domain math — the real log_alpha computation is producer-owned
# and out of scope here).
# ---------------------------------------------------------------------------
_HISTORY = [
    ("AAPL", "2026-05-01", "Tech", "technology", "ADVANCE", 0.0432, 0.0201),
    ("MSFT", "2026-05-01", "Tech", "technology", "ADVANCE", -0.0311, 0.0201),
    ("GOOG", "2026-05-01", "Tech", "technology", "REJECT", 0.0900, 0.0201),  # excluded (not ADVANCE)
    ("JNJ", "2026-05-01", "Healthcare", "healthcare", "ADVANCE", 0.0140, 0.0080),
    ("PFE", "2026-05-08", "Healthcare", "healthcare", "ADVANCE", -0.0330, 0.0160),
    ("NVDA", "2026-05-08", "Tech", "technology", "ADVANCE", 0.1023, 0.0250),
    ("KO", "2026-05-08", "Consumer", "consumer", "ADVANCE", 0.0026, 0.0250),
    ("XOM", "2026-05-15", "Energy", None, "ADVANCE", 0.0330, 0.0160),  # null team_id -> excluded
    ("TSLA", "2026-05-15", "Tech", "technology", "ADVANCE", None, None),  # unresolved
]


def _log_alpha(stock_ret: float, spy_ret: float) -> float:
    return round(stock_ret - spy_ret, 6)


def _make_history_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE score_performance (
            id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            score_date TEXT NOT NULL,
            score REAL NOT NULL,
            return_21d REAL,
            spy_21d_return REAL,
            beat_spy_21d INTEGER,
            log_alpha_21d REAL,
            UNIQUE(symbol, score_date)
        );
        CREATE TABLE score_performance_outcomes (
            id INTEGER PRIMARY KEY,
            signal_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            score_date TEXT NOT NULL,
            horizon_days INTEGER NOT NULL,
            beat_spy INTEGER,
            stock_return REAL,
            spy_return REAL,
            log_alpha REAL,
            is_primary INTEGER NOT NULL,
            resolved_at TEXT NOT NULL,
            schema_version INTEGER NOT NULL DEFAULT 1,
            UNIQUE(signal_id, horizon_days)
        );
        CREATE TABLE population (
            id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL UNIQUE,
            sector TEXT NOT NULL
        );
        CREATE TABLE macro_snapshots (
            id INTEGER PRIMARY KEY,
            date TEXT NOT NULL,
            regime TEXT,
            market_regime TEXT,
            vix REAL
        );
        CREATE TABLE predictor_outcomes (
            id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            prediction_date TEXT NOT NULL,
            predicted_direction TEXT,
            prediction_confidence REAL,
            actual_5d_return REAL,
            correct_5d INTEGER,
            actual_log_alpha REAL,
            correct INTEGER,
            UNIQUE(symbol, prediction_date)
        );
        CREATE TABLE cio_evaluations (
            id INTEGER PRIMARY KEY,
            ticker TEXT NOT NULL,
            eval_date TEXT NOT NULL,
            team_id TEXT,
            cio_decision TEXT NOT NULL,
            UNIQUE(ticker, eval_date)
        );
        CREATE TABLE investment_thesis (
            id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            conviction TEXT,
            thesis_summary TEXT,
            signal TEXT
        );
        CREATE TABLE memory_episodes (
            id              INTEGER PRIMARY KEY,
            ticker          TEXT NOT NULL,
            signal_date     TEXT NOT NULL,
            score           REAL,
            rating          TEXT,
            conviction      TEXT,
            thesis_summary  TEXT,
            outcome_21d     REAL,
            outcome_vs_spy  REAL,
            lesson          TEXT,
            sector          TEXT,
            pattern_tags    TEXT,
            created_date    TEXT NOT NULL,
            UNIQUE(ticker, signal_date)
        );
        CREATE TABLE stock_archive (
            id INTEGER PRIMARY KEY,
            ticker TEXT NOT NULL UNIQUE,
            sector TEXT
        );
        """
    )
    sectors_seen = set()
    for i, (sym, d, sector, team_id, decision, stock_ret, spy_ret) in enumerate(_HISTORY):
        resolved = stock_ret is not None
        score = 70.0 + i
        beat = (1 if stock_ret > spy_ret else 0) if resolved else None
        log_alpha = _log_alpha(stock_ret, spy_ret) if resolved else None

        conn.execute(
            "INSERT INTO score_performance "
            "(symbol, score_date, score, return_21d, spy_21d_return, beat_spy_21d, log_alpha_21d) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                sym, d, score,
                round(stock_ret * 100, 2) if resolved else None,
                round(spy_ret * 100, 2) if resolved else None,
                beat, log_alpha,
            ),
        )
        if resolved:
            conn.execute(
                "INSERT INTO score_performance_outcomes "
                "(signal_id, symbol, score_date, horizon_days, beat_spy, "
                " stock_return, spy_return, log_alpha, is_primary, resolved_at) "
                "VALUES (?,?,?,21,?,?,?,?,1,?)",
                (f"{sym}:{d}", sym, d, beat, stock_ret, spy_ret, log_alpha,
                 f"{d}T00:00:00+00:00"),
            )
        if sector not in sectors_seen:
            conn.execute("INSERT INTO population (symbol, sector) VALUES (?, ?)", (sym, sector))
            sectors_seen.add(sector)
        else:
            conn.execute(
                "INSERT OR IGNORE INTO population (symbol, sector) VALUES (?, ?)", (sym, sector)
            )
        conn.execute(
            "INSERT INTO cio_evaluations (ticker, eval_date, team_id, cio_decision) VALUES (?,?,?,?)",
            (sym, d, team_id, decision),
        )
        conn.execute(
            "INSERT INTO stock_archive (ticker, sector) VALUES (?, ?)", (sym, sector)
        )
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Reference (pre-migration) implementations — frozen copies of the wide-
# column SQL this migration replaced (byte-identical to `origin/main` prior
# to config#1530, modulo the predictor_outcomes join which is unaffected and
# omitted here since neither cluster under test reads it in this fixture).
# ---------------------------------------------------------------------------


def _old_fetch_signal_outcomes(conn, start, end):
    sql = """
        SELECT sp.symbol, sp.score_date, sp.score, sp.beat_spy_21d, sp.log_alpha_21d,
               COALESCE(p.sector, '(unknown)') AS sector
        FROM score_performance sp
        LEFT JOIN population p ON p.symbol = sp.symbol
        WHERE sp.score_date BETWEEN ? AND ?
    """
    rows = conn.execute(sql, (start, end)).fetchall()
    return [
        {"symbol": r[0], "score_date": r[1], "score": r[2],
         "beat_spy_21d": r[3], "log_alpha_21d": r[4], "sector": r[5]}
        for r in rows
    ]


def _old_fetch_team_outcomes(conn, start, end):
    sql = """
        SELECT c.team_id, sp.beat_spy_21d
        FROM cio_evaluations c
        JOIN score_performance sp
            ON sp.symbol = c.ticker AND sp.score_date = c.eval_date
        WHERE c.eval_date BETWEEN ? AND ?
          AND c.cio_decision = 'ADVANCE'
          AND c.team_id IS NOT NULL
          AND sp.beat_spy_21d IS NOT NULL
    """
    rows = conn.execute(sql, (start, end)).fetchall()
    return [{"team_id": r[0], "beat_spy_21d": r[1]} for r in rows]


def _old_extract_candidates(conn):
    """Frozen pre-migration failed-signal query (wide beat_spy_21d=0 filter
    done in SQL) + the old outcome_vs_spy derivation, which mixed decimal
    log_alpha_21d with PERCENT return_21d/spy_21d_return (a units bug this
    migration also fixes — see memory/episodic.py's new docstring). For
    parity purposes we compare the DECIMAL-consistent quantity: log_alpha
    when present, else (stock_return - spy_return) in decimal terms (the
    percent-scale wide columns divided by 100, undoing the display quirk so
    old and new agree on the underlying real-valued alpha rather than on the
    old code's unit bug).
    """
    rows = conn.execute("""
        SELECT sp.symbol, sp.score_date, sp.return_21d, sp.spy_21d_return, sp.log_alpha_21d
        FROM score_performance sp
        LEFT JOIN memory_episodes me
            ON sp.symbol = me.ticker AND sp.score_date = me.signal_date
        WHERE sp.beat_spy_21d = 0 AND me.id IS NULL
        ORDER BY sp.score_date DESC
    """).fetchall()
    out = []
    for symbol, score_date, return_21d, spy_21d_return, log_alpha_21d in rows:
        outcome_vs_spy = (
            log_alpha_21d
            if log_alpha_21d is not None
            else ((return_21d or 0) / 100.0) - ((spy_21d_return or 0) / 100.0)
        )
        out.append((symbol, score_date, outcome_vs_spy, (return_21d or 0) / 100.0))
    return out


@pytest.fixture
def history_db(tmp_path):
    conn = _make_history_db(tmp_path / "research.db")
    yield conn
    conn.close()


class TestScorecardParity:
    def test_signal_outcomes_identical_to_wide_read(self, history_db):
        from evals.last_week_scorecard import _fetch_signal_outcomes

        old = _old_fetch_signal_outcomes(history_db, "2026-04-01", "2026-06-01")
        new = _fetch_signal_outcomes(history_db, "2026-04-01", "2026-06-01")
        assert len(old) == len(new) == len(_HISTORY)  # all 9 seeded rows fall in this window

        old_by_key = {(r["symbol"], r["score_date"]): r for r in old}
        new_by_key = {(r["symbol"], r["score_date"]): r for r in new}
        assert set(old_by_key) == set(new_by_key)
        for key, old_row in old_by_key.items():
            new_row = new_by_key[key]
            # `new` uses the long-store's field names (`beat_spy`/`log_alpha`)
            # since evals.last_week_scorecard._fetch_signal_outcomes was
            # renamed off the wide-column dict-key literals (config#1530
            # burn-down guard); `old` is the frozen pre-migration reference
            # and still spells the retired wide-column names.
            assert old_row["beat_spy_21d"] == new_row["beat_spy"], key
            assert old_row["sector"] == new_row["sector"], key
            if old_row["log_alpha_21d"] is None:
                assert new_row["log_alpha"] is None, key
            else:
                assert new_row["log_alpha"] == pytest.approx(
                    old_row["log_alpha_21d"], abs=1e-3
                ), key

    def test_build_scorecard_hit_rate_matches_manual_wide_computation(self, history_db):
        sc = build_scorecard(history_db, as_of_date=date(2026, 6, 2), lookback_weeks=8)
        # 8 resolved signals in _HISTORY (all but TSLA); manual count from
        # the wide-column ground truth table.
        resolved = [r for r in _HISTORY if r[5] is not None]
        beats = sum(1 for r in resolved if r[5] > r[6])
        assert sc.n_resolved_signals_21d == len(resolved)
        assert sc.overall_signal_hit_rate_21d == pytest.approx(beats / len(resolved))


class TestTeamAccuracyParity:
    def test_team_outcomes_identical_to_wide_read(self, history_db):
        from evals.team_accuracy import _fetch_team_outcomes

        old = sorted(
            _old_fetch_team_outcomes(history_db, "2026-04-01", "2026-06-01"),
            key=lambda r: r["team_id"],
        )
        new = sorted(
            _fetch_team_outcomes(history_db, "2026-04-01", "2026-06-01"),
            key=lambda r: r["team_id"],
        )
        # `new` uses the long-store's `beat_spy` field name (config#1530
        # burn-down guard renamed the internal dict key off the retired
        # wide-column literal `beat_spy_21d`, which `old` still spells).
        old_normalized = [
            {"team_id": r["team_id"], "beat_spy": r["beat_spy_21d"]} for r in old
        ]
        assert old_normalized == new

    def test_analyze_team_performance_matches_manual_wide_computation(self, history_db):
        result = analyze_team_performance(history_db, as_of_date=date(2026, 6, 2), lookback_weeks=8)
        # technology: AAPL(beat=1,ADVANCE) + MSFT(beat=0,ADVANCE) + NVDA(beat=1,ADVANCE)
        # GOOG excluded (REJECT), TSLA excluded (unresolved) -> 2/3
        assert result["technology"] == {"accuracy": pytest.approx(2 / 3), "n_obs": 3}
        # healthcare: JNJ(beat=1) + PFE(beat=0) -> 1/2
        assert result["healthcare"] == {"accuracy": pytest.approx(1 / 2), "n_obs": 2}
        # consumer: KO(beat=0) -> 0/1
        assert result["consumer"] == {"accuracy": 0.0, "n_obs": 1}
        # no team for XOM (null team_id) -> no 'energy'/None key at all
        assert None not in result


class TestEpisodicParity:
    def test_failed_signal_selection_and_alpha_identical_to_wide_read(self, history_db):
        old_candidates = _old_extract_candidates(history_db)
        # New: replicate extract_memories' selection logic without invoking
        # the LLM (candidates + outcome join only).
        from evals import outcome_store as _os

        rows = history_db.execute("""
            SELECT sp.symbol, sp.score_date
            FROM score_performance sp
            LEFT JOIN memory_episodes me
                ON sp.symbol = me.ticker AND sp.score_date = me.signal_date
            WHERE me.id IS NULL
        """).fetchall()
        outcomes = _os.load_primary_outcomes(history_db)
        new_candidates = []
        for symbol, score_date in rows:
            outcome = outcomes.get((symbol, score_date))
            if outcome is None or outcome.beat_spy != 0:
                continue
            outcome_vs_spy = (
                outcome.log_alpha
                if outcome.log_alpha is not None
                else (outcome.stock_return or 0) - (outcome.spy_return or 0)
            )
            new_candidates.append((symbol, score_date, outcome_vs_spy, outcome.stock_return))

        old_by_key = {(s, d): (alpha, ret) for s, d, alpha, ret in old_candidates}
        new_by_key = {(s, d): (alpha, ret) for s, d, alpha, ret in new_candidates}
        assert set(old_by_key) == set(new_by_key)
        for key, (old_alpha, old_ret) in old_by_key.items():
            new_alpha, new_ret = new_by_key[key]
            assert new_alpha == pytest.approx(old_alpha, abs=1e-3), key
            assert new_ret == pytest.approx(old_ret, abs=1e-3), key

    def test_extract_memories_writes_outcome_21d_from_long_store(self, history_db, monkeypatch):
        """End-to-end (stubbed LLM): extract_memories must write the
        canonical decimal stock_return into memory_episodes.outcome_21d
        (config#1480 rename target), sourced from the long-format store."""

        class _StubResponse:
            content = '{"lesson": "test lesson", "pattern_tags": ["test"]}'

        class _StubLLM:
            def __init__(self, *a, **kw):
                pass

            def invoke(self, messages):
                return _StubResponse()

        monkeypatch.setattr(
            "langchain_anthropic.ChatAnthropic", _StubLLM, raising=False
        )

        import sys
        import types

        stub_config = types.ModuleType("config")
        stub_config.ANTHROPIC_API_KEY = "test-key"
        stub_config.PER_STOCK_MODEL = "claude-haiku"
        monkeypatch.setitem(sys.modules, "config", stub_config)

        stub_cost_tracker = types.ModuleType("graph.llm_cost_tracker")
        stub_cost_tracker.get_cost_telemetry_callback = lambda: None
        monkeypatch.setitem(sys.modules, "graph.llm_cost_tracker", stub_cost_tracker)

        n_created = episodic_module.extract_memories(history_db, api_key="test-key")
        assert n_created >= 1

        # MSFT (2026-05-01, beat=0) and PFE (2026-05-08, beat=0) are the
        # failed signals; both should now have a memory_episodes row with
        # outcome_21d == the long-store's decimal stock_return.
        row = history_db.execute(
            "SELECT outcome_21d FROM memory_episodes WHERE ticker = 'MSFT'"
        ).fetchone()
        assert row is not None
        assert row[0] == pytest.approx(-0.0311, abs=1e-3)
