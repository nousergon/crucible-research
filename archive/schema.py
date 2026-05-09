"""
Database schema and migrations for research.db.

Tables are defined as CREATE IF NOT EXISTS statements — safe for new and
existing databases. Migrations are versioned and tracked in a schema_version
table so each ALTER runs exactly once.

To add a new migration:
  1. Add an entry to MIGRATIONS with the next version number
  2. Bump SCHEMA_VERSION to match
  3. Deploy — ensure_schema() will apply it on next Lambda cold-start
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone

log = logging.getLogger(__name__)

SCHEMA_VERSION = 13

# ── Table Definitions ────────────────────────────────────────────────────────

TABLES_SQL = """
CREATE TABLE IF NOT EXISTS investment_thesis (
    id                       INTEGER PRIMARY KEY,
    symbol                   TEXT NOT NULL,
    date                     TEXT NOT NULL,
    run_time                 TEXT NOT NULL,
    rating                   TEXT NOT NULL,
    score                    REAL NOT NULL,
    technical_score          REAL,
    quant_score              REAL,
    qual_score               REAL,
    macro_modifier           REAL,
    thesis_summary           TEXT,
    prev_rating              TEXT,
    prev_score               REAL,
    last_material_change_date TEXT,
    stale_days               INTEGER,
    consistency_flag         INTEGER DEFAULT 0,
    UNIQUE(symbol, date, run_time)
);

CREATE TABLE IF NOT EXISTS agent_reports (
    id          INTEGER PRIMARY KEY,
    symbol      TEXT,
    date        TEXT NOT NULL,
    run_time    TEXT NOT NULL,
    agent_type  TEXT NOT NULL,
    report_md   TEXT NOT NULL,
    word_count  INTEGER,
    UNIQUE(symbol, date, run_time, agent_type)
);

CREATE TABLE IF NOT EXISTS candidate_tenures (
    id              INTEGER PRIMARY KEY,
    symbol          TEXT NOT NULL,
    slot            INTEGER NOT NULL,
    entry_date      TEXT NOT NULL,
    exit_date       TEXT,
    exit_reason     TEXT,
    replaced_by     TEXT,
    peak_score      REAL,
    exit_score      REAL,
    tenure_days     INTEGER
);

CREATE TABLE IF NOT EXISTS active_candidates (
    slot            INTEGER PRIMARY KEY,
    symbol          TEXT NOT NULL,
    entry_date      TEXT NOT NULL,
    prior_tenures   INTEGER NOT NULL DEFAULT 0,
    score           REAL,
    consecutive_low_runs INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS scanner_appearances (
    id              INTEGER PRIMARY KEY,
    symbol          TEXT NOT NULL,
    date            TEXT NOT NULL,
    scanner_rank    INTEGER NOT NULL,
    scan_path       TEXT,
    tech_score      REAL,
    quant_score     REAL,
    qual_score      REAL,
    final_score     REAL,
    selected        INTEGER NOT NULL DEFAULT 0,
    selection_reason TEXT,
    UNIQUE(symbol, date)
);

CREATE TABLE IF NOT EXISTS technical_scores (
    id              INTEGER PRIMARY KEY,
    symbol          TEXT NOT NULL,
    date            TEXT NOT NULL,
    rsi_14          REAL,
    macd_signal     REAL,
    price_vs_ma50   REAL,
    price_vs_ma200  REAL,
    momentum_20d    REAL,
    technical_score REAL,
    UNIQUE(symbol, date)
);

CREATE TABLE IF NOT EXISTS macro_snapshots (
    id                  INTEGER PRIMARY KEY,
    date                TEXT NOT NULL UNIQUE,
    fed_funds_rate      REAL,
    treasury_2yr        REAL,
    treasury_10yr       REAL,
    yield_curve_slope   REAL,
    vix                 REAL,
    sp500_close         REAL,
    sp500_30d_return    REAL,
    oil_wti             REAL,
    gold                REAL,
    copper              REAL,
    market_regime       TEXT,
    sector_modifiers    TEXT
);

CREATE TABLE IF NOT EXISTS score_performance (
    id              INTEGER PRIMARY KEY,
    symbol          TEXT NOT NULL,
    score_date      TEXT NOT NULL,
    score           REAL NOT NULL,
    price_on_date   REAL,
    price_10d       REAL,
    price_30d       REAL,
    spy_10d_return  REAL,
    spy_30d_return  REAL,
    return_10d      REAL,
    return_30d      REAL,
    beat_spy_10d    INTEGER,
    beat_spy_30d    INTEGER,
    eval_date_10d   TEXT,
    eval_date_30d   TEXT,
    UNIQUE(symbol, score_date)
);

CREATE TABLE IF NOT EXISTS news_article_hashes (
    id          INTEGER PRIMARY KEY,
    symbol      TEXT NOT NULL,
    article_hash TEXT NOT NULL,
    first_seen  TEXT NOT NULL,
    mention_count INTEGER NOT NULL DEFAULT 1,
    UNIQUE(symbol, article_hash)
);

CREATE TABLE IF NOT EXISTS predictor_outcomes (
    id                      INTEGER PRIMARY KEY,
    symbol                  TEXT NOT NULL,
    prediction_date         TEXT NOT NULL,
    predicted_direction     TEXT,
    prediction_confidence   REAL,
    p_up                    REAL,
    p_flat                  REAL,
    p_down                  REAL,
    score_modifier_applied  REAL DEFAULT 0.0,
    actual_5d_return        REAL,
    correct_5d              INTEGER,
    actual_log_alpha        REAL,
    horizon_days            INTEGER,
    correct                 INTEGER,
    UNIQUE(symbol, prediction_date)
);

CREATE TABLE IF NOT EXISTS population (
    id                  INTEGER PRIMARY KEY,
    symbol              TEXT NOT NULL UNIQUE,
    sector              TEXT NOT NULL,
    long_term_score     REAL,
    long_term_rating    TEXT,
    conviction          TEXT DEFAULT 'stable',
    price_target_upside REAL,
    thesis_summary      TEXT,
    entry_date          TEXT,
    tenure_weeks        INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS population_history (
    id          INTEGER PRIMARY KEY,
    date        TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    ticker_in   TEXT,
    ticker_out  TEXT,
    sector      TEXT,
    reason      TEXT,
    score_in    REAL,
    score_out   REAL
);

CREATE TABLE IF NOT EXISTS stock_archive (
    id              INTEGER PRIMARY KEY,
    ticker          TEXT NOT NULL UNIQUE,
    sector          TEXT NOT NULL,
    sector_team     TEXT NOT NULL,
    first_analyzed  TEXT NOT NULL,
    last_analyzed   TEXT NOT NULL,
    times_in_population INTEGER DEFAULT 0,
    current_status  TEXT DEFAULT 'inactive'
);

CREATE TABLE IF NOT EXISTS thesis_history (
    id              INTEGER PRIMARY KEY,
    ticker          TEXT NOT NULL,
    run_date        TEXT NOT NULL,
    author          TEXT NOT NULL,
    thesis_type     TEXT NOT NULL,
    bull_case       TEXT,
    bear_case       TEXT,
    catalysts       TEXT,
    risks           TEXT,
    conviction      INTEGER,
    score           REAL,
    rationale       TEXT
);

CREATE TABLE IF NOT EXISTS analyst_resources (
    id              INTEGER PRIMARY KEY,
    ticker          TEXT NOT NULL,
    run_date        TEXT NOT NULL,
    agent           TEXT NOT NULL,
    resource_type   TEXT NOT NULL,
    resource_detail TEXT,
    influence       TEXT DEFAULT 'supporting'
);

CREATE TABLE IF NOT EXISTS memory_episodes (
    id              INTEGER PRIMARY KEY,
    ticker          TEXT NOT NULL,
    signal_date     TEXT NOT NULL,
    score           REAL,
    rating          TEXT,
    conviction      TEXT,
    thesis_summary  TEXT,
    outcome_10d     REAL,
    outcome_vs_spy  REAL,
    lesson          TEXT,
    sector          TEXT,
    pattern_tags    TEXT,
    created_date    TEXT NOT NULL,
    UNIQUE(ticker, signal_date)
);

CREATE TABLE IF NOT EXISTS memory_semantic (
    id              INTEGER PRIMARY KEY,
    category        TEXT NOT NULL,
    source          TEXT NOT NULL,
    content         TEXT NOT NULL,
    sector          TEXT,
    related_tickers TEXT,
    created_date    TEXT NOT NULL,
    reinforced_date TEXT
);

CREATE TABLE IF NOT EXISTS scanner_evaluations (
    id                  INTEGER PRIMARY KEY,
    ticker              TEXT NOT NULL,
    eval_date           TEXT NOT NULL,
    sector              TEXT,
    tech_score          REAL,
    scan_path           TEXT,
    quant_filter_pass   INTEGER NOT NULL DEFAULT 0,
    liquidity_pass      INTEGER NOT NULL DEFAULT 1,
    volatility_pass     INTEGER NOT NULL DEFAULT 1,
    balance_sheet_pass  INTEGER NOT NULL DEFAULT 1,
    filter_fail_reason  TEXT,
    rsi_14              REAL,
    atr_pct             REAL,
    price_vs_ma200      REAL,
    current_price       REAL,
    avg_volume_20d      REAL,
    UNIQUE(ticker, eval_date)
);

CREATE TABLE IF NOT EXISTS team_candidates (
    id                  INTEGER PRIMARY KEY,
    ticker              TEXT NOT NULL,
    eval_date           TEXT NOT NULL,
    team_id             TEXT NOT NULL,
    quant_rank          INTEGER,
    quant_score         REAL,
    qual_score          REAL,
    team_recommended    INTEGER NOT NULL DEFAULT 0,
    UNIQUE(ticker, eval_date, team_id)
);

CREATE TABLE IF NOT EXISTS cio_evaluations (
    id                  INTEGER PRIMARY KEY,
    ticker              TEXT NOT NULL,
    eval_date           TEXT NOT NULL,
    team_id             TEXT,
    quant_score         REAL,
    qual_score          REAL,
    combined_score      REAL,
    macro_shift         REAL,
    final_score         REAL,
    cio_decision        TEXT NOT NULL,
    cio_conviction      INTEGER,
    cio_rank            INTEGER,
    rationale           TEXT,
    UNIQUE(ticker, eval_date)
);
"""

# ── Versioned Migrations ─────────────────────────────────────────────────────
#
# Each entry: version -> (description, SQL statement)
# Versions are applied in order. Never reorder or renumber existing entries.
# To add a new migration: append with the next version number and bump
# SCHEMA_VERSION at the top of this file.

MIGRATIONS: dict[int, tuple[str, str]] = {
    1: ("Add conviction to investment_thesis",
        "ALTER TABLE investment_thesis ADD COLUMN conviction TEXT"),
    2: ("Add signal to investment_thesis",
        "ALTER TABLE investment_thesis ADD COLUMN signal TEXT"),
    3: ("Add score_velocity_5d to investment_thesis",
        "ALTER TABLE investment_thesis ADD COLUMN score_velocity_5d REAL"),
    4: ("Add price_target_upside to investment_thesis",
        "ALTER TABLE investment_thesis ADD COLUMN price_target_upside REAL"),
    5: ("Add sector_ratings to macro_snapshots",
        "ALTER TABLE macro_snapshots ADD COLUMN sector_ratings TEXT"),
    6: ("Add predicted_direction to investment_thesis",
        "ALTER TABLE investment_thesis ADD COLUMN predicted_direction TEXT"),
    7: ("Add prediction_confidence to investment_thesis",
        "ALTER TABLE investment_thesis ADD COLUMN prediction_confidence REAL"),
    8: ("Add evaluation tables (scanner_evaluations, team_candidates, cio_evaluations)",
        "SELECT 1"),  # Tables created via CREATE IF NOT EXISTS above
    9: ("Add indexes for common query patterns",
        """
        CREATE INDEX IF NOT EXISTS idx_thesis_symbol_date ON investment_thesis(symbol, date);
        CREATE INDEX IF NOT EXISTS idx_score_perf_date ON score_performance(score_date);
        CREATE INDEX IF NOT EXISTS idx_scanner_eval_date ON scanner_evaluations(eval_date);
        CREATE INDEX IF NOT EXISTS idx_population_hist_date ON population_history(date);
        CREATE INDEX IF NOT EXISTS idx_agent_reports_symbol_date ON agent_reports(symbol, date);
        CREATE INDEX IF NOT EXISTS idx_macro_date ON macro_snapshots(date);
        """),
    10: ("Add quant_score/qual_score columns to investment_thesis",
         """
         ALTER TABLE investment_thesis ADD COLUMN quant_score REAL;
         ALTER TABLE investment_thesis ADD COLUMN qual_score REAL;
         """),
    11: ("Add quant_score/qual_score columns to scanner_appearances",
         """
         ALTER TABLE scanner_appearances ADD COLUMN quant_score REAL;
         ALTER TABLE scanner_appearances ADD COLUMN qual_score REAL;
         """),
    # Calibrator-v1 preliminaries: enrich score_performance with the
    # per-row context the v1 GBM upgrade in research_calibrator.py:5
    # is documented to need. v0 (today) is a bucket lookup keyed by
    # final score alone. v1 (queued behind enforce-flip + corpus depth)
    # is GBM on score + sub-scores + conviction + regime context.
    # Shipping the schema NOW means every Saturday going forward
    # enriches the labeled training set; v1 trains against rich rows
    # when its gate opens. Companion producer wire-up at
    # scoring/performance_tracker.py:record_new_buy_scores; companion
    # backfill from archived signals.json shipped separately.
    12: ("Add per-row context to score_performance for calibrator-v1",
         """
         ALTER TABLE score_performance ADD COLUMN quant_score REAL;
         ALTER TABLE score_performance ADD COLUMN qual_score REAL;
         ALTER TABLE score_performance ADD COLUMN conviction TEXT;
         ALTER TABLE score_performance ADD COLUMN sector_modifier REAL;
         ALTER TABLE score_performance ADD COLUMN market_regime TEXT;
         """),
    # Predictor 21d canonical-alpha migration (2026-05-09; plan at
    # alpha-engine-docs/private/predictor-21d-migration-260509.md). Aligns
    # the measurement substrate with the predictor's canonical 21d
    # log-domain training target shipped in alpha-engine-predictor #114
    # (Track A canonical-label cutover). Column names are horizon-agnostic
    # by design — `horizon_days` records the row's horizon-of-record so
    # a future flip (21d → 60d) becomes a `cfg.FORWARD_DAYS` change, not
    # another schema migration.
    #
    # Old `actual_5d_return` and `correct_5d` retained for historical
    # reads + transition parity. Backtester analytics use
    # `COALESCE(actual_log_alpha, actual_5d_return)` until parallel-write
    # window closes (~4 weeks) and the legacy columns are retired.
    13: ("Add horizon-agnostic predictor outcome columns",
         """
         ALTER TABLE predictor_outcomes ADD COLUMN actual_log_alpha REAL;
         ALTER TABLE predictor_outcomes ADD COLUMN horizon_days INTEGER;
         ALTER TABLE predictor_outcomes ADD COLUMN correct INTEGER;
         """),
}


# ── Schema Initialization ────────────────────────────────────────────────────

def ensure_schema(conn: sqlite3.Connection) -> None:
    """
    Create all tables and apply pending migrations with version tracking.

    Safe to call on every cold-start:
    - CREATE IF NOT EXISTS is idempotent for tables
    - Migrations are tracked in schema_version and only applied once
    - Duplicate-column errors are caught for backward compat with
      pre-versioning databases
    """
    # 1. Create base tables
    conn.executescript(TABLES_SQL)

    # 2. Create version tracking table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version     INTEGER PRIMARY KEY,
            description TEXT NOT NULL,
            applied_at  TEXT NOT NULL
        )
    """)

    # 3. Determine which migrations are already applied
    applied = {
        row[0]
        for row in conn.execute("SELECT version FROM schema_version").fetchall()
    }

    # 4. Apply pending migrations in order
    pending = sorted(v for v in MIGRATIONS if v not in applied)
    if pending:
        log.info("Applying %d schema migration(s): %s", len(pending), pending)

    for version in pending:
        desc, sql = MIGRATIONS[version]
        try:
            if ";" in sql.strip().rstrip(";"):
                conn.executescript(sql)
            else:
                conn.execute(sql)
        except sqlite3.OperationalError as e:
            # Column may already exist from pre-versioning era — safe to skip
            if "duplicate column" not in str(e).lower():
                log.error("Schema migration v%d failed: %s — %s", version, desc, e)
                raise
            log.debug("Migration v%d skipped (column already exists): %s", version, desc)

        # Record migration as applied
        conn.execute(
            "INSERT INTO schema_version (version, description, applied_at) VALUES (?, ?, ?)",
            (version, desc, datetime.now(timezone.utc).isoformat()),
        )

    conn.commit()
    if pending:
        log.info("Schema up to date (version %d)", SCHEMA_VERSION)
