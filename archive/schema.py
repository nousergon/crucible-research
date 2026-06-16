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

SCHEMA_VERSION = 19

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
    id                      INTEGER PRIMARY KEY,
    ticker                  TEXT NOT NULL,
    eval_date               TEXT NOT NULL,
    sector                  TEXT,
    tech_score              REAL,
    scan_path               TEXT,
    quant_filter_pass       INTEGER NOT NULL DEFAULT 0,
    liquidity_pass          INTEGER NOT NULL DEFAULT 1,
    volatility_pass         INTEGER NOT NULL DEFAULT 1,
    balance_sheet_pass      INTEGER NOT NULL DEFAULT 1,
    filter_fail_reason      TEXT,
    rsi_14                  REAL,
    atr_pct                 REAL,
    price_vs_ma200          REAL,
    current_price           REAL,
    avg_volume_20d          REAL,
    -- Focus list audit columns (v17 migration). Populated by archive_writer
    -- in shadow mode pre-cutover; agent contract change is gated behind
    -- factor-substrate Phase 2 + the FOCUS_LIST_GATING_ENABLED flip.
    focus_score             REAL,           -- regime-blended factor subscore (0-100)
    focus_stance            TEXT,           -- dominant factor: momentum/quality/value/low_vol
    focus_team_id           TEXT,           -- which sector team's focus list it falls in
    focus_rank_in_team      INTEGER,        -- 1-indexed rank within team focus list; NULL if not in any
    focus_rank_in_sector    INTEGER,        -- 1-indexed rank within sector
    focus_list_passed       INTEGER NOT NULL DEFAULT 0,  -- 1 if in any team's top-N
    agent_override          INTEGER NOT NULL DEFAULT 0,  -- 1 if @tool get_factor_profile called on this non-focus ticker
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
    rsi_sub_score       REAL,    -- 0-100, regime-aware mean-reversion signal
    macd_sub_score      REAL,    -- 0-100, MACD cross + above-zero state
    ma50_sub_score      REAL,    -- 0-100, price vs 50d MA
    ma200_sub_score     REAL,    -- 0-100, price vs 200d MA
    momentum_sub_score  REAL,    -- 0-100, 20d return percentile within universe
    UNIQUE(ticker, eval_date, team_id)
);

-- Scanner→team input-assignment ledger (v19). Records WHICH candidate each
-- sector team received and WHY, so the decision-review console can show the
-- complete input set per team — not just the names a team ended up ranking.
-- Without this the scanner→team partition is computed in-memory and discarded.
CREATE TABLE IF NOT EXISTS team_inputs (
    id              INTEGER PRIMARY KEY,
    ticker          TEXT NOT NULL,
    eval_date       TEXT NOT NULL,
    team_id         TEXT NOT NULL,
    source          TEXT,    -- 'scanner' (passed the weekly quant pre-filter) | 'held_population' (tracked stock)
    sector          TEXT,    -- the GICS sector that routed the ticker to this team
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
    rule_tags           TEXT,    -- JSON list[str] of closed-vocab tags; see migration 14
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
    # CIO rule-tag attribution. Pairs with prompt v1.3.0 + lib v0.7.0
    # rule_tags field on CIORawDecision. JSON-serialized list[str] of
    # closed-vocabulary tags (qual_veto, quant_veto, dual_score_floor,
    # rr_asymmetry, macro_alignment, portfolio_fit, catalyst_specificity,
    # prior_continuity, other) identifying which gating rule(s) drove
    # each decision. NULL on rows from prompts < v1.3.0 — backtester
    # analytics treat NULL as "untagged legacy" rather than coercing to
    # a default tag.
    14: ("Add rule_tags to cio_evaluations for per-decision attribution",
         "ALTER TABLE cio_evaluations ADD COLUMN rule_tags TEXT"),
    # Per-sub-signal scores in team_candidates. Surfaced from the
    # 2026-05-09 evaluator-email post-mortem on quant rank inversion in
    # healthcare/industrials/tech (corr(rank, 5d_ret) at +0.33-0.36).
    # Persisting the 5 sub-scores enables the backtester's
    # tech_weight_ablation optimizer (PR-C of this arc) to re-rank
    # historical team_candidates under alternate composite weights
    # without re-running the research pipeline. NULL on rows persisted
    # before the producer-side wire-up — backtester treats those as
    # "no sub-score data, ablation gates skip this row."
    15: ("Add per-sub-signal scores to team_candidates for ablation analysis",
         """
         ALTER TABLE team_candidates ADD COLUMN rsi_sub_score REAL;
         ALTER TABLE team_candidates ADD COLUMN macd_sub_score REAL;
         ALTER TABLE team_candidates ADD COLUMN ma50_sub_score REAL;
         ALTER TABLE team_candidates ADD COLUMN ma200_sub_score REAL;
         ALTER TABLE team_candidates ADD COLUMN momentum_sub_score REAL;
         """),
    # Stance taxonomy arc PR (2026-05-11) — denormalize the predictor's
    # stance label onto score_performance at write time (Kimball
    # dimensional pattern). Without this, the backtester's per-stance
    # attribution can only do compute-time joins with predictions.json
    # archive, which is fragile (S3 versioning, archive rotation) and
    # repeats the join every weekly run. Stamping stance on the fact
    # row creates a single source of truth + auditable history.
    #
    # NULL for rows scored before 2026-05-11 (no predictions.json
    # stance field existed). NULL also for rows where the predictor
    # didn't score the ticker (e.g., ticker outside the predictor's
    # population). Backtester's by_stance attribution treats NULL as
    # "no stance recorded" rather than coercing to a default label.
    16: ("Add stance column to score_performance for per-stance attribution",
         "ALTER TABLE score_performance ADD COLUMN stance TEXT"),
    # Focus list audit columns (PR 2 of the scanner-placement arc,
    # alpha-engine-docs/private/scanner-260514.md). Shadow-mode
    # observability: archive_writer populates these from the regime-
    # blended factor composite (Phase 1c + Phase 3 substrate) so the
    # weekly Saturday SF produces a real focus-list-vs-agent-pick
    # divergence audit without changing agent behavior. The
    # agent_override column is reserved for the PR 4 wiring of the
    # @tool get_factor_profile boundary; it stays 0 until then.
    # NULL on rows persisted before this migration — downstream
    # dashboards / backtester analytics treat NULL as "shadow logging
    # not yet active" rather than coercing to a default value.
    17: ("Add focus list audit columns to scanner_evaluations",
         """
         ALTER TABLE scanner_evaluations ADD COLUMN focus_score REAL;
         ALTER TABLE scanner_evaluations ADD COLUMN focus_stance TEXT;
         ALTER TABLE scanner_evaluations ADD COLUMN focus_team_id TEXT;
         ALTER TABLE scanner_evaluations ADD COLUMN focus_rank_in_team INTEGER;
         ALTER TABLE scanner_evaluations ADD COLUMN focus_rank_in_sector INTEGER;
         ALTER TABLE scanner_evaluations ADD COLUMN focus_list_passed INTEGER NOT NULL DEFAULT 0;
         ALTER TABLE scanner_evaluations ADD COLUMN agent_override INTEGER NOT NULL DEFAULT 0;
         """),
    # Canonical 21d horizon on score_performance (2026-05-29). Arithmetic
    # parity columns (price/return/spy/beat/eval_date_21d) plus the
    # canonical log-domain market-relative alpha (log_alpha_21d =
    # log_return_21d - log_spy_return_21d) the predictor trains on.
    # Producer = alpha-engine-data signal_returns._backfill_score_returns,
    # sourced from universe_returns' 21d columns (alpha-engine-data #197).
    # Powers the judge outcome-IC validation (ROADMAP L480 re-scope) — the
    # judge's quality scores correlated against realized canonical alpha.
    # NULL on rows persisted before this migration / before the 21d
    # forward window closes; backtester consumers treat NULL as
    # "outcome not yet realized".
    18: ("Add canonical 21d returns + log-alpha to score_performance",
         """
         ALTER TABLE score_performance ADD COLUMN price_21d REAL;
         ALTER TABLE score_performance ADD COLUMN return_21d REAL;
         ALTER TABLE score_performance ADD COLUMN spy_21d_return REAL;
         ALTER TABLE score_performance ADD COLUMN beat_spy_21d INTEGER;
         ALTER TABLE score_performance ADD COLUMN eval_date_21d TEXT;
         ALTER TABLE score_performance ADD COLUMN log_alpha_21d REAL;
         """),
    19: ("Add team_inputs ledger (scanner→team input-assignment audit)",
         "SELECT 1"),  # Table created via CREATE IF NOT EXISTS above
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
