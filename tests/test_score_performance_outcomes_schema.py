"""Authoritative research.db schema for the long-format outcome store
(EPIC config#1483 Phase 3a).

The score_performance_outcomes table is the root-cause replacement for the wide
horizon-suffixed score_performance columns. It is dual-written by the
alpha-engine-data producer (config#1483 Phase 2) and its authoritative DDL lives
here in archive/schema.py. These tests pin:
  - the table + indexes are created by ensure_schema()
  - SCHEMA_VERSION bumped to >= 21
  - the column set matches the nousergon_lib outcome_record contract's required
    fields (so a divergence between the schema and the contract is caught here,
    not at consumer-read time in Phase 3)
"""

from __future__ import annotations

import sqlite3

import pytest

from archive.schema import SCHEMA_VERSION, ensure_schema

_EXPECTED_COLUMNS = {
    "id",
    "signal_id",
    "symbol",
    "score_date",
    "horizon_days",
    "beat_spy",
    "stock_return",
    "spy_return",
    "log_alpha",
    "is_primary",
    "resolved_at",
    "schema_version",
}


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    ensure_schema(c)
    yield c
    c.close()


def test_schema_version_at_least_21():
    assert SCHEMA_VERSION >= 21


def test_table_exists_with_expected_columns(conn):
    cols = {r[1] for r in conn.execute("PRAGMA table_info(score_performance_outcomes)").fetchall()}
    assert cols == _EXPECTED_COLUMNS


def test_indexes_created(conn):
    idx = {r[1] for r in conn.execute("PRAGMA index_list(score_performance_outcomes)").fetchall()}
    # UNIQUE(signal_id, horizon_days) auto-index + the two query indexes.
    assert "idx_spo_horizon" in idx
    assert "idx_spo_score_date" in idx


def test_unique_signal_horizon(conn):
    conn.execute(
        "INSERT INTO score_performance_outcomes "
        "(signal_id, symbol, score_date, horizon_days, is_primary, resolved_at) "
        "VALUES ('AAPL:2026-03-02','AAPL','2026-03-02',21,1,'2026-04-01T00:00:00Z')"
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO score_performance_outcomes "
            "(signal_id, symbol, score_date, horizon_days, is_primary, resolved_at) "
            "VALUES ('AAPL:2026-03-02','AAPL','2026-03-02',21,1,'2026-04-01T00:00:00Z')"
        )


def test_columns_cover_contract_required_fields():
    """The table must carry every field the outcome_record contract requires,
    so a Phase-3 consumer reading a row can build a conforming record. Catches
    schema/contract drift here rather than at read time.

    Skips when the installed nousergon-lib predates the outcome_record contract
    (v0.77.0). Research is intentionally NOT repinned by this schema-only PR —
    the migration imports no lib code at runtime, and the repin lands with the
    Phase-3 consumer cutover. This guard activates automatically then.
    """
    contracts = pytest.importorskip("nousergon_lib.contracts")
    if "outcome_record" not in getattr(contracts, "CONTRACT_SCHEMAS", {}):
        pytest.skip("installed nousergon-lib predates the outcome_record contract (< v0.77.0)")
    schema = contracts.load_schema("outcome_record")
    required = set(schema["required"])
    # `symbol` is an additive denormalized convenience column not in the
    # contract; every contract-required field must exist as a table column.
    assert required <= _EXPECTED_COLUMNS
