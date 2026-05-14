"""Tests for the shadow focus-list audit wiring (PR 2 of the scanner-placement
arc, ``alpha-engine-docs/private/scanner-260514.md``).

Covers the v17 schema migration (scanner_evaluations focus_* columns) and
the archive_writer helper that projects per-ticker focus list audit fields
onto scanner_eval rows.
"""

import os
import sqlite3
import tempfile
from unittest.mock import MagicMock, patch

import pytest

ArchiveManager = pytest.importorskip(
    "archive.manager",
    reason="archive.manager requires gitignored config",
).ArchiveManager


@pytest.fixture
def archive_in_memory():
    """ArchiveManager with on-disk temp SQLite + mocked S3."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    manager = ArchiveManager(bucket="test-bucket", local_db_path=db_path)
    manager.s3 = MagicMock()
    manager.s3.get_object.side_effect = Exception("NoSuchKey")
    manager.s3.put_object = MagicMock()
    manager.s3.upload_file = MagicMock()
    manager.s3.download_file = MagicMock(side_effect=Exception("mock"))

    manager.db_conn = sqlite3.connect(db_path)
    manager.db_conn.row_factory = sqlite3.Row
    manager._ensure_schema()

    yield manager

    manager.close()
    os.unlink(db_path)


# ── Schema migration v17 ────────────────────────────────────────────────────


class TestSchemaV17:
    """v17 adds focus_* audit columns to scanner_evaluations."""

    def test_focus_columns_present(self, archive_in_memory):
        conn = archive_in_memory.db_conn
        cols = {
            r[1] for r in conn.execute(
                "PRAGMA table_info(scanner_evaluations)"
            ).fetchall()
        }
        expected_new = {
            "focus_score", "focus_stance", "focus_team_id",
            "focus_rank_in_team", "focus_rank_in_sector",
            "focus_list_passed", "agent_override",
        }
        assert expected_new.issubset(cols), f"missing: {expected_new - cols}"

    def test_focus_columns_have_correct_types(self, archive_in_memory):
        conn = archive_in_memory.db_conn
        col_types = {
            r[1]: r[2] for r in conn.execute(
                "PRAGMA table_info(scanner_evaluations)"
            ).fetchall()
        }
        assert col_types["focus_score"] == "REAL"
        assert col_types["focus_stance"] == "TEXT"
        assert col_types["focus_team_id"] == "TEXT"
        assert col_types["focus_rank_in_team"] == "INTEGER"
        assert col_types["focus_rank_in_sector"] == "INTEGER"
        assert col_types["focus_list_passed"] == "INTEGER"
        assert col_types["agent_override"] == "INTEGER"

    def test_v17_recorded_in_schema_version_table(self, archive_in_memory):
        conn = archive_in_memory.db_conn
        applied = {
            r[0] for r in conn.execute(
                "SELECT version FROM schema_version"
            ).fetchall()
        }
        assert 17 in applied

    def test_schema_version_constant_bumped(self):
        from archive.schema import SCHEMA_VERSION
        assert SCHEMA_VERSION >= 17


# ── write_scanner_evaluations with focus_* fields ───────────────────────────


class TestWriteScannerEvaluationsWithFocus:
    """archive.manager.ArchiveManager.write_scanner_evaluations honors
    the new focus_* fields when they're present in the evaluation dict."""

    def test_focus_fields_persisted(self, archive_in_memory):
        am = archive_in_memory
        am.write_scanner_evaluations([
            {
                "ticker": "NVDA",
                "eval_date": "2026-05-17",
                "sector": "Technology",
                "tech_score": 78.0,
                "quant_filter_pass": 1,
                "focus_score": 82.5,
                "focus_stance": "momentum",
                "focus_team_id": "technology",
                "focus_rank_in_team": 1,
                "focus_rank_in_sector": 1,
                "focus_list_passed": 1,
                "agent_override": 0,
            },
        ])
        row = am.db_conn.execute(
            "SELECT * FROM scanner_evaluations WHERE ticker='NVDA'"
        ).fetchone()
        assert row["focus_score"] == 82.5
        assert row["focus_stance"] == "momentum"
        assert row["focus_team_id"] == "technology"
        assert row["focus_rank_in_team"] == 1
        assert row["focus_rank_in_sector"] == 1
        assert row["focus_list_passed"] == 1
        assert row["agent_override"] == 0

    def test_missing_focus_fields_persist_as_null(self, archive_in_memory):
        """Evaluation dict without focus_* fields → NULL columns, not
        zero/empty-string. focus_list_passed + agent_override default 0."""
        am = archive_in_memory
        am.write_scanner_evaluations([
            {
                "ticker": "AAPL",
                "eval_date": "2026-05-17",
                "sector": "Technology",
                "tech_score": 75.0,
                "quant_filter_pass": 1,
            },
        ])
        row = am.db_conn.execute(
            "SELECT * FROM scanner_evaluations WHERE ticker='AAPL'"
        ).fetchone()
        assert row["focus_score"] is None
        assert row["focus_stance"] is None
        assert row["focus_team_id"] is None
        assert row["focus_rank_in_team"] is None
        assert row["focus_rank_in_sector"] is None
        assert row["focus_list_passed"] == 0  # NOT NULL DEFAULT 0
        assert row["agent_override"] == 0

    def test_near_miss_row_below_top_n(self, archive_in_memory):
        """Tickers scored but not in top-N → focus_score present,
        focus_rank_in_team NULL, focus_list_passed=0 (the 'near miss'
        observability path)."""
        am = archive_in_memory
        am.write_scanner_evaluations([
            {
                "ticker": "NEARMISS",
                "eval_date": "2026-05-17",
                "sector": "Technology",
                "focus_score": 45.0,
                "focus_stance": "value",
                "focus_team_id": "technology",
                "focus_rank_in_team": None,
                "focus_rank_in_sector": 27,
                "focus_list_passed": 0,
            },
        ])
        row = am.db_conn.execute(
            "SELECT * FROM scanner_evaluations WHERE ticker='NEARMISS'"
        ).fetchone()
        assert row["focus_score"] == 45.0
        assert row["focus_rank_in_team"] is None
        assert row["focus_rank_in_sector"] == 27
        assert row["focus_list_passed"] == 0


# ── _compute_focus_list_audit_lookup ────────────────────────────────────────


class TestComputeFocusListAuditLookup:
    """graph.research_graph._compute_focus_list_audit_lookup composes
    read_factor_profiles_from_s3 + compute_focus_scores + build_focus_list
    into the per-ticker audit dict archive_writer projects onto rows."""

    @pytest.fixture
    def sample_profiles(self):
        return {
            "NVDA": {
                "sector": "Technology", "quality_score": 70.0,
                "momentum_score": 95.0, "value_score": 20.0, "low_vol_score": 25.0,
            },
            "MSFT": {
                "sector": "Technology", "quality_score": 90.0,
                "momentum_score": 60.0, "value_score": 40.0, "low_vol_score": 55.0,
            },
            "JPM": {
                "sector": "Financials", "quality_score": 75.0,
                "momentum_score": 70.0, "value_score": 80.0, "low_vol_score": 50.0,
            },
        }

    def test_returns_empty_when_factor_blend_disabled(self):
        from graph import research_graph as rg
        with patch.object(rg, "FACTOR_BLEND_ENABLED", False):
            result = rg._compute_focus_list_audit_lookup(
                market_regime="bull", sector_map={},
            )
        assert result == {}

    def test_returns_empty_when_profile_artifact_missing(self):
        from graph import research_graph as rg
        with patch.object(rg, "FACTOR_BLEND_ENABLED", True), \
             patch.object(rg, "read_factor_profiles_from_s3", return_value=None):
            result = rg._compute_focus_list_audit_lookup(
                market_regime="bull", sector_map={},
            )
        assert result == {}

    def test_returns_empty_when_profile_artifact_empty(self):
        from graph import research_graph as rg
        with patch.object(rg, "FACTOR_BLEND_ENABLED", True), \
             patch.object(rg, "read_factor_profiles_from_s3", return_value={}):
            result = rg._compute_focus_list_audit_lookup(
                market_regime="bull", sector_map={},
            )
        assert result == {}

    def test_populates_lookup_for_focus_list_members(self, sample_profiles):
        from graph import research_graph as rg
        sector_map = {
            "NVDA": "Technology", "MSFT": "Technology", "JPM": "Financials",
        }
        with patch.object(rg, "FACTOR_BLEND_ENABLED", True), \
             patch.object(rg, "read_factor_profiles_from_s3",
                          return_value=sample_profiles):
            result = rg._compute_focus_list_audit_lookup(
                market_regime="bull", sector_map=sector_map,
            )
        # All 3 tickers in their respective team focus lists (under default
        # team size of 18, every scored ticker passes when sectors are small)
        for ticker in ("NVDA", "MSFT", "JPM"):
            assert ticker in result
            assert result[ticker]["focus_list_passed"] == 1
            assert result[ticker]["focus_rank_in_team"] is not None
            assert result[ticker]["focus_team_id"] is not None

    def test_bull_ordering_puts_momentum_first(self, sample_profiles):
        """In BULL the momentum-heavy ticker should rank above the
        quality-heavy ticker (sanity check the regime blend feeds through)."""
        from graph import research_graph as rg
        sector_map = {"NVDA": "Technology", "MSFT": "Technology"}
        with patch.object(rg, "FACTOR_BLEND_ENABLED", True), \
             patch.object(rg, "read_factor_profiles_from_s3",
                          return_value={
                              k: v for k, v in sample_profiles.items()
                              if k in ("NVDA", "MSFT")
                          }):
            result = rg._compute_focus_list_audit_lookup(
                market_regime="bull", sector_map=sector_map,
            )
        # NVDA (momentum=95) > MSFT (momentum=60) in BULL — NVDA ranks 1
        assert result["NVDA"]["focus_rank_in_team"] == 1
        assert result["MSFT"]["focus_rank_in_team"] == 2

    def test_stance_threaded_through(self, sample_profiles):
        from graph import research_graph as rg
        sector_map = {"NVDA": "Technology"}
        with patch.object(rg, "FACTOR_BLEND_ENABLED", True), \
             patch.object(rg, "read_factor_profiles_from_s3",
                          return_value={"NVDA": sample_profiles["NVDA"]}):
            result = rg._compute_focus_list_audit_lookup(
                market_regime="bull", sector_map=sector_map,
            )
        # NVDA's dominant factor is momentum_score=95 → stance "momentum"
        assert result["NVDA"]["focus_stance"] == "momentum"

    def test_unknown_sector_ticker_dropped(self, sample_profiles):
        """Ticker whose sector doesn't map to any team → not in lookup
        (rather than synthesizing a fake team_id)."""
        from graph import research_graph as rg
        profiles = {
            "ORPHAN": {
                # Invented label not present in SECTOR_TEAM_MAP — covers the
                # graceful-degrade path for profiles whose ``sector`` field
                # comes from an upstream feed with a label the team_config
                # mapping hasn't been taught yet.
                "sector": "Aerospace & Crypto Hybrids",
                "quality_score": 80.0, "momentum_score": 80.0,
                "value_score": 80.0, "low_vol_score": 80.0,
            },
        }
        with patch.object(rg, "FACTOR_BLEND_ENABLED", True), \
             patch.object(rg, "read_factor_profiles_from_s3",
                          return_value=profiles):
            result = rg._compute_focus_list_audit_lookup(
                market_regime="bull",
                sector_map={"ORPHAN": "Aerospace & Crypto Hybrids"},
            )
        assert "ORPHAN" not in result

    def test_near_miss_below_top_n(self):
        """When more candidates exist than the team focus-list cap, names
        below the cap end up in lookup with focus_list_passed=0 +
        focus_rank_in_team=None — the 'near miss' audit path."""
        from graph import research_graph as rg
        # Generate 25 Tech tickers — more than FOCUS_LIST_HARD_CAP=20
        profiles = {
            f"T{i:02d}": {
                "sector": "Technology",
                "quality_score": 50 + i,
                "momentum_score": 50 + i,
                "value_score": 50 + i,
                "low_vol_score": 30,
            }
            for i in range(25)
        }
        sector_map = {f"T{i:02d}": "Technology" for i in range(25)}
        with patch.object(rg, "FACTOR_BLEND_ENABLED", True), \
             patch.object(rg, "read_factor_profiles_from_s3",
                          return_value=profiles):
            result = rg._compute_focus_list_audit_lookup(
                market_regime="bull", sector_map=sector_map,
            )
        passed = [t for t, v in result.items() if v["focus_list_passed"] == 1]
        not_passed = [t for t, v in result.items() if v["focus_list_passed"] == 0]
        # Default focus list size is 18 (FOCUS_LIST_DEFAULT_SIZE)
        assert len(passed) == 18
        # Remaining scored-but-not-passed names carry focus_score with
        # focus_rank_in_team=None
        assert len(not_passed) == 25 - 18
        for ticker in not_passed:
            assert result[ticker]["focus_score"] is not None
            assert result[ticker]["focus_rank_in_team"] is None
            assert result[ticker]["focus_team_id"] == "technology"
