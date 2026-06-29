"""Tests for per-sector tool-usage analysis over analyst_resources (config#925).

Covers the round trip that was previously dead:
  1. ArchiveManager.save_analyst_resource → load_analyst_resources
  2. aggregate_tool_usage_by_sector pure-compute aggregation
"""

import os
import sqlite3
import tempfile

import pytest

ArchiveManager = pytest.importorskip(
    "archive.manager", reason="archive.manager requires gitignored config"
).ArchiveManager
from archive.tool_usage_analysis import (  # noqa: E402
    TEAM_RESOURCE_TICKER,
    aggregate_tool_usage_by_sector,
    sector_label_for_team,
    team_id_from_agent,
)
from unittest.mock import MagicMock  # noqa: E402


@pytest.fixture
def archive_in_memory():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    manager = ArchiveManager(bucket="test-bucket", local_db_path=db_path)
    manager.s3 = MagicMock()
    manager.db_conn = sqlite3.connect(db_path)
    manager.db_conn.row_factory = sqlite3.Row
    manager._ensure_schema()
    yield manager
    manager.close()
    os.unlink(db_path)


class TestLoadAnalystResources:
    def test_save_then_load_roundtrip(self, archive_in_memory):
        am = archive_in_memory
        am.save_analyst_resource(
            ticker=TEAM_RESOURCE_TICKER, run_date="2026-06-01",
            agent="team:technology", resource_type="price_history",
        )
        am.save_analyst_resource(
            ticker=TEAM_RESOURCE_TICKER, run_date="2026-06-08",
            agent="team:healthcare", resource_type="news_search",
        )
        rows = am.load_analyst_resources()
        assert len(rows) == 2
        assert {r["resource_type"] for r in rows} == {"price_history", "news_search"}

    def test_since_date_filter(self, archive_in_memory):
        am = archive_in_memory
        am.save_analyst_resource(
            ticker="*", run_date="2026-05-01", agent="team:technology",
            resource_type="old_tool",
        )
        am.save_analyst_resource(
            ticker="*", run_date="2026-06-08", agent="team:technology",
            resource_type="new_tool",
        )
        rows = am.load_analyst_resources(since_date="2026-06-01")
        assert [r["resource_type"] for r in rows] == ["new_tool"]

    def test_agent_prefix_filter(self, archive_in_memory):
        am = archive_in_memory
        am.save_analyst_resource(
            ticker="*", run_date="2026-06-08", agent="team:technology",
            resource_type="t1",
        )
        am.save_analyst_resource(
            ticker="*", run_date="2026-06-08", agent="macro",
            resource_type="t2",
        )
        rows = am.load_analyst_resources(agent_prefix="team:")
        assert [r["resource_type"] for r in rows] == ["t1"]

    def test_empty_when_no_conn(self, archive_in_memory):
        am = archive_in_memory
        am.db_conn = None
        assert am.load_analyst_resources() == []


class TestAggregateToolUsageBySector:
    def test_helpers(self):
        assert team_id_from_agent("team:technology") == "technology"
        assert team_id_from_agent("macro") == "macro"
        assert team_id_from_agent("") == "unknown"
        # sector label falls back to team_id when config unavailable / unknown
        assert sector_label_for_team("not_a_real_team") == "not_a_real_team"

    def test_aggregation_shape_and_counts(self):
        rows = [
            {"agent": "team:technology", "resource_type": "price_history"},
            {"agent": "team:technology", "resource_type": "price_history"},
            {"agent": "team:technology", "resource_type": "news_search"},
            {"agent": "team:healthcare", "resource_type": "filings_search"},
        ]
        out = aggregate_tool_usage_by_sector(rows)
        assert out["totals"] == {"n_rows": 4, "n_sectors": 2, "n_tools": 3}
        assert out["by_tool"]["price_history"] == 2

        tech = out["by_sector"][sector_label_for_team("technology")]
        assert tech["team_id"] == "technology"
        assert tech["total_calls"] == 3
        assert tech["tool_counts"]["price_history"] == 2
        assert tech["tool_share"]["price_history"] == pytest.approx(0.6667, abs=1e-3)
        assert tech["top_tool"] == "price_history"

    def test_unused_tools_by_sector(self):
        rows = [
            {"agent": "team:technology", "resource_type": "price_history"},
            {"agent": "team:healthcare", "resource_type": "filings_search"},
        ]
        out = aggregate_tool_usage_by_sector(rows)
        tech_sector = sector_label_for_team("technology")
        hc_sector = sector_label_for_team("healthcare")
        # technology never used filings_search; healthcare never used price_history
        assert out["unused_tools_by_sector"][tech_sector] == ["filings_search"]
        assert out["unused_tools_by_sector"][hc_sector] == ["price_history"]

    def test_ignores_rows_without_tool(self):
        rows = [{"agent": "team:technology", "resource_type": None},
                {"agent": "team:technology", "resource_type": ""}]
        out = aggregate_tool_usage_by_sector(rows)
        assert out["totals"]["n_rows"] == 0
        assert out["by_sector"] == {}

    def test_empty(self):
        out = aggregate_tool_usage_by_sector([])
        assert out["totals"] == {"n_rows": 0, "n_sectors": 0, "n_tools": 0}
        assert out["by_sector"] == {}
        assert out["unused_tools_by_sector"] == {}
