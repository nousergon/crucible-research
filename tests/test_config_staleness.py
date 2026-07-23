"""config#2891 — consumer-side staleness assertion for the weekly-tuned S3
config pointers (config/research_params.json, config/scoring_weights.json).

Verifies the shared ``check_s3_pointer_staleness`` WARN-only signal fires
when a pointer's LastModified is older than the 2-weekly-cycle threshold,
and stays silent when fresh or absent — a defense-in-depth signal
(config#1724), never raising or blocking a load.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from io import BytesIO
from unittest.mock import MagicMock

import config
from config import WEEKLY_CONFIG_STALE_HOURS, check_s3_pointer_staleness


def _make_s3_response(body: dict, last_modified) -> dict:
    return {"Body": BytesIO(json.dumps(body).encode()), "LastModified": last_modified}


def test_check_s3_pointer_staleness_warns_past_threshold(caplog):
    old = datetime.now(UTC) - timedelta(hours=WEEKLY_CONFIG_STALE_HOURS + 1)
    with caplog.at_level(logging.ERROR):
        check_s3_pointer_staleness(old, "config/research_params.json")
    assert any("STALE config/research_params.json" in r.message for r in caplog.records)


def test_check_s3_pointer_staleness_silent_when_fresh(caplog):
    fresh = datetime.now(UTC) - timedelta(hours=1)
    with caplog.at_level(logging.ERROR):
        check_s3_pointer_staleness(fresh, "config/research_params.json")
    assert not any("STALE" in r.message for r in caplog.records)


def test_check_s3_pointer_staleness_silent_when_absent(caplog):
    with caplog.at_level(logging.ERROR):
        check_s3_pointer_staleness(None, "config/research_params.json")
    assert not any("STALE" in r.message for r in caplog.records)


def test_load_research_params_from_s3_warns_on_stale_write(monkeypatch, caplog, tmp_path):
    """End-to-end: a simulated stale write on the real S3 read path logs the WARN."""
    monkeypatch.setattr(config, "_RESEARCH_PARAMS_CACHE_PATH", str(tmp_path / "research_params_cache.json"))
    stale_time = datetime.now(UTC) - timedelta(days=30)
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = _make_s3_response(
        {"short_interest_buy_boost": 2.0}, stale_time
    )
    mock_boto3 = MagicMock()
    mock_boto3.client.return_value = mock_s3
    monkeypatch.setitem(__import__("sys").modules, "boto3", mock_boto3)
    with caplog.at_level(logging.ERROR):
        config._load_research_params_from_s3()
    assert any("STALE config/research_params.json" in r.message for r in caplog.records)
