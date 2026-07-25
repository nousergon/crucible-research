"""config#2891 — consumer-side staleness assertion for config/scoring_weights.json
(scoring/aggregator.py's ``_load_weights_from_s3``, sharing config.py's
``check_s3_pointer_staleness`` WARN-only signal)."""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from io import BytesIO
from unittest.mock import MagicMock

from scoring import aggregator


def _make_s3_response(body: dict, last_modified) -> dict:
    return {"Body": BytesIO(json.dumps(body).encode()), "LastModified": last_modified}


def test_load_weights_from_s3_warns_on_stale_write(monkeypatch, caplog, tmp_path):
    """End-to-end: a simulated stale write on the real S3 read path logs the WARN."""
    monkeypatch.setattr(aggregator, "_WEIGHTS_CACHE_PATH", str(tmp_path / "scoring_weights_cache.json"))
    stale_time = datetime.now(UTC) - timedelta(days=30)
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = _make_s3_response(
        {"quant": 0.6, "qual": 0.4}, stale_time
    )
    mock_boto3 = MagicMock()
    mock_boto3.client.return_value = mock_s3
    monkeypatch.setitem(__import__("sys").modules, "boto3", mock_boto3)
    with caplog.at_level(logging.ERROR):
        aggregator._load_weights_from_s3()
    assert any("STALE config/scoring_weights.json" in r.message for r in caplog.records)


def test_load_weights_from_s3_silent_when_fresh(monkeypatch, caplog, tmp_path):
    monkeypatch.setattr(aggregator, "_WEIGHTS_CACHE_PATH", str(tmp_path / "scoring_weights_cache.json"))
    fresh = datetime.now(UTC) - timedelta(hours=1)
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = _make_s3_response(
        {"quant": 0.6, "qual": 0.4}, fresh
    )
    mock_boto3 = MagicMock()
    mock_boto3.client.return_value = mock_s3
    monkeypatch.setitem(__import__("sys").modules, "boto3", mock_boto3)
    with caplog.at_level(logging.ERROR):
        aggregator._load_weights_from_s3()
    assert not any("STALE" in r.message for r in caplog.records)
