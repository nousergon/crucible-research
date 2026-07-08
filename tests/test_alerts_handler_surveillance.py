"""Tests for the surveillance Lambda (formerly intraday price alerts).

Locks behavior contracts after the L1067 rewrite (2026-05-13):
- Heartbeat staleness check fires before any other surveillance work
- Universe drift detection (research universe ⊄ daemon subscribed)
- Price-move alerts use S3 snapshot prices, not yfinance
- Findings emitted as silent Telegram rollup (digest tier)
- Daemon-down emitted as loud Telegram push (critical tier) + short-circuit

Pure-logic + mocked-S3/sqlite/telegram tests; no real boto3 or yfinance.
"""

from __future__ import annotations

import datetime
import json
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

# alerts_handler.py lives under lambda/ which is not a package; add to path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "lambda"))
sys.path.insert(0, str(_REPO_ROOT))

import alerts_handler as ah  # noqa: E402


# ── _heartbeat_stale ────────────────────────────────────────────────────────


class TestHeartbeatStale:
    def test_none_heartbeat_is_stale(self):
        is_stale, reason = ah._heartbeat_stale(None, stale_threshold_sec=180)
        assert is_stale is True
        assert "no heartbeat" in reason

    def test_missing_timestamp_field_is_stale(self):
        is_stale, reason = ah._heartbeat_stale({}, stale_threshold_sec=180)
        assert is_stale is True
        assert "missing timestamp" in reason

    def test_malformed_timestamp_is_stale(self):
        is_stale, reason = ah._heartbeat_stale(
            {"timestamp": "not-a-timestamp"}, stale_threshold_sec=180,
        )
        assert is_stale is True
        assert "malformed" in reason.lower()

    def test_fresh_heartbeat_is_not_stale(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        ts = now.isoformat().replace("+00:00", "") + "Z"
        is_stale, reason = ah._heartbeat_stale(
            {"timestamp": ts}, stale_threshold_sec=180,
        )
        assert is_stale is False
        assert reason == ""

    def test_old_heartbeat_is_stale(self):
        old = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=400)
        ts = old.isoformat().replace("+00:00", "") + "Z"
        is_stale, reason = ah._heartbeat_stale(
            {"timestamp": ts}, stale_threshold_sec=180,
        )
        assert is_stale is True
        assert "old" in reason

    def test_threshold_boundary(self):
        # 100s old, threshold 180s → not stale
        old = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=100)
        ts = old.isoformat().replace("+00:00", "") + "Z"
        is_stale, _ = ah._heartbeat_stale(
            {"timestamp": ts}, stale_threshold_sec=180,
        )
        assert is_stale is False


# ── _check_universe_drift ───────────────────────────────────────────────────


class TestUniverseDrift:
    def test_no_drift_returns_none(self):
        research = {"AAPL", "MSFT"}
        subscribed = ["AAPL", "MSFT", "SPY"]
        assert ah._check_universe_drift(research, subscribed) is None

    def test_research_superset_of_daemon_returns_finding(self):
        research = {"AAPL", "MSFT", "NVDA"}
        subscribed = ["AAPL", "SPY"]  # missing MSFT + NVDA
        finding = ah._check_universe_drift(research, subscribed)
        assert finding is not None
        assert "Universe drift" in finding
        assert "MSFT" in finding
        assert "NVDA" in finding

    def test_daemon_superset_of_research_returns_none(self):
        # Inverse drift (daemon has tickers research doesn't) is NOT flagged.
        # SPY + rotated-out positions stay subscribed legitimately.
        research = {"AAPL"}
        subscribed = ["AAPL", "SPY", "OLD_HOLDING"]
        assert ah._check_universe_drift(research, subscribed) is None

    def test_more_than_5_missing_truncates_with_more_suffix(self):
        research = {f"T{i}" for i in range(10)}
        subscribed = []
        finding = ah._check_universe_drift(research, subscribed)
        assert finding is not None
        assert "+5 more" in finding


# ── _check_price_moves ──────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_cooldown():
    ah._alerts_fired.clear()
    yield
    ah._alerts_fired.clear()


class TestPriceMoves:
    def test_empty_prices_returns_empty(self):
        result = ah._check_price_moves(None, {"AAPL": 150.0}, {"AAPL"})
        assert result == []

    def test_no_prices_field_returns_empty(self):
        result = ah._check_price_moves({}, {"AAPL": 150.0}, {"AAPL"})
        assert result == []

    def test_below_threshold_no_alert(self):
        latest = {"prices": {"AAPL": {"last": 151.0}}}
        result = ah._check_price_moves(latest, {"AAPL": 150.0}, {"AAPL"})
        # 0.67% move, below 5% default threshold
        assert result == []

    def test_above_threshold_fires(self):
        latest = {"prices": {"AAPL": {"last": 160.0}}}
        result = ah._check_price_moves(latest, {"AAPL": 150.0}, {"AAPL"})
        assert len(result) == 1
        assert "AAPL" in result[0]
        assert "+" in result[0]

    def test_negative_move_above_threshold(self):
        latest = {"prices": {"AAPL": {"last": 140.0}}}
        result = ah._check_price_moves(latest, {"AAPL": 150.0}, {"AAPL"})
        assert len(result) == 1
        assert "AAPL" in result[0]
        assert "-" in result[0]

    def test_ticker_outside_research_universe_skipped(self):
        # daemon's snapshot includes SPY but research universe doesn't.
        # SPY shouldn't generate a finding even on big moves.
        latest = {"prices": {"SPY": {"last": 480.0}, "AAPL": {"last": 160.0}}}
        result = ah._check_price_moves(latest, {"SPY": 400.0, "AAPL": 150.0}, {"AAPL"})
        assert len(result) == 1
        assert "SPY" not in result[0]

    def test_cooldown_prevents_repeat(self):
        latest = {"prices": {"AAPL": {"last": 160.0}}}
        first = ah._check_price_moves(latest, {"AAPL": 150.0}, {"AAPL"})
        assert len(first) == 1
        second = ah._check_price_moves(latest, {"AAPL": 150.0}, {"AAPL"})
        assert second == []

    def test_missing_prior_close_skipped(self):
        latest = {"prices": {"AAPL": {"last": 160.0}}}
        result = ah._check_price_moves(latest, {}, {"AAPL"})
        assert result == []

    def test_zero_prior_close_skipped(self):
        latest = {"prices": {"AAPL": {"last": 160.0}}}
        result = ah._check_price_moves(latest, {"AAPL": 0.0}, {"AAPL"})
        assert result == []


# ── _read_s3_json ───────────────────────────────────────────────────────────


class TestReadS3Json:
    def test_returns_parsed_json_on_success(self):
        s3 = MagicMock()
        body = MagicMock()
        body.read.return_value = b'{"foo": "bar"}'
        s3.get_object.return_value = {"Body": body}
        result = ah._read_s3_json(s3, "bucket", "key")
        assert result == {"foo": "bar"}

    def test_returns_none_on_no_such_key(self):
        s3 = MagicMock()
        s3.get_object.side_effect = ClientError(
            error_response={"Error": {"Code": "NoSuchKey"}},
            operation_name="GetObject",
        )
        assert ah._read_s3_json(s3, "bucket", "key") is None

    def test_returns_none_on_404(self):
        s3 = MagicMock()
        s3.get_object.side_effect = ClientError(
            error_response={"Error": {"Code": "404"}},
            operation_name="GetObject",
        )
        assert ah._read_s3_json(s3, "bucket", "key") is None

    def test_returns_none_on_invalid_json(self):
        s3 = MagicMock()
        body = MagicMock()
        body.read.return_value = b'not-json'
        s3.get_object.return_value = {"Body": body}
        assert ah._read_s3_json(s3, "bucket", "key") is None


# ── handler integration ────────────────────────────────────────────────────


@pytest.fixture
def fresh_heartbeat_payload():
    now = datetime.datetime.now(datetime.timezone.utc)
    return {
        "timestamp": now.isoformat().replace("+00:00", "") + "Z",
        "ib_connected": True,
        "daemon_pid": 12345,
        "subscribed_tickers": ["AAPL", "MSFT", "SPY"],
        "subscribed_count": 3,
    }


@pytest.fixture
def stale_heartbeat_payload():
    old = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=500)
    return {
        "timestamp": old.isoformat().replace("+00:00", "") + "Z",
        "ib_connected": True,
        "daemon_pid": 12345,
        "subscribed_tickers": ["AAPL"],
    }


@pytest.fixture
def seeded_research_db(tmp_path):
    """Seed a minimal research.db with population + active_candidates +
    technical_scores tables matching the alerts_handler reads.
    """
    db_path = tmp_path / "research.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE population (symbol TEXT, date TEXT);
        CREATE TABLE active_candidates (symbol TEXT);
        CREATE TABLE technical_scores (symbol TEXT, date TEXT, current_price REAL);
        INSERT INTO population (symbol, date) VALUES
            ('AAPL', '2026-05-12'),
            ('MSFT', '2026-05-12'),
            ('NVDA', '2026-05-12');
        INSERT INTO active_candidates (symbol) VALUES ('GOOG');
        INSERT INTO technical_scores (symbol, date, current_price) VALUES
            ('AAPL', '2026-05-12', 150.0),
            ('MSFT', '2026-05-12', 400.0),
            ('NVDA', '2026-05-12', 800.0),
            ('GOOG', '2026-05-12', 175.0);
    """)
    conn.commit()
    conn.close()
    return db_path


class TestHandlerIntegration:
    @patch("alerts_handler.is_market_open", return_value=False)
    def test_skips_when_market_closed(self, _mock_open):
        result = ah.handler({}, None)
        assert result == {"status": "SKIPPED", "reason": "market_closed"}

    @patch("alerts_handler.is_market_open", return_value=True)
    @patch("preflight.ResearchPreflight")
    @patch("ops_alerts.publish_ops_alert")
    @patch("alerts_handler.boto3.client")
    def test_stale_heartbeat_fires_critical_push_and_short_circuits(
        self,
        mock_boto3_client,
        mock_publish_ops_alert,
        _mock_preflight,
        _mock_open,
        stale_heartbeat_payload,
    ):
        s3 = MagicMock()
        mock_boto3_client.return_value = s3
        body = MagicMock()
        body.read.return_value = json.dumps(stale_heartbeat_payload).encode()
        s3.get_object.return_value = {"Body": body}

        result = ah.handler({}, None)

        assert result["status"] == "DAEMON_DOWN"
        mock_publish_ops_alert.assert_called_once()
        msg = mock_publish_ops_alert.call_args.args[0]
        assert mock_publish_ops_alert.call_args.kwargs["severity"] == "critical"
        assert "DAEMON DOWN" in msg
        # No call to research.db download — handler short-circuited.
        s3.download_file.assert_not_called()

    @patch("alerts_handler.is_market_open", return_value=True)
    @patch("preflight.ResearchPreflight")
    @patch("ops_alerts.publish_ops_alert")
    @patch("ops_alerts.publish_ops_digest")
    @patch("alerts_handler.boto3.client")
    def test_clean_run_emits_no_findings_when_universe_aligned(
        self,
        mock_boto3_client,
        mock_publish_ops_digest,
        mock_publish_ops_alert,
        _mock_preflight,
        _mock_open,
        fresh_heartbeat_payload,
        seeded_research_db,
        monkeypatch,
    ):
        # Heartbeat subscribed = research universe = {AAPL, MSFT, NVDA, GOOG, SPY}.
        # All prices flat → no findings → no rollup sent.
        fresh_heartbeat_payload["subscribed_tickers"] = [
            "AAPL", "MSFT", "NVDA", "GOOG", "SPY",
        ]
        latest_prices = {
            "timestamp": "2026-05-13T16:00:00Z",
            "prices": {
                "AAPL": {"last": 150.0},
                "MSFT": {"last": 400.0},
                "NVDA": {"last": 800.0},
                "GOOG": {"last": 175.0},
            },
        }
        s3 = _wire_s3_mock(
            mock_boto3_client,
            heartbeat=fresh_heartbeat_payload,
            latest_prices=latest_prices,
            seeded_db_src=seeded_research_db,
            monkeypatch=monkeypatch,
        )

        result = ah.handler({}, None)

        assert result["status"] == "OK"
        assert result["findings_count"] == 0
        mock_publish_ops_digest.assert_not_called()
        mock_publish_ops_alert.assert_not_called()

    @patch("alerts_handler.is_market_open", return_value=True)
    @patch("preflight.ResearchPreflight")
    @patch("ops_alerts.publish_ops_alert")
    @patch("ops_alerts.publish_ops_digest")
    @patch("alerts_handler.boto3.client")
    def test_universe_drift_emits_finding(
        self,
        mock_boto3_client,
        mock_publish_ops_digest,
        mock_publish_ops_alert,
        _mock_preflight,
        _mock_open,
        fresh_heartbeat_payload,
        seeded_research_db,
        monkeypatch,
    ):
        # Heartbeat subscribed = {AAPL, SPY} only; research has 4 tickers →
        # drift finding for MSFT, NVDA, GOOG.
        fresh_heartbeat_payload["subscribed_tickers"] = ["AAPL", "SPY"]
        latest_prices = {"timestamp": "now", "prices": {}}
        _wire_s3_mock(
            mock_boto3_client,
            heartbeat=fresh_heartbeat_payload,
            latest_prices=latest_prices,
            seeded_db_src=seeded_research_db,
            monkeypatch=monkeypatch,
        )

        result = ah.handler({}, None)

        assert result["status"] == "OK"
        assert result["findings_count"] >= 1
        mock_publish_ops_digest.assert_called_once()
        findings = mock_publish_ops_digest.call_args.args[0]
        assert any("Universe drift" in f for f in findings)
        mock_publish_ops_alert.assert_not_called()


def _wire_s3_mock(mock_boto3_client, *, heartbeat, latest_prices, seeded_db_src, monkeypatch):
    """Wire a mocked boto3 S3 client that:
    - Returns heartbeat for intraday/heartbeat.json get_object
    - Returns latest_prices for intraday/latest_prices.json get_object
    - Copies seeded_research_db into tempfile.gettempdir()/research_alerts.db on download_file
    """
    s3 = MagicMock()
    mock_boto3_client.return_value = s3

    def _get_object(Bucket, Key):
        if Key == ah.INTRADAY_HEARTBEAT_KEY:
            body = MagicMock()
            body.read.return_value = json.dumps(heartbeat).encode()
            return {"Body": body}
        if Key == ah.INTRADAY_LATEST_PRICES_KEY:
            body = MagicMock()
            body.read.return_value = json.dumps(latest_prices).encode()
            return {"Body": body}
        raise ClientError(
            error_response={"Error": {"Code": "NoSuchKey"}},
            operation_name="GetObject",
        )

    s3.get_object.side_effect = _get_object

    def _download_file(Bucket, Key, Filename):
        # Copy the seeded fixture db into the tempdir path the handler reads.
        import shutil
        shutil.copy(str(seeded_db_src), Filename)

    s3.download_file.side_effect = _download_file
    return s3
