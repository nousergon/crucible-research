"""Tests for research ops_alerts flow-doctor routing (config#1749)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ops_alerts import publish_ops_alert, publish_ops_digest


@patch("ops_alerts.get_flow_doctor", return_value=None)
def test_publish_ops_alert_sns_only_when_flow_doctor_inactive(_mock_fd):
    with patch("krepis.alerts.publish") as publish_mock:
        publish_ops_alert("gap", severity="WARN", source="test")
    publish_mock.assert_called_once()
    assert publish_mock.call_args.kwargs["telegram"] is False


@patch("ops_alerts.get_flow_doctor")
def test_publish_ops_alert_routes_telegram_via_flow_doctor(mock_get_fd):
    mock_fd = MagicMock()
    mock_fd.notify_event.return_value = "rid-1"
    mock_get_fd.return_value = mock_fd
    with patch("krepis.alerts.publish") as publish_mock:
        publish_ops_alert(
            "gap",
            severity="critical",
            source="research:alerts_handler",
            dedup_key="k1",
        )
    assert publish_mock.call_args.kwargs["telegram"] is False
    mock_fd.notify_event.assert_called_once()
    assert mock_fd.notify_event.call_args.kwargs["severity"] == "critical"


@patch("ops_alerts.get_flow_doctor", return_value=None)
def test_publish_ops_digest_noop_on_empty_findings(_mock_fd):
    assert publish_ops_digest([], source="test") is True


@patch("ops_alerts.get_flow_doctor", return_value=None)
def test_publish_ops_digest_false_when_flow_doctor_inactive(_mock_fd):
    assert publish_ops_digest(["drift"], source="test") is False


@patch("ops_alerts._telegram_notifier_for_topic")
@patch("ops_alerts.get_flow_doctor")
def test_publish_ops_digest_sends_silent_raw(mock_get_fd, mock_find_notifier):
    mock_notifier = MagicMock()
    mock_notifier.send_raw.return_value = "telegram:-100"
    mock_find_notifier.return_value = mock_notifier
    mock_get_fd.return_value = MagicMock()

    ok = publish_ops_digest(
        ["Universe drift: MSFT"],
        header="Surveillance Digest — 1 finding",
        source="research:alerts_handler",
    )

    assert ok is True
    mock_notifier.send_raw.assert_called_once()
    assert mock_notifier.send_raw.call_args.kwargs["disable_notification"] is True
    assert "Universe drift" in mock_notifier.send_raw.call_args.args[0]
