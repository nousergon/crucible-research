"""Tests for the challenger producer runner (config#1223 B3 / config#1683) —
FAIL-HARD shadow emission: per-producer isolation (each producer gets its
attempt) but any gap raises ChallengerShadowGapError after the observe alert
fires. Prior-population threading unchanged."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import producers.runner as runner  # noqa: E402
from producers.registry import ProducerSpec  # noqa: E402


def test_run_challengers_isolates_attempts_then_raises_on_gap(monkeypatch):
    """config#1683 fail-hard: a failing producer does not starve the other
    producer's ATTEMPT (its artifact still lands), but the gap RAISES —
    experiments never silently thin out."""
    seen = {}

    def good_build(run_date, am, *, run_time="", population=None):
        seen["population"] = population
        seen["run_time"] = run_time
        return {"date": run_date, "signals": {}, "universe": [], "buy_candidates": [], "population": []}

    def bad_build(run_date, am, *, run_time="", population=None):
        raise RuntimeError("boom")

    specs = [
        ProducerSpec("good", "challenger", "v1", "ok", good_build),
        ProducerSpec("bad", "challenger", "v1", "raises", bad_build),
    ]
    monkeypatch.setattr(runner, "challenger_producers", lambda: specs)
    monkeypatch.setattr(runner, "publish_observe_alert", lambda message, **kw: True)

    am = MagicMock()
    am.write_shadow_signals_json.side_effect = (
        lambda name, rd, ga, payload: f"signals_shadow/{name}/{rd}/signals.json"
    )

    prior_pop = [{"ticker": "HELD"}]
    with pytest.raises(runner.ChallengerShadowGapError, match="bad"):
        runner.run_challengers(
            am, "2026-06-19", run_time="2026-06-19T09:00Z", population=prior_pop
        )

    # good's artifact was still written BEFORE the raise (isolation kept),
    # and the snapshotted prior population + run_time were threaded through.
    assert am.write_shadow_signals_json.call_count == 1
    assert seen["population"] is prior_pop
    assert seen["run_time"] == "2026-06-19T09:00Z"


def test_run_challengers_generated_at_falls_back_to_run_date(monkeypatch):
    captured = {}

    def build(run_date, am, *, run_time="", population=None):
        return {"date": run_date}

    monkeypatch.setattr(runner, "challenger_producers",
                        lambda: [ProducerSpec("p", "challenger", "v1", "", build)])
    am = MagicMock()
    am.write_shadow_signals_json.side_effect = lambda name, rd, ga, payload: captured.update(ga=ga) or "k"
    runner.run_challengers(am, "2026-06-19")  # no run_time
    assert captured["ga"] == "2026-06-19"


def test_run_challengers_alerts_loud_on_producer_gap(monkeypatch):
    """config#1403/#1683: a producer that emits nothing fires the LOUD observe
    alert BEFORE the gap raises (the alert pages even if a caller catches)."""
    def good_build(run_date, am, *, run_time="", population=None):
        return {"date": run_date}

    def bad_build(run_date, am, *, run_time="", population=None):
        raise RuntimeError("boom")

    specs = [
        ProducerSpec("good", "challenger", "v1", "ok", good_build),
        ProducerSpec("bad", "challenger", "v1", "raises", bad_build),
    ]
    monkeypatch.setattr(runner, "challenger_producers", lambda: specs)
    alerts = []
    monkeypatch.setattr(runner, "publish_observe_alert",
                        lambda message, **kw: alerts.append((message, kw)) or True)

    am = MagicMock()
    am.write_shadow_signals_json.side_effect = lambda name, rd, ga, payload: "k"
    with pytest.raises(runner.ChallengerShadowGapError):
        runner.run_challengers(am, "2026-06-19")

    assert len(alerts) == 1
    msg, kw = alerts[0]
    assert "challenger shadow gap" in msg and "bad" in msg
    assert kw["dedup_key"] == "challenger_shadow_gap:2026-06-19"
    assert kw["source"] == "research:challenger_producers"


def test_run_challengers_silent_when_all_emit(monkeypatch):
    """No gap → no alert (the alert must fire ONLY on a real always-on gap)."""
    def good_build(run_date, am, *, run_time="", population=None):
        return {"date": run_date}

    specs = [
        ProducerSpec("a", "challenger", "v1", "ok", good_build),
        ProducerSpec("b", "challenger", "v1", "ok", good_build),
    ]
    monkeypatch.setattr(runner, "challenger_producers", lambda: specs)
    alerts = []
    monkeypatch.setattr(runner, "publish_observe_alert",
                        lambda message, **kw: alerts.append(message) or True)

    am = MagicMock()
    am.write_shadow_signals_json.side_effect = lambda name, rd, ga, payload: "k"
    res = runner.run_challengers(am, "2026-06-19")

    assert res["written"] == {"a": "k", "b": "k"} and not res["errors"]
    assert alerts == []
