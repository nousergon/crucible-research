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


class TestExperimentRecordWiring:
    """alpha-engine-config#3077 Phase C: experiment_record emission rides
    alongside the shadow-signal write, per producer, FAIL-SOFT — isolated
    from the FAIL-HARD ChallengerShadowGapError doctrine above."""

    def test_writes_experiment_record_for_each_successful_producer(self, monkeypatch):
        def build(run_date, am, *, run_time="", population=None):
            return {"date": run_date}

        specs = [ProducerSpec("no_agent_quant", "challenger", "v1", "ok", build)]
        monkeypatch.setattr(runner, "challenger_producers", lambda: specs)
        monkeypatch.setattr(runner, "publish_observe_alert", lambda message, **kw: True)

        am = MagicMock()
        am.write_shadow_signals_json.side_effect = (
            lambda name, rd, ga, payload: f"signals_shadow/{name}/{rd}/signals.json"
        )
        runner.run_challengers(am, "2026-06-19")

        # 2 _s3_put calls (dated + latest) for the one challenger's record.
        record_puts = [c for c in am._s3_put.call_args_list
                       if c.args[0].startswith("experiments/no_agent_quant/records/")]
        assert len(record_puts) == 2
        keys = {c.args[0] for c in record_puts}
        assert keys == {
            "experiments/no_agent_quant/records/2026-06-19.json",
            "experiments/no_agent_quant/records/latest.json",
        }
        import json
        dated_body = json.loads(
            next(c.args[1] for c in record_puts if c.args[0].endswith("2026-06-19.json"))
        )
        assert dated_body["experiment_id"] == "no-agent-quant"
        assert dated_body["status"] == "complete"

    def test_failed_producer_still_gets_a_failed_experiment_record(self, monkeypatch):
        # config#1683's gap still RAISES after the loop — but the failing
        # producer's OWN experiment_record (status="failed") is still
        # written during its own loop iteration, before the aggregate raise.
        def bad_build(run_date, am, *, run_time="", population=None):
            raise RuntimeError("boom")

        specs = [ProducerSpec("bad", "challenger", "v1", "raises", bad_build)]
        monkeypatch.setattr(runner, "challenger_producers", lambda: specs)
        monkeypatch.setattr(runner, "publish_observe_alert", lambda message, **kw: True)

        am = MagicMock()
        with pytest.raises(runner.ChallengerShadowGapError):
            runner.run_challengers(am, "2026-06-19")

        record_puts = [c for c in am._s3_put.call_args_list
                       if c.args[0].startswith("experiments/bad/records/")]
        assert len(record_puts) == 2
        import json
        dated_body = json.loads(
            next(c.args[1] for c in record_puts if c.args[0].endswith("2026-06-19.json"))
        )
        assert dated_body["status"] == "failed"
        row = next(a for a in dated_body["artifacts"] if a["name"] == "shadow_signals")
        assert row["status"] == "absent"
        assert "boom" in row["reason"]

    def test_experiment_record_bug_does_not_touch_shadow_gap_doctrine(self, monkeypatch):
        # A bug in the NEW fail-soft record path must be fully isolated from
        # the existing FAIL-HARD shadow-signal gap logic: a successful
        # shadow-signal write must still count as written even if the
        # record-emission code raises.
        def good_build(run_date, am, *, run_time="", population=None):
            return {"date": run_date}

        specs = [ProducerSpec("good", "challenger", "v1", "ok", good_build)]
        monkeypatch.setattr(runner, "challenger_producers", lambda: specs)
        alerts = []
        monkeypatch.setattr(runner, "publish_observe_alert",
                            lambda message, **kw: alerts.append((message, kw)) or True)
        monkeypatch.setattr(
            runner, "build_challenger_experiment_record",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("record bug")),
        )

        am = MagicMock()
        am.write_shadow_signals_json.side_effect = lambda name, rd, ga, payload: "k"
        # No ChallengerShadowGapError — the shadow write itself succeeded;
        # only the fail-soft record path broke.
        res = runner.run_challengers(am, "2026-06-19")

        assert res["written"] == {"good": "k"}
        assert res["errors"] == {}
        # the record-path failure fired its OWN loud alert, distinct from
        # the shadow-gap alert (which never fires here — no gap occurred).
        assert len(alerts) == 1
        msg, kw = alerts[0]
        assert "experiment_record emission failed" in msg
        assert kw["source"] == "research:experiment_record"
        assert kw["dedup_key"] == "experiment_record_gap:good:2026-06-19"
