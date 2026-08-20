"""The two artifacts `alpha-engine-config-I7856` REVIVED, asserted at the
repointed call site rather than at the helper.

Both had exactly one producer — an inline post-step in `lambda/handler.py`'s
champion research pass — and both lost it when `crucible-research-PR685`
deleted that pass. Neither absence raised anything, because both consumers are
graceful: the console Flow-Doctor pane just stops listing a flow, and the
Distillation-Corpus panel renders an explainer. That is the fleet's dominant
bug class (`principles.md` §2.7) — a component emitting nothing that no surface
renders as absent.

Measured 2026-08-20 before this change:

  * `s3://alpha-engine-research/_flow_doctor/heartbeat/` held `backtester`,
    `data-collector`, `executor`, `predictor-inference` — and NO `research`.
    Four of five producers reporting, the fifth silently missing from a
    dynamically-discovered list.
  * `decision_artifacts/distillation/corpus_stats/latest.json` frozen at
    2026-07-01, and `compute_corpus_stats` had zero invokers fleet-wide, while
    Think Tank kept writing `_sft_raw/` daily. The config#1542 kill-gate clock
    reads its trigger metric from that artifact.

These tests assert the WIRING. A unit test of `compute_corpus_stats` or of
`emit_heartbeat` would have passed every day of those seven weeks.
"""

from __future__ import annotations

import sys
import types

# ── Flow-doctor heartbeat, repointed onto signals_envelope_handler ────────


def _load(relpath, name):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _source(relpath: str) -> str:
    from pathlib import Path

    return (Path(__file__).resolve().parents[1] / relpath).read_text()


def test_signals_envelope_handler_emits_the_flow_doctor_heartbeat():
    """The live signals producer must call `emit_heartbeat`, or the research
    flow never appears on the console Flow-Doctor pane at all."""
    src = _source("lambda/signals_envelope_handler.py")
    assert "emit_heartbeat" in src, (
        "lambda/signals_envelope_handler.py no longer emits a flow-doctor "
        "heartbeat — the research producer disappears from the console's "
        "dynamically-discovered flow list, which renders as absent, not broken"
    )
    assert 'hasattr(fd, "emit_heartbeat")' in src, (
        "the hasattr guard is load-bearing: nousergon_lib deploys "
        "independently of this image and emit_heartbeat only exists in "
        "flow-doctor >=0.6.2, so a version-skewed pin would AttributeError at "
        "end-of-run on the success path"
    )


def test_the_heartbeat_is_gated_behind_preflight_like_the_health_stamp():
    """A heartbeat written off the Friday transport smoke is the same
    false-green the health stamp's `not preflight` gate exists to prevent."""
    src = _source("lambda/signals_envelope_handler.py")
    guard = src.index("if not preflight:")
    heartbeat = src.index("emit_heartbeat")
    assert guard < heartbeat, (
        "the flow-doctor heartbeat is emitted OUTSIDE the `if not preflight:` "
        "block — a preflight run would stamp the research flow fresh without "
        "having produced anything"
    )
    # ...and inside the production-target branch, not on the shadow path.
    production = src.index('if target == "production":')
    assert production < heartbeat


def test_the_heartbeat_write_is_fail_soft_but_never_silent():
    """signals.json is already persisted by this point; sinking the run on an
    observability write would trade the primary deliverable for the stamp.
    Fail-soft is only legitimate with a recording surface."""
    src = _source("lambda/signals_envelope_handler.py")
    block = src[src.index("emit_heartbeat"):]
    assert "publish_observe_alert" in block[:2000], (
        "the heartbeat write swallows its exception with no alert — a silent "
        "swallow on an observability path is how the absence went unnoticed "
        "for seven weeks in the first place"
    )
    assert "flow_doctor_heartbeat_write_fail" in src


# ── corpus_stats, repointed onto the weekly AggregateCosts stage ──────────


def test_aggregate_costs_handler_refreshes_the_corpus_stats_artifact():
    src = _source("lambda/aggregate_costs_handler.py")
    assert "_refresh_corpus_stats" in src
    assert "compute_corpus_stats" in src, (
        "scripts/corpus_stats.py::compute_corpus_stats has no invoker again — "
        "the config#1542 distill-or-shelve kill-gate clock reads a frozen "
        "artifact and can never auto-start"
    )


def test_corpus_stats_refresh_runs_and_reports_into_the_summary(monkeypatch):
    mod = _load("lambda/aggregate_costs_handler.py", "aggregate_costs_handler_under_test")

    calls = {}

    def _fake_compute(s3_client, bucket, target_date=None, write=True):
        calls["args"] = (bucket, target_date)
        return {"totals": {"deduped_pairs": 421}, "output_key": "k/latest.json"}

    fake = types.ModuleType("scripts.corpus_stats")
    fake.compute_corpus_stats = _fake_compute
    monkeypatch.setitem(sys.modules, "scripts.corpus_stats", fake)

    summary: dict = {}
    mod._refresh_corpus_stats(object(), "bkt", "2026-08-22", summary)

    assert calls["args"] == ("bkt", "2026-08-22")
    assert summary["corpus_stats"] == {
        "deduped_pairs": 421,
        "output_key": "k/latest.json",
    }


def test_corpus_stats_failure_never_costs_the_stage_its_parquet(monkeypatch):
    """The cost parquet is this stage's primary deliverable and is already
    written by the time the refresh runs."""
    mod = _load("lambda/aggregate_costs_handler.py", "aggregate_costs_handler_under_test2")

    fake = types.ModuleType("scripts.corpus_stats")

    def _boom(*a, **k):
        raise RuntimeError("s3 exploded")

    fake.compute_corpus_stats = _boom
    monkeypatch.setitem(sys.modules, "scripts.corpus_stats", fake)

    alerts: list[str] = []
    fake_alerts = types.ModuleType("observe_alerts")
    fake_alerts.publish_observe_alert = lambda msg, **k: alerts.append(msg)
    monkeypatch.setitem(sys.modules, "observe_alerts", fake_alerts)

    summary: dict = {}
    mod._refresh_corpus_stats(object(), "bkt", "2026-08-22", summary)  # must not raise

    assert "corpus_stats" not in summary
    assert alerts and "corpus_stats" in alerts[0], (
        "the refresh failed silently — fail-soft without a recording surface "
        "is the defect, not the mitigation"
    )


# ── The retirements, asserted so they cannot quietly come back ────────────


def test_the_dead_weekly_launcher_and_its_box_entrypoint_are_gone():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for rel in (
        "infrastructure/spot_research_weekly.sh",
        "infrastructure/weekly_box_runner.py",
        "data_manifest.py",
    ):
        assert not (root / rel).exists(), (
            f"{rel} is back. It drove the retired research graph "
            "(handler(weekly_run=True) raises RetiredResearchPathError) or, in "
            "data_manifest.py's case, had zero callers after PR685. Re-adding "
            "one needs its consumer named first."
        )
