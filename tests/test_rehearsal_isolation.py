"""alpha-engine-config-I12351: a weekly-SF rehearsal writes nothing to a live key.

Two halves, both pinned here:

- REHEARSAL (execution name ``rehearsal-*``): every S3 write is re-keyed under
  ``rehearsal/<name>/``. ``universe_membership/latest.json``, the dated
  membership partition, ``signals/latest.json`` and the dated signals
  partition are never written at their live keys. Staleness refusals degrade
  and are recorded.
- PRODUCTION (any other execution name, or none): writes land at the live keys
  exactly as before, and every staleness refusal still raises.

Real boto3 clients against moto, so the botocore hook is exercised for real.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from datetime import date
from pathlib import Path

import boto3
import pytest
from botocore.exceptions import ClientError
from moto import mock_aws

from data import rehearsal
from data.rehearsal import (
    RehearsalLiveWriteRefused,
    isolate,
    namespace_prefix,
    rehearsal_name,
    tolerate_stale,
)

BUCKET = "alpha-engine-research"
REHEARSAL = "rehearsal-2026-10-15-1"
ARN = f"arn:aws:states:us-east-1:711398986525:execution:ne-weekly-freshness-pipeline:{REHEARSAL}"
PROD_ARN = "arn:aws:states:us-east-1:711398986525:execution:ne-weekly-freshness-pipeline:a1b2c3d4"
NS = namespace_prefix(REHEARSAL)

LIVE_POINTERS = (
    "universe_membership/latest.json",
    "universe_membership/2026-10-15/membership.json",
    "signals/latest.json",
    "signals/2026-10-15/signals.json",
    "regime/latest.json",
    "scanner/universe/latest.json",
    "config/scanner_spec_champion.json",
)


@pytest.fixture
def s3(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.delenv("RUN_TOKEN", raising=False)
    monkeypatch.delenv("AE_PIPELINE_ROLE", raising=False)
    with mock_aws():
        # A fresh default session per test so a hook can never leak between
        # tests through boto3's module-level session.
        boto3.setup_default_session()
        client = boto3.client("s3")
        client.create_bucket(Bucket=BUCKET)
        yield client
    boto3.DEFAULT_SESSION = None


def _keys(client) -> set[str]:
    resp = client.list_objects_v2(Bucket=BUCKET)
    return {o["Key"] for o in resp.get("Contents", [])}


def _get(client, key) -> bytes:
    return client.get_object(Bucket=BUCKET, Key=key)["Body"].read()


# ── the marker ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "event,env,expected",
    [
        ({"execution_name": REHEARSAL}, {}, REHEARSAL),
        ({"execution_arn": ARN}, {}, REHEARSAL),
        ({"execution_id": ARN}, {}, REHEARSAL),
        ({}, {"RUN_TOKEN": REHEARSAL}, REHEARSAL),
        ({}, {"AE_PIPELINE_ROLE": "rehearsal"}, "rehearsal-manual-run"),
        ({"execution_name": "a1b2c3d4-weekly"}, {}, None),
        ({"execution_arn": PROD_ARN}, {}, None),
        ({"execution_name": "watch-rerun-2026-10-17-1"}, {}, None),
        ({}, {}, None),
        ({}, {"RUN_TOKEN": "spot-reference-20261017"}, None),
        # The event names the execution; a stray env token never overrides it.
        ({"execution_arn": PROD_ARN}, {"RUN_TOKEN": REHEARSAL}, None),
    ],
)
def test_rehearsal_name(event, env, expected):
    assert rehearsal_name(event, env) == expected


# ── isolation: rehearsal ───────────────────────────────────────────────────


def test_rehearsal_writes_nothing_to_a_live_key(s3):
    s3.put_object(Bucket=BUCKET, Key="universe_membership/latest.json", Body=b"PRODUCTION")
    before = _keys(s3)

    with isolate({"execution_name": REHEARSAL}, stage="Scanner") as ctx:
        assert ctx is not None
        c = boto3.client("s3")  # created inside, as every handler path does
        for key in LIVE_POINTERS:
            c.put_object(Bucket=BUCKET, Key=key, Body=b"REHEARSAL")

    after = _keys(s3)
    assert after - before == {f"{NS}{k}" for k in LIVE_POINTERS}
    assert _get(s3, "universe_membership/latest.json") == b"PRODUCTION"
    for key in LIVE_POINTERS:
        assert _get(s3, f"{NS}{key}") == b"REHEARSAL"
    assert ctx.summary()["writes_redirected"] == len(LIVE_POINTERS)


def test_rehearsal_reads_its_own_writes_and_live_inputs(s3):
    s3.put_object(Bucket=BUCKET, Key="universe_membership/latest.json", Body=b"PRODUCTION")
    s3.put_object(Bucket=BUCKET, Key="features/2026-10-15/technical.parquet", Body=b"LIVE-INPUT")

    with isolate({"execution_name": REHEARSAL}, stage="Scanner"):
        c = boto3.client("s3")
        # Before it writes, the rehearsal sees production's pointer (the
        # re-cut decision compares against it).
        assert _get(c, "universe_membership/latest.json") == b"PRODUCTION"
        c.put_object(Bucket=BUCKET, Key="universe_membership/latest.json", Body=b"REHEARSAL")
        # After, its read-back sees its own write.
        assert _get(c, "universe_membership/latest.json") == b"REHEARSAL"
        c.head_object(Bucket=BUCKET, Key="universe_membership/latest.json")
        assert _get(c, "features/2026-10-15/technical.parquet") == b"LIVE-INPUT"


def test_later_stage_reads_what_an_earlier_stage_of_the_same_rehearsal_wrote(s3):
    """SignalsEnvelope reads the board the rehearsal's Scanner wrote."""
    board = "scanner/universe/2026-10-15/universe.json"
    with isolate({"execution_name": REHEARSAL}, stage="Scanner"):
        boto3.client("s3").put_object(Bucket=BUCKET, Key=board, Body=b"BOARD")
    assert board not in _keys(s3)

    with isolate({"execution_name": REHEARSAL}, stage="SignalsEnvelope"):
        assert _get(boto3.client("s3"), board) == b"BOARD"

    # A different rehearsal does not see it.
    with isolate({"execution_name": "rehearsal-2026-10-16-1"}, stage="SignalsEnvelope"):
        with pytest.raises(ClientError, match="NoSuchKey"):
            boto3.client("s3").get_object(Bucket=BUCKET, Key=board)


def test_multipart_copy_and_batch_delete_are_rekeyed(s3):
    s3.put_object(Bucket=BUCKET, Key="signals/latest.json", Body=b"PRODUCTION")
    with isolate({"execution_name": REHEARSAL}, stage="Scanner"):
        c = boto3.client("s3")
        c.upload_fileobj(io.BytesIO(b"X" * 1024), BUCKET, "candidates/2026-10-15/candidates.json")
        c.copy_object(
            Bucket=BUCKET,
            Key="candidates/latest.json",
            CopySource={"Bucket": BUCKET, "Key": "candidates/2026-10-15/candidates.json"},
        )
        c.delete_objects(Bucket=BUCKET, Delete={"Objects": [{"Key": "signals/latest.json"}]})
        c.delete_object(Bucket=BUCKET, Key="signals/latest.json")
    keys = _keys(s3)
    assert "signals/latest.json" in keys  # the live pointer survived both deletes
    assert "candidates/2026-10-15/candidates.json" not in keys
    assert "candidates/latest.json" not in keys
    assert _get(s3, f"{NS}candidates/latest.json") == b"X" * 1024


def test_unknown_s3_write_is_refused_not_passed_through(s3):
    with isolate({"execution_name": REHEARSAL}, stage="Scanner"):
        c = boto3.client("s3")
        with pytest.raises(RehearsalLiveWriteRefused):
            c.put_bucket_tagging(Bucket=BUCKET, Tagging={"TagSet": []})


def test_preexisting_client_can_be_attached(s3):
    early = boto3.client("s3")  # created BEFORE isolation, so not covered by the session hook
    with isolate({"execution_name": REHEARSAL}, stage="Scanner") as ctx:
        ctx.attach(early)
        early.put_object(Bucket=BUCKET, Key="regime/latest.json", Body=b"R")
        # Attaching twice (or to a client that inherited the hook) is a no-op.
        ctx.attach(early)
        boto3.client("s3").put_object(Bucket=BUCKET, Key="signals/latest.json", Body=b"S")
    assert _keys(s3) == {f"{NS}regime/latest.json", f"{NS}signals/latest.json"}


# ── isolation: production unchanged ────────────────────────────────────────


@pytest.mark.parametrize(
    "event",
    [
        {"run_date": "2026-10-16"},
        {"run_date": "2026-10-16", "execution_name": "a1b2c3d4"},
        {"run_date": "2026-10-16", "execution_arn": PROD_ARN},
        {"run_date": "2026-10-16", "execution_name": "watch-rerun-2026-10-17-1"},
    ],
)
def test_production_writes_are_unchanged(s3, event):
    with isolate(event, stage="Scanner") as ctx:
        assert ctx is None
        c = boto3.client("s3")
        for key in LIVE_POINTERS:
            c.put_object(Bucket=BUCKET, Key=key, Body=b"PRODUCTION")
    assert _keys(s3) == set(LIVE_POINTERS)
    assert not any(k.startswith("rehearsal/") for k in _keys(s3))


def test_hook_is_inert_after_the_rehearsal_invocation(s3):
    """A warm container: a client created during a rehearsal and reused by a
    later production invocation must write live keys."""
    with isolate({"execution_name": REHEARSAL}, stage="Scanner"):
        cached = boto3.client("s3")
    with isolate({"execution_arn": PROD_ARN}, stage="Scanner"):
        cached.put_object(Bucket=BUCKET, Key="universe_membership/latest.json", Body=b"P")
        boto3.client("s3").put_object(Bucket=BUCKET, Key="signals/latest.json", Body=b"P")
    assert _keys(s3) == {"universe_membership/latest.json", "signals/latest.json"}
    assert rehearsal.active() is None


# ── the real write sites ───────────────────────────────────────────────────


def _membership() -> dict:
    return {"generated_at": "2026-10-16T02:27:00+00:00", "cuts": {}, "universe_count": 0}


def test_universe_membership_writer_under_rehearsal_and_production(s3):
    from scoring.universe_membership import read_latest_membership, write_universe_membership_to_s3

    s3.put_object(Bucket=BUCKET, Key="universe_membership/latest.json", Body=json.dumps({"run_date": "2026-10-09"}))

    with isolate({"execution_name": REHEARSAL}, stage="Scanner"):
        write_universe_membership_to_s3(_membership(), "2026-10-15", bucket=BUCKET)
        # The handler's read-back sees the rehearsal's own cut ...
        assert read_latest_membership(bucket=BUCKET)["generated_at"] == "2026-10-16T02:27:00+00:00"
    # ... and production's pointer and partitions are untouched.
    assert json.loads(_get(s3, "universe_membership/latest.json")) == {"run_date": "2026-10-09"}
    assert not [k for k in _keys(s3) if k.startswith("universe_membership/2026-10-15/")]

    write_universe_membership_to_s3(_membership(), "2026-10-16", bucket=BUCKET)
    assert json.loads(_get(s3, "universe_membership/latest.json"))["generated_at"] == "2026-10-16T02:27:00+00:00"
    assert "universe_membership/2026-10-16/membership.json" in _keys(s3)


def test_iso_week_recut_is_not_taken_by_a_rehearsal(s3):
    """The live pointer still carries the prior week after a rehearsal re-cut,
    so Saturday's production run decides the week's cut itself."""
    from scoring.universe_membership import read_latest_membership, should_recut, write_universe_membership_to_s3

    prior = {"run_date": "2026-10-09", "cut_effective_date": "2026-10-09"}
    s3.put_object(Bucket=BUCKET, Key="universe_membership/latest.json", Body=json.dumps(prior))
    with isolate({"execution_name": REHEARSAL}, stage="Scanner"):
        assert should_recut("2026-10-15", read_latest_membership(bucket=BUCKET), cadence="weekly") is True
        write_universe_membership_to_s3(
            {**_membership(), "run_date": "2026-10-15", "cut_effective_date": "2026-10-15"},
            "2026-10-15",
            bucket=BUCKET,
        )
    live = read_latest_membership(bucket=BUCKET)
    assert live == prior
    assert should_recut("2026-10-16", live, cadence="weekly") is True


def test_signals_envelope_writer_under_rehearsal_and_production(s3, monkeypatch):
    import scoring.promotion_guards as guards
    from scoring.signals_envelope import write_envelope

    monkeypatch.setattr(guards, "assert_promotable", lambda *a, **k: None)
    s3.put_object(Bucket=BUCKET, Key="signals/latest.json", Body=b"PRODUCTION")
    env = {"universe": [], "market_regime": "neutral"}

    with isolate({"execution_name": REHEARSAL}, stage="SignalsEnvelope"):
        dated, latest = write_envelope(env, "2026-10-15", target="production", bucket=BUCKET)
    assert _get(s3, "signals/latest.json") == b"PRODUCTION"
    assert dated not in _keys(s3)
    assert f"{NS}{dated}" in _keys(s3) and f"{NS}{latest}" in _keys(s3)

    write_envelope(env, "2026-10-16", target="production", bucket=BUCKET)
    assert json.loads(_get(s3, "signals/latest.json")) == env


# ── handler wiring ─────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.mark.parametrize(
    "rel,name",
    [
        ("lambda/scanner_handler.py", "_rehearsal_scanner_handler"),
        ("lambda/signals_envelope_handler.py", "_rehearsal_signals_handler"),
        # alpha-engine-config-I12351 (C3): rehearsal-2026-10-16-2's
        # AggregateCosts and RationaleClustering wrote live keys too.
        ("lambda/aggregate_costs_handler.py", "_rehearsal_aggregate_costs_handler"),
        ("lambda/rationale_clustering_handler.py", "_rehearsal_rationale_clustering_handler"),
    ],
)
def test_handlers_activate_isolation_before_run(s3, monkeypatch, rel, name):
    mod = _load(name, rel)

    def fake_run(event, context):
        boto3.client("s3").put_object(Bucket=BUCKET, Key="universe_membership/latest.json", Body=b"W")
        boto3.client("s3").put_object(Bucket=BUCKET, Key="signals/latest.json", Body=b"W")
        return {"status": "OK"}

    monkeypatch.setattr(mod, "_run", fake_run)

    out = mod.handler({"run_date": "2026-10-15", "execution_name": REHEARSAL}, None)
    assert _keys(s3) == {f"{NS}universe_membership/latest.json", f"{NS}signals/latest.json"}
    assert out["status"] == "OK"
    assert out["rehearsal"]["execution_name"] == REHEARSAL
    assert out["rehearsal"]["writes_redirected"] == 2

    out = mod.handler({"run_date": "2026-10-16", "execution_name": "a1b2c3d4"}, None)
    assert "rehearsal" not in out
    assert {"universe_membership/latest.json", "signals/latest.json"} <= _keys(s3)


def test_research_self_test_mode_is_isolated(s3, monkeypatch):
    """alpha-engine-config-I12351 (C3): rehearsal-2026-10-16-2's
    ResearchSelfTest wrote the LIVE ``research/2026-10-15/self_test.json`` and
    ``ops/checks/ae-research-self-test/latest.json``: a fresh, genuine verdict
    the freshness monitor and the console read as the week's."""
    mod = _load("_rehearsal_research_handler", "lambda/handler.py")
    live = ("research/2026-10-15/self_test.json", "ops/checks/ae-research-self-test/latest.json")

    def fake_emit(trading_date):
        for key in live:
            boto3.client("s3").put_object(Bucket=BUCKET, Key=key, Body=b"V")

    monkeypatch.setattr(mod, "_ensure_init", lambda: None)
    monkeypatch.setattr(mod, "_maybe_emit_self_test", fake_emit)
    monkeypatch.setattr(mod, "_resolve_self_test_date", lambda event: "2026-10-15")
    monkeypatch.setattr(mod, "_assert_self_test_coverage", lambda event, started: {})

    out = mod._run({"mode": "self_test", "date": "2026-10-15", "execution_name": REHEARSAL}, None)
    assert _keys(s3) == {f"{NS}{key}" for key in live}
    assert out["status"] == "OK"
    assert out["rehearsal"]["writes_redirected"] == 2

    out = mod._run({"mode": "self_test", "date": "2026-10-15", "execution_name": "a1b2c3d4"}, None)
    assert "rehearsal" not in out
    assert set(live) <= _keys(s3)


# ── stale data: degrade under rehearsal, refuse in production ──────────────


def test_tolerate_stale_is_false_outside_a_rehearsal():
    assert rehearsal.active() is None
    assert tolerate_stale("x", where="y") is False


def test_signals_board_fallback_staleness(s3):
    from scoring.signals_envelope import _assert_board_fallback_fresh

    stale = {"as_of": "2026-10-09"}
    with pytest.raises(RuntimeError, match="trading days stale"):
        _assert_board_fallback_fresh(stale, "2026-10-16")
    with isolate({"execution_arn": PROD_ARN}, stage="SignalsEnvelope"):
        with pytest.raises(RuntimeError, match="trading days stale"):
            _assert_board_fallback_fresh(stale, "2026-10-16")

    with isolate({"execution_name": REHEARSAL}, stage="SignalsEnvelope") as ctx:
        _assert_board_fallback_fresh(stale, "2026-10-16")  # degrades, no raise
        # An integrity fault is not a staleness refusal: it still raises.
        with pytest.raises(RuntimeError, match="no 'as_of'"):
            _assert_board_fallback_fresh({}, "2026-10-16")
    (d,) = ctx.summary()["degradations"]
    assert d["where"] == "signals_envelope.universe_board_fallback"
    assert "5 trading days stale" in d["refusal"]
    assert ctx.summary()["degraded"] is True


def test_feature_store_staleness(s3):
    import pandas as pd

    from data.fetchers.feature_store_reader import FeatureStoreStalenessError, _assert_fresh

    old = pd.Timestamp("2026-10-05")
    with pytest.raises(FeatureStoreStalenessError):
        _assert_fresh(old, date(2026, 10, 15), source="arcticdb", what="technical")
    with isolate({"execution_name": REHEARSAL}, stage="Scanner") as ctx:
        _assert_fresh(old, date(2026, 10, 15), source="arcticdb", what="technical")
    assert ctx.summary()["degradations"][0]["where"] == "feature_store_reader._assert_fresh[technical]"


def test_factor_snapshot_staleness(s3):
    import pandas as pd

    from scoring.factor_scoring import FactorSnapshotStalenessError, _resolve_group_snapshot

    buf = io.BytesIO()
    pd.DataFrame({"ticker": ["AAPL"], "x": [1.0]}).to_parquet(buf, engine="pyarrow")
    s3.put_object(Bucket=BUCKET, Key="features/2026-09-26/fundamental.parquet", Body=buf.getvalue())

    def resolve():
        return _resolve_group_snapshot(
            boto3.client("s3"),
            BUCKET,
            "fundamental",
            "2026-10-15",
            10,
            candidate_dates=["2026-09-26"],
        )

    with pytest.raises(FactorSnapshotStalenessError):
        resolve()
    with isolate({"execution_name": REHEARSAL}, stage="Scanner") as ctx:
        df, resolved = resolve()
    assert resolved == "2026-09-26" and len(df) == 1
    assert ctx.summary()["degradations"][0]["where"] == "factor_scoring._resolve_group_snapshot[fundamental]"
