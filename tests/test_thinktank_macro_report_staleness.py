"""Instance 1 of the frozen-upstream bug class (alpha-engine-config-I2638).

``archive/macro/macro_report.md`` last changed 2026-03-16 — its producer,
``ArchiveManager.save_macro_report``, lost its only call site when the
multi-agent graph retired. Think Tank's daily shadow run kept reading it and
reconciling its themes against that five-month-old backdrop, and every surface
downstream — the theme artifacts, the run manifest, the producer leaderboard
scoring this arm — showed a healthy run.

Absence was already handled (WARN + ``sources_present`` false). These pin the
part that was missing: staleness is detected, alerted, recorded on the run
manifest, stamped on every theme written, and stated in the prompt the macro
model actually sees.

Think Tank DEGRADES rather than raises here on purpose: the daily shadow run
is non-blocking by contract, and what should replace the macro report as the
theme anchor is an unratified design fork (I2638). The fix makes the staleness
impossible to miss; it does not pick a new anchor.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import boto3
from moto import mock_aws

from thinktank.context import MACRO_REPORT_KEY, load_context
from thinktank.storage import ThinktankStore

BUCKET = "alpha-engine-research"


def _store() -> ThinktankStore:
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)
    return ThinktankStore(bucket=BUCKET, s3_client=s3)


def _load(store: ThinktankStore):
    """load_context with the RAG probes stubbed — irrelevant to freshness."""
    with (
        patch("nousergon_lib.rag.is_available", return_value=False),
    ):
        return load_context(store)


@mock_aws
def test_fresh_macro_report_is_recorded_as_fresh_and_does_not_alert(freshness_alerts):
    store = _store()
    store.s3.put_object(Bucket=BUCKET, Key=MACRO_REPORT_KEY, Body=b"# macro\nfresh")
    bundle = _load(store)

    assert bundle.sources_present["macro_report"] is True
    verdict = bundle.freshness["macro_report"]
    # A source that emits NO verdict is unobserved, not healthy — the fresh
    # case is recorded too, not just the broken one.
    assert verdict.is_fresh
    assert bundle.stale_sources() == []
    assert bundle.stale_input_records() == []
    assert freshness_alerts == []


@mock_aws
def test_stale_macro_report_degrades_loudly_without_halting_the_run(
    freshness_alerts, caplog
):
    """The bug: this read succeeded silently for five months."""
    store = _store()
    store.s3.put_object(Bucket=BUCKET, Key=MACRO_REPORT_KEY, Body=b"# macro\nMarch")
    stale = datetime.now(UTC) - timedelta(days=157)

    with patch.object(ThinktankStore, "last_modified", return_value=stale):
        bundle = _load(store)

    # The run continues — the shadow arm is non-blocking by contract.
    assert bundle.macro_report_md is not None
    assert bundle.sources_present["macro_report"] is True
    # ...but the staleness is now on three surfaces at once.
    verdict = bundle.freshness["macro_report"]
    assert verdict.status == "stale"
    assert verdict.age_days > 150
    assert verdict.degraded_reason
    assert bundle.stale_sources() == ["macro_report"]
    assert len(freshness_alerts) == 1
    assert freshness_alerts[0]["severity"] == "error"
    assert any(
        MACRO_REPORT_KEY in rec.message and rec.levelname == "ERROR"
        for rec in caplog.records
    )


@mock_aws
def test_missing_macro_report_is_undated_not_silently_fresh(freshness_alerts):
    store = _store()  # object never written
    bundle = _load(store)

    assert bundle.sources_present["macro_report"] is False
    assert bundle.freshness["macro_report"].status == "undated"
    assert bundle.stale_sources() == ["macro_report"]
    assert len(freshness_alerts) == 1


@mock_aws
def test_run_manifest_records_the_freshness_of_every_dated_input():
    """The durable record: an operator reading the manifest of a degraded run
    can see WHICH input was stale and how old it was."""
    from thinktank.schemas import RunManifest

    store = _store()
    store.s3.put_object(Bucket=BUCKET, Key=MACRO_REPORT_KEY, Body=b"# macro")
    stale = datetime.now(UTC) - timedelta(days=157)
    with patch.object(ThinktankStore, "last_modified", return_value=stale):
        bundle = _load(store)

    manifest = RunManifest(
        run_id="x", mode="daily", trading_day="2026-08-20",
        calendar_date="2026-08-20", started_at="2026-08-20T00:00:00Z",
    )
    manifest.context_source_freshness = bundle.freshness_records()
    manifest.degraded_inputs = bundle.stale_sources()

    assert manifest.degraded_inputs == ["macro_report"]
    record = manifest.context_source_freshness["macro_report"]
    assert record["artifact"] == MACRO_REPORT_KEY
    assert record["status"] == "stale"
    assert record["age_days"] > 150
    # Serializable: the manifest is written to S3 as JSON.
    assert "stale" in manifest.model_dump_json()


@mock_aws
def test_stale_anchor_is_visible_on_the_theme_prompt_and_the_theme_artifact():
    """A downstream reader of a Think Tank theme — including the producer
    leaderboard scoring this arm — must be able to tell a theme anchored to a
    five-month-old macro report from one anchored to last Saturday's."""
    from thinktank.themes import ThemeKeeper

    store = _store()
    store.s3.put_object(
        Bucket=BUCKET, Key=MACRO_REPORT_KEY, Body=b"# macro\nMarch conditions"
    )
    stale = datetime.now(UTC) - timedelta(days=157)
    with patch.object(ThinktankStore, "last_modified", return_value=stale):
        bundle = _load(store)

    keeper = ThemeKeeper(
        store, client=None, ctx=bundle,  # client unused by the paths exercised
        trading_day="2026-08-20", calendar_date="2026-08-20",
    )

    # (a) in-band: the model is TOLD the report is dated, in the prompt.
    block = keeper._macro_report_block()
    assert "STALE-INPUT" in block
    assert MACRO_REPORT_KEY in block
    assert "March conditions" in block  # the report itself is still passed

    # (b) out-of-band: every theme written carries the verdict.
    records = bundle.stale_input_records()
    assert [r["artifact"] for r in records] == [MACRO_REPORT_KEY]
    assert records[0]["status"] == "stale"


@mock_aws
def test_fresh_anchor_leaves_the_prompt_untouched():
    from thinktank.themes import ThemeKeeper

    store = _store()
    store.s3.put_object(Bucket=BUCKET, Key=MACRO_REPORT_KEY, Body=b"# macro\ncurrent")
    bundle = _load(store)
    keeper = ThemeKeeper(
        store, client=None, ctx=bundle,
        trading_day="2026-08-20", calendar_date="2026-08-20",
    )
    assert keeper._macro_report_block() == "# macro\ncurrent"
