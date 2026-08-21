"""Think Tank's macro ANCHOR — the regime substrate + the news backdrop.

alpha-engine-config-I2638, Brian ruling 2026-08-21. Think Tank used to anchor
its macro theme on ``archive/macro/macro_report.md``, whose producer
(``ArchiveManager.save_macro_report``) lost its only call site when the
multi-agent graph retired. The object froze at 2026-03-16 and the daily shadow
run reconciled its themes against that backdrop for 158 days while every
surface downstream — the theme artifacts, the run manifest, the producer
leaderboard scoring this arm — showed a healthy run. The freshness primitive
made it visible; the ruling removes the source rather than resurrecting a
retired agent path.

These pin the replacement, in both directions:

- the anchor is the ``regime/`` substrate plus the daily news aggregates, and
  the retired Markdown report is not read at all;
- each leg is freshness-checked and each verdict reaches the same three
  surfaces the old one did — the ops alert, the run manifest, and the prompt
  the macro model actually sees.

The staleness posture is unchanged and deliberate: Think Tank DEGRADES rather
than raises, because the daily shadow run is non-blocking by contract.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from unittest.mock import patch

import boto3
import pytest
from moto import mock_aws

from thinktank.context import REGIME_SUBSTRATE_PREFIX, load_context
from thinktank.storage import ThinktankStore

BUCKET = "alpha-engine-research"
RETIRED_MACRO_REPORT_KEY = "archive/macro/macro_report.md"


def _substrate(*, calendar_date: str) -> dict:
    """A regime-substrate payload with the fields the renderer reads.

    Mirrors ``crucible-predictor``'s ``regime/substrate.py::build_regime_substrate``
    output shape — a consumer-side fixture of a cross-repo contract, so a
    producer-side rename shows up here as a failure rather than as a silently
    emptier prompt.
    """
    return {
        "calendar_date": calendar_date,
        "trading_day": calendar_date,
        "run_id": "2608150900",
        "schema_version": "1.0",
        "hmm": {
            "probs": {"bear": 0.11, "neutral": 0.22, "bull": 0.67},
            "argmax": "bull",
            "weeks_in_current_state": 5,
            "log_likelihood": -12.5,
        },
        "composite": {
            "intensity_z": 0.83,
            "per_feature_z": {"vix_level": -1.42, "hy_oas_bps": -0.30, "spy_20d_return": 0.91},
            "features_used": ["vix_level", "hy_oas_bps", "spy_20d_return"],
            "implied_severity": "calm",
        },
        "bocpd": {
            "change_signal": False,
            "max_runlength_prob": 0.74,
            "change_confidence": 0.08,
        },
        "guardrails": {
            "vix_caution_breached": False,
            "vix_bear_breached": False,
            "spy_30d_caution_breached": True,
            "spy_30d_bear_breached": False,
            "hy_oas_caution_breached": False,
            "active_severity_floor": None,
        },
        "features": {"vix_level": 14.2, "hy_oas_bps": 291.0, "spy_20d_return": 0.031},
    }


def _store() -> ThinktankStore:
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)
    return ThinktankStore(bucket=BUCKET, s3_client=s3)


def _put_substrate(store: ThinktankStore, *, calendar_date: str) -> dict:
    """Write a substrate artifact + its ``latest.json`` pointer sidecar, the
    same two objects the producer writes."""
    payload = _substrate(calendar_date=calendar_date)
    _write_substrate(store, payload)
    return payload


def _write_substrate(store: ThinktankStore, payload: dict) -> None:
    """The dated artifact plus the ``latest.json`` POINTER sidecar — the
    sidecar carries ``artifact_key``, it is not a copy of the payload
    (``nousergon_lib.eval_artifacts`` resolution contract)."""
    key = f"{REGIME_SUBSTRATE_PREFIX}/{payload['run_id']}.json"
    store.s3.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(payload).encode())
    store.s3.put_object(
        Bucket=BUCKET,
        Key=f"{REGIME_SUBSTRATE_PREFIX}/latest.json",
        Body=json.dumps({"artifact_key": key}).encode(),
    )


def _news_rows(*, aggregate_date: str) -> dict[str, dict]:
    return {
        "AAPL": {
            "ticker": "AAPL",
            "aggregate_date": aggregate_date,
            "n_articles": 12,
            "lm_sentiment_mean": 0.21,
            "lm_sentiment_trusted_mean": 0.18,
            "event_count": 2,
            "event_severity_max": 3.0,
            "event_categories": "guidance,product",
            "top_event_descriptions": "Raised FY guidance",
        },
        "XOM": {
            "ticker": "XOM",
            "aggregate_date": aggregate_date,
            "n_articles": 5,
            "lm_sentiment_mean": -0.34,
            "lm_sentiment_trusted_mean": -0.30,
            "event_count": 1,
            "event_severity_max": 7.5,
            "event_categories": "regulatory",
            "top_event_descriptions": "Regulator opens inquiry",
        },
    }


def _load(store: ThinktankStore, *, news: dict[str, dict] | None = None):
    """load_context with the RAG probe and the news substrate reader stubbed —
    neither is under test here."""
    with (
        patch("nousergon_lib.rag.is_available", return_value=False),
        patch("thinktank.context._load_news", return_value=news or {}),
    ):
        return load_context(store)


def _today(offset_days: int = 0) -> str:
    return (date.today() - timedelta(days=offset_days)).isoformat()


# ── the retired source is GONE, not merely stale ─────────────────────────────


def test_the_retired_macro_report_is_not_read_at_all():
    """The regression guard for the ruling itself: a re-introduced read would
    silently restore the frozen anchor, and nothing else in the suite would
    notice because a stale read still *works*."""
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent
    offenders: list[str] = []
    for path in sorted((src / "thinktank").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # Docstrings and comments may still NAME the retired key — the module
        # docstrings explain why it is gone, and deleting that history would
        # cost the next reader the reason. What must not exist is a live
        # string literal that could be handed to S3 as a key.
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(
                node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            )
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        }
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and RETIRED_MACRO_REPORT_KEY in node.value
                and id(node) not in docstrings
            ):
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == [], (
        f"{offenders} still carry {RETIRED_MACRO_REPORT_KEY} as a live string — "
        "Think Tank anchors on the regime substrate + news aggregates "
        "(alpha-engine-config-I2638 ruling)."
    )


@mock_aws
def test_context_bundle_no_longer_carries_the_macro_report():
    bundle = _load(_store())
    assert not hasattr(bundle, "macro_report_md")
    assert "macro_report" not in bundle.sources_present
    assert "macro_report" not in bundle.freshness


# ── freshness of each anchor leg ─────────────────────────────────────────────


@mock_aws
def test_fresh_anchor_is_recorded_as_fresh_and_does_not_alert(freshness_alerts):
    store = _store()
    _put_substrate(store, calendar_date=_today(2))
    bundle = _load(store, news=_news_rows(aggregate_date=_today(0)))

    assert bundle.sources_present["regime_substrate"] is True
    assert bundle.sources_present["news_aggregates"] is True
    # A source that emits NO verdict is unobserved, not healthy — the fresh
    # case is recorded too, not just the broken one.
    assert bundle.freshness["regime_substrate"].is_fresh
    assert bundle.freshness["news_aggregates"].is_fresh
    assert bundle.stale_sources() == []
    assert bundle.stale_input_records() == []
    assert freshness_alerts == []


@mock_aws
def test_a_substrate_frozen_for_months_degrades_loudly_without_halting(
    freshness_alerts, caplog
):
    """The original bug, re-posed against the new anchor: the read succeeds,
    the payload parses, and the run completes — so only the verdict can tell
    the operator the regime picture stopped moving in March."""
    store = _store()
    _put_substrate(store, calendar_date=_today(158))
    bundle = _load(store, news=_news_rows(aggregate_date=_today(0)))

    # The run continues — the shadow arm is non-blocking by contract.
    assert bundle.regime_substrate is not None
    assert bundle.sources_present["regime_substrate"] is True
    # ...but the staleness is on three surfaces at once.
    verdict = bundle.freshness["regime_substrate"]
    assert verdict.status == "stale"
    assert verdict.age_days > 150
    assert verdict.degraded_reason
    assert bundle.stale_sources() == ["regime_substrate"]
    assert len(freshness_alerts) == 1
    assert freshness_alerts[0]["severity"] == "error"
    assert any(
        f"{REGIME_SUBSTRATE_PREFIX}/" in rec.message and rec.levelname == "ERROR"
        for rec in caplog.records
    )


@mock_aws
def test_a_missing_substrate_is_undated_not_silently_fresh(freshness_alerts):
    store = _store()  # nothing written under regime/
    bundle = _load(store, news=_news_rows(aggregate_date=_today(0)))

    assert bundle.sources_present["regime_substrate"] is False
    assert bundle.freshness["regime_substrate"].status == "undated"
    assert bundle.stale_sources() == ["regime_substrate"]
    assert len(freshness_alerts) == 1


@mock_aws
def test_a_substrate_with_no_in_band_date_is_undated(freshness_alerts):
    """The as-of comes from the payload, never from the object's LastModified:
    a re-upload of an old payload must not read as fresh."""
    store = _store()
    payload = _substrate(calendar_date=_today(0))
    payload.pop("calendar_date")
    payload.pop("model_metadata", None)
    _write_substrate(store, payload)
    bundle = _load(store, news=_news_rows(aggregate_date=_today(0)))

    assert bundle.freshness["regime_substrate"].status == "undated"


@mock_aws
def test_an_empty_news_table_is_undated_not_calm(freshness_alerts):
    """The intraweek leg gets its own verdict: no rows is UNOBSERVED, and a
    macro anchor missing half of itself must not look healthy."""
    store = _store()
    _put_substrate(store, calendar_date=_today(2))
    bundle = _load(store, news={})

    assert bundle.sources_present["news_aggregates"] is False
    assert bundle.freshness["news_aggregates"].status == "undated"
    assert bundle.stale_sources() == ["news_aggregates"]


@mock_aws
def test_stale_news_aggregates_trip_the_daily_tolerance(freshness_alerts):
    store = _store()
    _put_substrate(store, calendar_date=_today(2))
    bundle = _load(store, news=_news_rows(aggregate_date=_today(9)))

    verdict = bundle.freshness["news_aggregates"]
    assert verdict.status == "stale"
    assert verdict.age_days > 8
    # The weekly leg is unaffected — the two legs are judged independently.
    assert bundle.freshness["regime_substrate"].is_fresh
    assert bundle.stale_sources() == ["news_aggregates"]


# ── the durable record ───────────────────────────────────────────────────────


@mock_aws
def test_run_manifest_records_the_freshness_of_every_dated_input():
    """An operator reading the manifest of a degraded run can see WHICH input
    was stale and how old it was."""
    from thinktank.schemas import RunManifest

    store = _store()
    _put_substrate(store, calendar_date=_today(158))
    bundle = _load(store, news=_news_rows(aggregate_date=_today(0)))

    manifest = RunManifest(
        run_id="x", mode="daily", trading_day="2026-08-20",
        calendar_date="2026-08-20", started_at="2026-08-20T00:00:00Z",
    )
    manifest.context_source_freshness = bundle.freshness_records()
    manifest.degraded_inputs = bundle.stale_sources()

    assert manifest.degraded_inputs == ["regime_substrate"]
    record = manifest.context_source_freshness["regime_substrate"]
    assert record["artifact"] == f"{REGIME_SUBSTRATE_PREFIX}/"
    assert record["status"] == "stale"
    assert record["age_days"] > 150
    # Both legs are recorded, fresh one included.
    assert manifest.context_source_freshness["news_aggregates"]["status"] == "fresh"
    # Serializable: the manifest is written to S3 as JSON.
    assert "stale" in manifest.model_dump_json()


# ── what the model is actually handed ────────────────────────────────────────


def _keeper(store, bundle):
    from thinktank.themes import ThemeKeeper

    return ThemeKeeper(
        store, client=None, ctx=bundle,  # client unused by the paths exercised
        trading_day="2026-08-20", calendar_date="2026-08-20",
    )


@mock_aws
def test_the_anchor_block_carries_the_substrate_and_the_news_backdrop():
    store = _store()
    _put_substrate(store, calendar_date=_today(2))
    bundle = _load(store, news=_news_rows(aggregate_date=_today(0)))

    block = _keeper(store, bundle)._macro_anchor_block()

    # Substrate leg: state, conviction, intensity, change-point, guardrails.
    assert "QUANTITATIVE REGIME SUBSTRATE" in block
    assert "bull" in block
    assert "5 week(s) in this state" in block
    assert "0.83" in block  # composite intensity_z
    assert "spy_30d_caution_breached" in block
    assert "vix_level" in block
    # News leg: coverage, sentiment, categories, the severe events.
    assert "NEWS BACKDROP" in block
    assert "2 name(s)" in block
    assert "17 article(s)" in block
    assert "Regulator opens inquiry" in block
    # Fresh on both legs ⇒ no banner.
    assert "STALE-INPUT" not in block


@mock_aws
def test_a_stale_leg_is_named_in_the_prompt_and_on_the_theme_artifact():
    """A downstream reader of a Think Tank theme — including the producer
    leaderboard scoring this arm — must be able to tell a theme anchored to a
    five-month-old regime read from one anchored to last Saturday's."""
    store = _store()
    _put_substrate(store, calendar_date=_today(158))
    bundle = _load(store, news=_news_rows(aggregate_date=_today(0)))

    # (a) in-band: the model is TOLD the leg is dated, in the prompt.
    block = _keeper(store, bundle)._macro_anchor_block()
    assert "STALE-INPUT" in block
    assert f"{REGIME_SUBSTRATE_PREFIX}/" in block
    assert "DATED reference" in block
    assert "QUANTITATIVE REGIME SUBSTRATE" in block  # still passed, not dropped

    # (b) out-of-band: every theme written carries the verdict.
    records = bundle.stale_input_records()
    assert [r["artifact"] for r in records] == [f"{REGIME_SUBSTRATE_PREFIX}/"]
    assert records[0]["status"] == "stale"


@mock_aws
def test_a_missing_substrate_reads_as_unobserved_not_as_neutral():
    """The renderer must never fabricate a regime. 'Not available' is the
    honest rendering; 'neutral' would be a conclusion nobody measured."""
    store = _store()
    bundle = _load(store, news=_news_rows(aggregate_date=_today(0)))

    block = _keeper(store, bundle)._macro_anchor_block()
    assert "not available this run" in block
    assert "UNOBSERVED" in block


@mock_aws
def test_a_total_macro_feature_outage_renders_as_na_not_as_zero():
    """config-I7272: ``intensity_z`` of None means every candidate feature was
    missing or degenerate — undefined, not a measured calm reading."""
    from thinktank.themes import _render_regime_substrate

    payload = _substrate(calendar_date=_today(1))
    payload["composite"]["intensity_z"] = None
    rendered = _render_regime_substrate(payload)
    assert "composite intensity_z: n/a" in rendered
    assert "composite intensity_z: 0.00" not in rendered


# ── the cross-repo prompt-placeholder cutover ────────────────────────────────


@pytest.mark.parametrize("placeholder", ["macro_anchor", "macro_report"])
def test_both_prompt_placeholder_names_are_supplied(placeholder):
    """The Think Tank spot box clones crucible-research AND alpha-engine-config
    ``main`` fresh on every nightly run, and ``str.format`` raises KeyError on a
    placeholder the caller did not pass. Renaming ``{macro_report}`` to
    ``{macro_anchor}`` in one repo alone is therefore broken in whichever
    direction lands first — there is no safe merge order. Supplying both names
    makes the two merges independent.

    RETIRE the legacy name once the config-side prompt has cut over
    (alpha-engine-config-I7962); this test is what makes that a deliberate
    two-repo step rather than a silent breakage.
    """
    from thinktank.themes import _anchor_kwargs

    kwargs = _anchor_kwargs("ANCHOR-TEXT")
    assert kwargs[placeholder] == "ANCHOR-TEXT"
