"""``thinktank/archive.py::save_moat_profile`` — the append-only,
idempotent-on-run_date time series (config#2678, port of
``archive/manager.py::save_moat_profile`` onto ``ThinktankStore``).

Mirrors ``tests/test_archive.py::TestSaveMoatProfile`` (the legacy
module's test class) against moto S3 instead of a mocked boto3 client,
matching this package's existing test convention
(``tests/test_thinktank_run.py``).
"""

from __future__ import annotations

import boto3
import pytest
from moto import mock_aws

from thinktank.archive import save_moat_profile
from thinktank.storage import ThinktankStore

BUCKET = "alpha-engine-research"


@pytest.fixture()
def store():
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        yield ThinktankStore(BUCKET, s3)


def test_empty_assessment_is_noop(store):
    save_moat_profile(store, "AAPL", "2026-07-17", {})
    save_moat_profile(store, "AAPL", "2026-07-17", None)  # type: ignore[arg-type]
    assert store.get_json("thinktank/moat_profile/AAPL.json") is None


def test_first_write_seeds_single_entry_list(store):
    save_moat_profile(store, "AAPL", "2026-07-17", {"primary_type": "wide", "trend": "stable"})
    history = store.get_json("thinktank/moat_profile/AAPL.json")
    assert history == [{"run_date": "2026-07-17", "primary_type": "wide", "trend": "stable"}]


def test_append_preserves_prior_entries(store):
    save_moat_profile(store, "AAPL", "2026-07-03", {"primary_type": "narrow"})
    save_moat_profile(store, "AAPL", "2026-07-10", {"primary_type": "narrow"})
    save_moat_profile(store, "AAPL", "2026-07-17", {"primary_type": "wide"})
    history = store.get_json("thinktank/moat_profile/AAPL.json")
    assert [e["run_date"] for e in history] == ["2026-07-03", "2026-07-10", "2026-07-17"]
    assert history[-1]["primary_type"] == "wide"


def test_idempotent_on_same_run_date(store):
    save_moat_profile(store, "AAPL", "2026-07-17", {"primary_type": "narrow"})
    save_moat_profile(store, "AAPL", "2026-07-17", {"primary_type": "wide"})
    history = store.get_json("thinktank/moat_profile/AAPL.json")
    assert len(history) == 1
    assert history[0]["primary_type"] == "wide"


def test_chronological_sort_normalizes_out_of_order_history(store):
    save_moat_profile(store, "AAPL", "2026-07-17", {"primary_type": "wide"})
    save_moat_profile(store, "AAPL", "2026-07-03", {"primary_type": "narrow"})
    save_moat_profile(store, "AAPL", "2026-07-10", {"primary_type": "narrow"})
    history = store.get_json("thinktank/moat_profile/AAPL.json")
    assert [e["run_date"] for e in history] == ["2026-07-03", "2026-07-10", "2026-07-17"]


def test_per_ticker_namespacing(store):
    save_moat_profile(store, "AAPL", "2026-07-17", {"primary_type": "wide"})
    save_moat_profile(store, "MSFT", "2026-07-17", {"primary_type": "narrow"})
    assert store.get_json("thinktank/moat_profile/AAPL.json")[0]["primary_type"] == "wide"
    assert store.get_json("thinktank/moat_profile/MSFT.json")[0]["primary_type"] == "narrow"
