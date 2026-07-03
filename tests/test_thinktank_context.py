"""Unit tests for the rag_filings context probe (config#1606).

The probe must not report ``sources_present.rag_filings=true`` unless
per-ticker ``retrieve()`` can actually succeed. DB reachability alone
(``nousergon_lib.rag.is_available()``) is necessary but not sufficient —
``retrieve()`` also needs a resolvable ``VOYAGE_API_KEY`` (query-embedding
credential). This pins: probe true + secret present -> True; probe true +
secret absent -> False (the bug being fixed); probe false -> False
regardless of the secret (unchanged behavior, no secret lookup needed).
"""

from __future__ import annotations

from unittest.mock import patch

import boto3
from moto import mock_aws

from thinktank.context import load_context
from thinktank.storage import ThinktankStore

BUCKET = "alpha-engine-research"


def _store() -> ThinktankStore:
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)
    return ThinktankStore(bucket=BUCKET, s3_client=s3)


@mock_aws
def test_rag_filings_true_when_db_reachable_and_key_present():
    store = _store()
    with (
        patch("nousergon_lib.rag.is_available", return_value=True),
        patch("nousergon_lib.secrets.get_secret", return_value="voyage-key-xyz"),
    ):
        bundle = load_context(store)
    assert bundle.sources_present["rag_filings"] is True
    assert bundle.rag_available is True


@mock_aws
def test_rag_filings_false_when_db_reachable_but_key_absent(caplog):
    """The bug: probe reported true while VOYAGE_API_KEY was unset, so
    every per-ticker retrieve() failed downstream. The probe must now
    catch this and record rag_filings=false with a WARN."""
    store = _store()
    with (
        patch("nousergon_lib.rag.is_available", return_value=True),
        patch("nousergon_lib.secrets.get_secret", return_value=None),
    ):
        bundle = load_context(store)
    assert bundle.sources_present["rag_filings"] is False
    assert bundle.rag_available is False
    assert any(
        "VOYAGE_API_KEY" in rec.message for rec in caplog.records
    )


@mock_aws
def test_rag_filings_false_when_db_unreachable_key_not_checked():
    store = _store()
    with (
        patch("nousergon_lib.rag.is_available", return_value=False),
        patch("nousergon_lib.secrets.get_secret") as mock_get_secret,
    ):
        bundle = load_context(store)
    assert bundle.sources_present["rag_filings"] is False
    mock_get_secret.assert_not_called()


@mock_aws
def test_rag_filings_false_when_secret_probe_raises():
    store = _store()
    with (
        patch("nousergon_lib.rag.is_available", return_value=True),
        patch(
            "nousergon_lib.secrets.get_secret",
            side_effect=RuntimeError("SSM unavailable"),
        ),
    ):
        bundle = load_context(store)
    assert bundle.sources_present["rag_filings"] is False
