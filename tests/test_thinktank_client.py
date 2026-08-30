"""ThinktankClient: validation loop, cost meter, SFT staging/flush."""

from __future__ import annotations

import json
from types import SimpleNamespace

import boto3
import pytest
from moto import mock_aws
from pydantic import BaseModel

from thinktank.client import (
    _STRUCTURED_ATTEMPTS,
    SftCaptureWriteError,
    ThinktankClient,
    ThinktankLLMError,
)
from thinktank.settings import ThinktankSettings, TierSpec


class _Out(BaseModel):
    model_config = {"extra": "forbid"}
    answer: str
    score: int


def _settings() -> ThinktankSettings:
    return ThinktankSettings(
        bucket="alpha-engine-research",
        daily_new_names=5,
        rank_ceiling=150,
        sweep_chunk_size=25,
        stale_after_days=30,
        monthly_budget_usd_default=25.0,
        budget_ssm_param="/thinktank/monthly_budget_usd",
        tiers={
            "thesis": TierSpec(
                name="thesis",
                group="med",
                max_tokens=100,
                structured_outputs=True,
            )
        },
    )


# Every tier is group-addressed (alpha-engine-config-I9302): resolving a
# client means calling krepis.router.resolve_group_structured. This suite
# tests ThinktankClient's OWN validation/retry/cost/SFT logic, not routing
# itself (that is test_thinktank_group_addressing.py's job) — so the router
# call is stubbed to a fixed route and every fixture below asserts on the
# `client_factory` test seam exactly as it did before I9302 removed the
# pinned-provider mode; only how the ModelSpec is obtained has changed.
_FAKE_ROUTE = {
    "schema_version": 2,
    "group": "med",
    "route": "litellm_proxy",
    "provider": "litellm",
    "deployment_id": "med",
    "api_base_url": "https://router.example:8443",
    "auth_token_type": "litellm_master_key",
    "registry_id": "litellm:group:med",
    "primary_registry_id": "deepseek-v4-flash-max",
    "params": {},
}


@pytest.fixture(autouse=True)
def _stub_router(monkeypatch):
    import krepis.router as _kr

    monkeypatch.setenv("LITELLM_MASTER_KEY", "consumer-test")
    monkeypatch.setattr(_kr, "resolve_group_structured", lambda *a, **k: dict(_FAKE_ROUTE))
    # A FAILED call is priced from the addressed deployment
    # (thinktank/client.py::_priceable_model_for_failed_call), mapped back to
    # an upstream model via this function — bound into thinktank.client's own
    # namespace at import time, so patch it there.
    import thinktank.client as _tc

    monkeypatch.setattr(_tc, "served_model_for_deployment", lambda _name: "deepseek-v4-flash-max")


@pytest.fixture(autouse=True)
def _stub_price_card(monkeypatch):
    """Group-addressed cost is priced from the model that actually SERVED
    (``krepis.cost.PriceCard``), not a per-tier literal — there is no
    pinned tier left to carry one (alpha-engine-config-I9302). $1/M in,
    $2/M out reproduces the pre-I9302 fixture's own numbers so the retry/
    validation/SFT assertions below are unaffected by the routing change."""
    from datetime import date as _date

    from krepis.cost import PriceCard, PriceTable

    table = PriceTable(
        cards=[
            PriceCard(
                model_name="deepseek-v4-flash-max",
                effective_from=_date(2026, 1, 1),
                input_per_1m=1.0,
                output_per_1m=2.0,
                cache_read_per_1m=0.0,
                cache_create_per_1m=0.0,
            )
        ]
    )
    monkeypatch.setattr("krepis.cost.load_default_pricing", lambda: table)


class _FakeCompletions:
    def __init__(self, bodies: list[str]):
        self._bodies = list(bodies)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        body = self._bodies.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=body))],
            usage=SimpleNamespace(prompt_tokens=1_000_000, completion_tokens=500_000),
            # A real served model, distinct from the "med" group alias — the
            # krepis I6543 masquerade guard refuses to bill/record a call
            # under the bare alias (krepis.llm.LLMClient._served_model_for).
            model="deepseek-v4-flash-max",
        )


def _client(bodies: list[str], monkeypatch) -> tuple[ThinktankClient, _FakeCompletions]:
    fake = _FakeCompletions(bodies)
    holder = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    client = ThinktankClient(
        settings=_settings(),
        run_id="testrun",
        client_factory=lambda provider, key: holder,
    )
    return client, fake


def test_valid_response_parses_and_costs(monkeypatch):
    client, fake = _client([json.dumps({"answer": "yes", "score": 7})], monkeypatch)
    result = client.complete("thesis", agent_id="a", system="s", user="u", response_model=_Out)
    assert result.parsed.answer == "yes"
    # 1M in @ $1/M + 0.5M out @ $2/M = $2.00
    assert result.cost_usd == pytest.approx(2.0)
    assert client.total_cost_usd() == pytest.approx(2.0)
    assert fake.calls[0]["response_format"]["json_schema"]["name"] == "_Out"


def test_markdown_fenced_json_is_tolerated(monkeypatch):
    body = "```json\n" + json.dumps({"answer": "ok", "score": 1}) + "\n```"
    client, _ = _client([body], monkeypatch)
    result = client.complete("thesis", agent_id="a", system="s", user="u", response_model=_Out)
    assert result.parsed.answer == "ok"


# ── retry behaviour, post-krepis migration (alpha-engine-config#5223) ──────
#
# The transport is now ``krepis.llm.LLMClient.structured()``, so these tests
# assert the CONTRACT thinktank depends on rather than the fork's internals.
# What must survive the migration, per the fork's own design notes:
#   * a body the SDK cannot decode is retried FRESH (no corrective turn);
#   * a null-choices body is retried, not raised as a bare TypeError;
#   * a schema violation gets a corrective turn carrying the error back;
#   * the budget is _STRUCTURED_ATTEMPTS, not the library's smaller default;
#   * exhaustion is LOUD (ThinktankLLMError), never a silent degrade.


@pytest.fixture()
def no_backoff(monkeypatch):
    """krepis >= 0.25.0 sleeps between body-level retries (krepis#93). Tests
    assert the retry happened, not how long it waited."""
    monkeypatch.setattr("krepis.llm._retry_backoff_sleep", lambda _attempt: None)


def _decode_error_of_a_keepalive_only_body() -> json.JSONDecodeError:
    """The exact exception the OpenAI SDK propagates when the gateway answers
    200 with keep-alive padding and no payload.

    Live incident 2026-07-30, flow-doctor report 019fb37b8b792803c2a92b924042:
    ``JSONDecodeError: Expecting value: line 1497 column 1 (char 8228)`` —
    ``json.loads`` skips leading whitespace before its first decode, so the
    reported offset IS the length of the whitespace run.
    """
    body = "    \n" * 1495 + " " * 752 + "\n"
    assert len(body) == 8228 and body.count("\n") == 1496
    with pytest.raises(json.JSONDecodeError) as exc_info:
        json.loads(body)
    assert str(exc_info.value) == "Expecting value: line 1497 column 1 (char 8228)"
    return exc_info.value


class _ScriptedCompletions:
    """Returns each item in turn; an Exception item is raised instead."""

    def __init__(self, items):
        self._items = list(items)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self._items.pop(0)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, str):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=item))],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
                model="deepseek-v4-flash-max",
            )
        return item


def _scripted_client(items) -> tuple[ThinktankClient, _ScriptedCompletions]:
    fake = _ScriptedCompletions(items)
    holder = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    client = ThinktankClient(
        settings=_settings(),
        run_id="testrun",
        client_factory=lambda provider, key: holder,
    )
    return client, fake


def test_bounded_retry_recovers_once(monkeypatch):
    # Valid JSON, wrong shape (missing `score`) — a genuine model mistake, so
    # it takes a corrective retry that feeds the validation error back.
    client, fake = _client(
        [json.dumps({"answer": "bad"}), json.dumps({"answer": "fixed", "score": 2})],
        monkeypatch,
    )
    result = client.complete("thesis", agent_id="a", system="s", user="u", response_model=_Out)
    assert result.parsed.answer == "fixed"
    assert len(fake.calls) == 2
    assert "failed validation with:" in fake.calls[1]["messages"][-1]["content"]


def test_fails_loud_after_schema_retries_exhausted(monkeypatch):
    client, fake = _client([json.dumps({"answer": "bad"})] * _STRUCTURED_ATTEMPTS, monkeypatch)
    with pytest.raises(ThinktankLLMError):
        client.complete("thesis", agent_id="a", system="s", user="u", response_model=_Out)
    assert len(fake.calls) == _STRUCTURED_ATTEMPTS
    # spend from failed attempts is still metered
    assert client.total_cost_usd() > 0


def test_non_json_body_retries_fresh_then_succeeds(monkeypatch, no_backoff):
    # A non-JSON provider body (rate-limit / error page) is provider flakiness,
    # not a model mistake — retried FRESH, with no corrective turn appended.
    client, fake = _client(
        ["not json at all", "<html>rate limited</html>", json.dumps({"answer": "ok", "score": 3})],
        monkeypatch,
    )
    result = client.complete("thesis", agent_id="a", system="s", user="u", response_model=_Out)
    assert result.parsed.answer == "ok"
    assert len(fake.calls) == 3


def test_fails_loud_after_non_json_retries_exhausted(monkeypatch, no_backoff):
    client, fake = _client(["garbage one", "garbage two", "garbage three"], monkeypatch)
    with pytest.raises(ThinktankLLMError):
        client.complete("thesis", agent_id="a", system="s", user="u", response_model=_Out)
    assert len(fake.calls) == _STRUCTURED_ATTEMPTS


def test_transport_decode_error_is_transient_and_retried(monkeypatch, no_backoff):
    """A body the SDK cannot decode is provider flakiness, not a model mistake.

    It is invisible to the SDK's own ``max_retries`` (parsing happens after the
    transaction is already considered final), so unless the transport
    classifies it, it escapes the retry loop entirely — which is how one
    gateway hiccup killed the 2026-07-30 daily Think Tank run.
    """
    client, fake = _scripted_client(
        [
            _decode_error_of_a_keepalive_only_body(),
            _decode_error_of_a_keepalive_only_body(),
            json.dumps({"answer": "ok", "score": 4}),
        ]
    )
    result = client.complete("thesis", agent_id="a", system="s", user="u", response_model=_Out)
    assert result.parsed.answer == "ok"
    assert len(fake.calls) == 3


def test_transport_decode_error_fails_loud_after_retries_exhausted(monkeypatch, no_backoff):
    """Bounded, not swallowed: a PERSISTENT decode failure still raises."""
    client, fake = _scripted_client([_decode_error_of_a_keepalive_only_body() for _ in range(_STRUCTURED_ATTEMPTS)])
    with pytest.raises(ThinktankLLMError):
        client.complete("thesis", agent_id="a", system="s", user="u", response_model=_Out)
    assert len(fake.calls) == _STRUCTURED_ATTEMPTS


# Null-choices bodies (OpenRouter's 200-with-an-error shape) are covered as a
# bug CLASS across both call sites in tests/test_null_choices_provider_error.py,
# not duplicated here.


def test_transient_provider_error_propagates_for_the_sdk_to_retry(monkeypatch):
    """A connection blip is the SDK's own ``max_retries`` job (it backs off on
    status/connection failures). It is not swallowed here."""
    from openai import APIConnectionError

    client, fake = _scripted_client([APIConnectionError(request=SimpleNamespace())])
    with pytest.raises(APIConnectionError):
        client.complete("thesis", agent_id="a", system="s", user="u", response_model=_Out)
    assert len(fake.calls) == 1


def test_retry_budget_is_not_silently_shrunk_by_the_migration():
    """The fork ran 3 transport attempts and 3 non-JSON attempts on separate
    budgets. krepis defaults to 2 on ONE budget — taking that default would
    have shrunk the budget as a side effect of consolidating, on exactly the
    failure that motivated the migration."""
    assert _STRUCTURED_ATTEMPTS >= 3


def test_sft_flush_gated_by_capture_flag(monkeypatch):
    client, _ = _client([json.dumps({"answer": "y", "score": 1})], monkeypatch)
    client.complete("thesis", agent_id="a", system="s", user="u", response_model=_Out)
    monkeypatch.delenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", raising=False)
    assert client.flush_sft(None, "alpha-engine-research", "2026-07-02") == 0


def test_sft_flush_writes_jsonl_and_raises_loud(monkeypatch):
    monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "true")
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="alpha-engine-research")
        client, _ = _client([json.dumps({"answer": "y", "score": 1})], monkeypatch)
        client.complete("thesis", agent_id="a", system="s", user="u", response_model=_Out)
        flushed = client.flush_sft(s3, "alpha-engine-research", "2026-07-02")
        assert flushed == 1
        key = "decision_artifacts/_sft_raw/2026-07-02/testrun/a.jsonl"
        row = json.loads(s3.get_object(Bucket="alpha-engine-research", Key=key)["Body"].read())
        assert row["producer"] == "crucible_thinktank"
        assert row["meta"]["tier"] == "thesis"

        # write failure surfaces loud (no-silent-fails)
        client2, _ = _client([json.dumps({"answer": "y", "score": 1})], monkeypatch)
        client2.complete("thesis", agent_id="a", system="s", user="u", response_model=_Out)
        with pytest.raises(SftCaptureWriteError):
            client2.flush_sft(s3, "no-such-bucket-xyz", "2026-07-02")


def test_sft_meta_rides_into_row_meta(monkeypatch):
    client, _ = _client([json.dumps({"answer": "y", "score": 1})], monkeypatch)
    client.complete(
        "thesis",
        agent_id="a",
        system="s",
        user="u",
        response_model=_Out,
        sft_meta={"ticker": "AAPL", "thesis_version": 3, "capture_run_id": "testrun-AAPL-v3"},
    )
    row = client._sft_rows["a"][0]
    assert row.meta["ticker"] == "AAPL"
    assert row.meta["capture_run_id"] == "testrun-AAPL-v3"
    assert row.meta["run_id"] == "testrun"  # base keys not clobbered
