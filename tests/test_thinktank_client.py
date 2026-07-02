"""ThinktankClient: validation loop, cost meter, SFT staging/flush."""

from __future__ import annotations

import json
from types import SimpleNamespace

import boto3
import pytest
from moto import mock_aws
from pydantic import BaseModel

from thinktank.client import SftCaptureWriteError, ThinktankClient, ThinktankLLMError
from thinktank.settings import ProviderSpec, ThinktankSettings, TierSpec


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
        providers={"fake": ProviderSpec(name="fake", base_url="http://x", key_secret="OPENROUTER_API_KEY")},
        tiers={
            "thesis": TierSpec(
                name="thesis", provider="fake", model="fake/model",
                max_tokens=100, price_in_per_m=1.0, price_out_per_m=2.0,
                structured_outputs=True,
            )
        },
    )


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
        )


def _client(bodies: list[str], monkeypatch) -> tuple[ThinktankClient, _FakeCompletions]:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
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
    result = client.complete(
        "thesis", agent_id="a", system="s", user="u", response_model=_Out
    )
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


def test_bounded_retry_recovers_once(monkeypatch):
    client, fake = _client(
        ["not json at all", json.dumps({"answer": "fixed", "score": 2})], monkeypatch
    )
    result = client.complete("thesis", agent_id="a", system="s", user="u", response_model=_Out)
    assert result.parsed.answer == "fixed"
    assert len(fake.calls) == 2
    # corrective turn fed the validation error back
    assert "failed schema validation" in fake.calls[1]["messages"][-1]["content"]


def test_fails_loud_after_bounded_retry(monkeypatch):
    client, fake = _client(["nope", "still nope"], monkeypatch)
    with pytest.raises(ThinktankLLMError):
        client.complete("thesis", agent_id="a", system="s", user="u", response_model=_Out)
    assert len(fake.calls) == 2
    # spend from failed attempts is still metered
    assert client.total_cost_usd() > 0


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
        row = json.loads(
            s3.get_object(Bucket="alpha-engine-research", Key=key)["Body"].read()
        )
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
        "thesis", agent_id="a", system="s", user="u", response_model=_Out,
        sft_meta={"ticker": "AAPL", "thesis_version": 3, "capture_run_id": "testrun-AAPL-v3"},
    )
    row = client._sft_rows["a"][0]
    assert row.meta["ticker"] == "AAPL"
    assert row.meta["capture_run_id"] == "testrun-AAPL-v3"
    assert row.meta["run_id"] == "testrun"  # base keys not clobbered
