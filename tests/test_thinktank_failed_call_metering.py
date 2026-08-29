"""An error handler must not be able to destroy the error it is handling.

Live failure, run ``b4a21acfbbfc`` (2026-08-11). The Think Tank's ``pillar``
tier failed on the box; ``complete()`` caught the ``LLMError`` and — before
re-raising it — recorded the failed call's spend. A failed call has no served
model, and ``_cost_for`` refuses to price a group-addressed call it cannot
identify, so the accounting raised FIRST and the run aborted reporting:

    tier=pillar group=med served_model='': ... Add a price card for '' in
    krepis model_pricing.yaml.

That message is about the accounting, not about why the call failed. The real
cause never reached the manifest, the alert, or the log — it is unrecoverable
from that run. Several sessions of Think Tank work chased the mask.

Two properties, both regression-guarded here:

1. Whatever happens to the accounting, the transport error is what propagates.
2. A failed call is still metered where it honestly can be — against the
   deployment this client ADDRESSED, which is known before the call is made.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
from krepis.llm import LLMError, LLMUsage  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import thinktank.client as client_mod  # noqa: E402
from thinktank.client import ThinktankClient, ThinktankLLMError  # noqa: E402
from thinktank.settings import ThinktankSettings, TierSpec  # noqa: E402


class _Answer(BaseModel):
    value: str = ""


def _settings(tier: TierSpec) -> ThinktankSettings:
    return ThinktankSettings(
        bucket="b", daily_new_names=1, rank_ceiling=1, sweep_chunk_size=1,
        stale_after_days=1, monthly_budget_usd_default=1.0,
        budget_ssm_param="/p", tiers={tier.name: tier},
    )


class _Spec:
    def __init__(self, model: str) -> None:
        self.model = model


class _FailingLLM:
    """Stands in for ``krepis.llm.LLMClient``: exhausts, then raises."""

    def __init__(self, model: str = "med-deepseek-v4-flash-max") -> None:
        self.spec = _Spec(model)

    def structured(self, **_kwargs):
        raise LLMError(
            "upstream returned 503 on every attempt",
            usage=LLMUsage(input_tokens=1_000_000, output_tokens=0),
        )


def _client(tier: TierSpec, monkeypatch, *, llm=None) -> ThinktankClient:
    c = ThinktankClient(settings=_settings(tier), run_id="r")
    monkeypatch.setattr(
        c, "_llm_client_for", lambda *a, **k: llm or _FailingLLM()
    )
    return c


def _complete(c: ThinktankClient, tier: TierSpec):
    return c.complete(
        tier.name,
        agent_id="analyst_pillar",
        system="s",
        user="u",
        response_model=_Answer,
    )


PILLAR = TierSpec(name="pillar", group="med", max_tokens=4000)


class TestTheTransportErrorSurvivesAccounting:
    def test_unpriceable_failed_call_still_reports_the_real_cause(
        self, monkeypatch, caplog
    ):
        """The metering path cannot resolve anything — the deployment maps to
        no upstream model. The run must still abort naming the 503."""
        monkeypatch.setattr(
            client_mod, "served_model_for_deployment", lambda _n: None
        )
        c = _client(PILLAR, monkeypatch)

        with pytest.raises(ThinktankLLMError) as excinfo:
            _complete(c, PILLAR)

        message = str(excinfo.value)
        assert "response failed after bounded retries" in message
        assert "upstream returned 503" in message
        # The mask, precisely: an accounting complaint standing where the
        # cause belongs.
        assert "price card" not in message
        assert "fail open" not in message

    def test_the_unmetered_spend_is_reported_rather_than_dropped_silently(
        self, monkeypatch, caplog
    ):
        """The swallow is narrow and loud: flow-doctor is attached to the root
        logger at ERROR, so this line alerts."""
        monkeypatch.setattr(
            client_mod, "served_model_for_deployment", lambda _n: None
        )
        c = _client(PILLAR, monkeypatch)

        with caplog.at_level("ERROR"), pytest.raises(ThinktankLLMError):
            _complete(c, PILLAR)

        assert any(
            "could not be metered" in r.getMessage() for r in caplog.records
        )

    def test_original_exception_is_the_cause_chain(self, monkeypatch):
        """``raise ... from exc`` must chain the LLMError, not the pricing
        error — the chain is what a traceback prints."""
        monkeypatch.setattr(
            client_mod, "served_model_for_deployment", lambda _n: None
        )
        c = _client(PILLAR, monkeypatch)

        with pytest.raises(ThinktankLLMError) as excinfo:
            _complete(c, PILLAR)

        assert isinstance(excinfo.value.__cause__, LLMError)


class TestFailedCallIsStillMetered:
    def test_priced_against_the_addressed_deployment(self, monkeypatch):
        """The served model is unknown; the ADDRESSED one is not. A failed
        call contributing zero would understate the run against the monthly
        budget guard."""
        monkeypatch.setattr(
            client_mod,
            "served_model_for_deployment",
            lambda _n: "deepseek-v4-flash",
        )
        c = _client(PILLAR, monkeypatch)

        with pytest.raises(ThinktankLLMError):
            _complete(c, PILLAR)

        usage = c._usage["pillar"]
        assert usage.calls == 1
        assert usage.input_tokens == 1_000_000
        assert usage.cost_usd == pytest.approx(0.14, rel=1e-6)


class TestPriceableModelHelper:
    def test_group_tier_maps_deployment_to_upstream_model(self, monkeypatch):
        monkeypatch.setattr(
            client_mod,
            "served_model_for_deployment",
            lambda n: "deepseek-v4-flash" if n == "med-deepseek-v4-flash-max" else None,
        )
        assert client_mod._priceable_model_for_failed_call(
            _FailingLLM(), PILLAR
        ) == "deepseek-v4-flash"

    def test_a_missing_registry_does_not_propagate(self, monkeypatch):
        """``served_model_for_deployment`` raises when no registry is found.
        Here that must degrade to the deployment name, never to an exception:
        this helper runs on the failure path and may not add a second one."""

        def _boom(_n):
            raise FileNotFoundError("no LLM_MODEL_REGISTRY.yaml")

        monkeypatch.setattr(client_mod, "served_model_for_deployment", _boom)
        assert client_mod._priceable_model_for_failed_call(
            _FailingLLM(), PILLAR
        ) == "med-deepseek-v4-flash-max"
