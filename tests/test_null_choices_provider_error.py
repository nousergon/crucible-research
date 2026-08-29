"""OpenRouter reports upstream provider failures in the BODY of a 200 response:
``choices`` comes back null (or empty) with an ``error`` object beside it.

The SDK builds that object without complaint, so it is NOT an exception and
slips past every ``except`` guarding the call. The subsequent ``choices[0]``
then raises::

    TypeError: 'NoneType' object is not subscriptable

which escapes the bounded-retry loop entirely, kills the whole batch on one
transient upstream hiccup, and discards the provider's own error message
unread. This is what had been reddening the judge perturbation smoke on every
PR since 2026-07-25 (green 12:51, red 12:57 the same day).

Two call sites had the identical unguarded subscript — ``evals/judge.py`` and
``thinktank/client.py`` — so this is a bug CLASS, not one defect. Both now
classify a null-choices body as a retryable provider error.

These tests simulate the response shape rather than waiting for a live
provider to misbehave: the failure is transient by nature, so the only way to
prove the guard works is to construct it.
"""

from __future__ import annotations

import types

import pytest
from pydantic import BaseModel

# Imported at module (collection) level, not lazily inside a fixture:
# ``thinktank.client``'s own ``from krepis.router import resolve_group_spec``
# binding must be captured BEFORE any per-test fixture (including the root
# conftest's autouse router stub) monkeypatches ``krepis.router`` — a lazy
# first import inside a fixture body would instead bind to whatever
# ``krepis.router.resolve_group_spec`` already points to at that moment
# (alpha-engine-config-I9302 fallout: this file used to import it lazily and
# a bare-alias `stub-model` spec with no `api_key_env` leaked in, raising
# `LLMConfigError` before ever reaching the null-choices guard under test).
from thinktank.client import ThinktankClient  # noqa: E402


class _NullChoicesResponse:
    """A 200 body carrying a provider error instead of choices."""

    def __init__(self, choices=None):
        self.choices = choices
        self.error = {"message": "upstream provider error", "code": 502}
        self.id = "gen-null-choices"
        self.model = "deepseek/deepseek-v4-flash"
        self.usage = None


class _OkResponse:
    def __init__(self, content='{"ok": true}'):
        msg = types.SimpleNamespace(content=content, tool_calls=[])
        self.choices = [types.SimpleNamespace(message=msg, finish_reason="stop")]
        self.error = None
        self.id = "gen-ok"
        self.model = "deepseek/deepseek-v4-flash"
        self.usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=5)


# ── thinktank/client.py — the provider-call chokepoint ────────────────────


def _make_client(responses):
    """A stub whose .chat.completions.create returns each item in turn."""
    seq = iter(responses)

    def create(**_kwargs):
        item = next(seq)
        if isinstance(item, Exception):
            raise item
        return item

    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))


def _tt_settings():
    from thinktank.settings import ThinktankSettings, TierSpec

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


class _Out(BaseModel):
    ok: bool


@pytest.fixture()
def _tt(monkeypatch):
    """A ThinktankClient over a scripted transport, exercised through its
    PUBLIC surface.

    Post-migration (alpha-engine-config#5223) the retry loop lives in
    ``krepis.llm``, not in a private ``_create_completion`` on this class — so
    the guard is asserted where a caller actually meets it. krepis >= 0.25.0
    sleeps between body-level retries (krepis#93); tests assert the retry
    happened, not how long it waited.
    """
    monkeypatch.setattr("krepis.llm._retry_backoff_sleep", lambda _attempt: None)

    from datetime import date as _date

    import krepis.router as _kr
    from krepis.cost import PriceCard, PriceTable

    monkeypatch.setenv("LITELLM_MASTER_KEY", "consumer-test")
    monkeypatch.setattr(
        _kr, "resolve_group_structured",
        lambda *a, **k: {
            "schema_version": 2, "group": "med", "route": "litellm_proxy",
            "provider": "litellm", "deployment_id": "med",
            "api_base_url": "https://router.example:8443",
            "auth_token_type": "litellm_master_key",
            "registry_id": "litellm:group:med",
            "primary_registry_id": "deepseek-v4-flash-max", "params": {},
        },
    )
    _price_table = PriceTable(cards=[PriceCard(
        model_name="deepseek/deepseek-v4-flash", effective_from=_date(2026, 1, 1),
        input_per_1m=1.0, output_per_1m=2.0,
        cache_read_per_1m=0.0, cache_create_per_1m=0.0,
    )])
    monkeypatch.setattr("krepis.cost.load_default_pricing", lambda: _price_table)

    def build(responses):
        stub = _make_client(responses)
        return ThinktankClient(
            settings=_tt_settings(),
            run_id="testrun",
            client_factory=lambda _provider, _key: stub,
        )

    return build


@pytest.mark.parametrize("empty", [None, []])
def test_thinktank_retries_a_null_choices_body_instead_of_crashing(_tt, empty):
    """The regression: null choices must be retried, not raised as TypeError.

    Re-confirmed 2026-07-30 while migrating onto ``krepis.llm``: on krepis
    0.24.x this raised the original ``TypeError`` again, because the library
    had no guard at any of its five ``choices[0]`` reads. The fork's protection
    was being dropped by adopting the shared code. krepis#93 fixed it at the
    chokepoint; requirements.txt floors at >= 0.25.0 for exactly this.
    """
    client = _tt([_NullChoicesResponse(empty), _OkResponse()])
    result = client.complete("thesis", agent_id="a", system="s", user="u", response_model=_Out)
    assert result.parsed.ok is True, "a null-choices body must be retried, not fatal"


def test_thinktank_gives_up_with_a_diagnosable_error_not_a_typeerror(_tt):
    """After exhausting retries the caller must learn WHAT the provider said.

    A bare TypeError names none of it — the original failure discarded the
    provider payload entirely.
    """
    from thinktank.client import _STRUCTURED_ATTEMPTS, ThinktankLLMError

    client = _tt([_NullChoicesResponse() for _ in range(_STRUCTURED_ATTEMPTS)])
    with pytest.raises(Exception) as exc_info:
        client.complete("thesis", agent_id="a", system="s", user="u", response_model=_Out)

    assert not isinstance(exc_info.value, TypeError), (
        "a null-choices body must not surface as TypeError: 'NoneType' object "
        "is not subscriptable — that was the undiagnosable original failure"
    )
    assert isinstance(exc_info.value, ThinktankLLMError)
    text = str(exc_info.value)
    assert "no choices" in text
    assert "upstream provider error" in text, "provider payload must be surfaced"


def test_thinktank_still_returns_a_healthy_response_unchanged(_tt):
    """The guard must not intercept the happy path."""
    client = _tt([_OkResponse()])
    result = client.complete("thesis", agent_id="a", system="s", user="u", response_model=_Out)
    assert result.parsed.ok is True


# ── evals/judge.py — the same shape, guarded inline ───────────────────────


def test_judge_source_guards_choices_before_subscripting():
    """``evals/judge.py`` must not subscript ``resp.choices`` unguarded.

    Asserted against the source because the surrounding function needs a live
    client, rubric fixtures and an artifact to invoke; the invariant worth
    pinning is simply that the raw subscript never returns.
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "evals" / "judge.py").read_text()

    offenders = [
        line.strip() for line in src.splitlines() if "resp.choices[0]" in line and not line.strip().startswith("#")
    ]
    assert not offenders, (
        "resp.choices[0] must be preceded by a null/empty check — OpenRouter "
        f"returns choices=null on upstream failure: {offenders}"
    )
    assert "provider returned no choices" in src, "the null-choices branch must surface the provider payload"
