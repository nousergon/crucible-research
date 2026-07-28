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


@pytest.mark.parametrize("empty", [None, []])
def test_thinktank_retries_a_null_choices_body_instead_of_crashing(monkeypatch, empty):
    """The regression: null choices must be retried, not raised as TypeError."""
    from thinktank import client as tt

    monkeypatch.setattr(tt, "_backoff_sleep", lambda *a, **k: None)

    ok = _OkResponse()
    stub = _make_client([_NullChoicesResponse(empty), ok])

    inst = tt.ThinktankClient.__new__(tt.ThinktankClient)
    resp = tt.ThinktankClient._create_completion(
        inst,
        stub,
        [{"role": "user", "content": "hi"}],
        {},
        tier_name="t",
        agent_id="a",
    )

    assert resp is ok, "a null-choices body must be retried, not returned"
    # The whole point: the caller can subscript the result.
    assert resp.choices[0].message.content


def test_thinktank_gives_up_with_a_diagnosable_error_not_a_typeerror(monkeypatch):
    """After exhausting retries the caller must learn WHAT the provider said.

    A bare TypeError names none of it — the original failure discarded the
    provider payload entirely.
    """
    from thinktank import client as tt

    monkeypatch.setattr(tt, "_backoff_sleep", lambda *a, **k: None)
    stub = _make_client([_NullChoicesResponse() for _ in range(tt._HTTP_RETRY_ATTEMPTS)])
    inst = tt.ThinktankClient.__new__(tt.ThinktankClient)

    with pytest.raises(Exception) as exc_info:
        tt.ThinktankClient._create_completion(
            inst,
            stub,
            [{"role": "user", "content": "hi"}],
            {},
            tier_name="t",
            agent_id="a",
        )

    assert not isinstance(exc_info.value, TypeError), (
        "a null-choices body must not surface as TypeError: 'NoneType' object "
        "is not subscriptable — that was the undiagnosable original failure"
    )
    text = str(exc_info.value)
    assert "no choices" in text
    assert "upstream provider error" in text, "provider payload must be surfaced"


def test_thinktank_still_returns_a_healthy_response_unchanged(monkeypatch):
    """The guard must not intercept the happy path."""
    from thinktank import client as tt

    monkeypatch.setattr(tt, "_backoff_sleep", lambda *a, **k: None)
    ok = _OkResponse()
    inst = tt.ThinktankClient.__new__(tt.ThinktankClient)

    resp = tt.ThinktankClient._create_completion(
        inst,
        _make_client([ok]),
        [{"role": "user", "content": "hi"}],
        {},
        tier_name="t",
        agent_id="a",
    )
    assert resp is ok


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
