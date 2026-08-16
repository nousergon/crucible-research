"""Repo-root pytest fixtures and env defaults.

Sets ``AWS_DEFAULT_REGION`` for the test process so any lazily-built
``boto3.client(...)`` (e.g. ``evals/orchestrator.py``'s CloudWatch
client when ``cloudwatch_client`` isn't injected) succeeds without
``NoRegionError``. Production Lambdas inherit ``AWS_REGION`` from the
runtime; tests without this default fail in CI where no region is
configured. moto's mocked services also require region to be set.

Also pins ``ALPHA_ENGINE_SECRETS_SOURCE=env`` for the test process so
``alpha_engine_lib.secrets.get_secret()`` (post 2026-05-12 .env→SSM
migration) reads from monkeypatched env vars only — never the real
SSM Parameter Store. Set at module import time (not just inside a
fixture body) because ``config.py`` reads secrets at module load,
which happens during test collection before per-test fixtures fire.
"""

from __future__ import annotations

import os
import sys
import types
from unittest.mock import MagicMock

import pytest

# Apply at import time so it's set before any test fixture builds a
# boto3 client. ``setdefault`` means a developer with their own
# AWS_DEFAULT_REGION exported keeps it.
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
# Same rationale for the secrets-source toggle — must be set before
# config.py imports so its module-level get_secret() reads from env.
os.environ.setdefault("ALPHA_ENGINE_SECRETS_SOURCE", "env")


@pytest.fixture(autouse=True)
def _isolate_secrets_from_ssm(monkeypatch):
    """Re-pin ``ALPHA_ENGINE_SECRETS_SOURCE=env`` per test + clear the
    per-process secret cache. Belt-and-suspenders against tests that
    monkeypatch the toggle themselves and forget to restore it.

    See ``alpha-engine-docs/private/env-to-ssm-260512.md`` § Risks.
    """
    monkeypatch.setenv("ALPHA_ENGINE_SECRETS_SOURCE", "env")
    try:
        from nousergon_lib.secrets import clear_cache
    except ImportError:
        yield
        return
    clear_cache()
    yield
    clear_cache()


@pytest.fixture(autouse=True)
def _clear_arctic_union_cache():
    """Empty the reader's in-run union memo around every test.

    ``feature_store_reader._ARCTIC_UNION_CACHE`` is process state keyed on
    ``(tickers, ref_date)`` (alpha-engine-config-I6855). Tests share both —
    ``["AAPL"]`` on one pinned date is the house fixture — so without this a
    test asserting ``read_batch`` was called would silently be served the
    PREVIOUS test's rows and pass while exercising nothing. Cleared on both
    sides so neither an inherited entry nor a leaked one can do it.
    """
    try:
        from data.fetchers.feature_store_reader import _ARCTIC_UNION_CACHE
    except ImportError:
        yield
        return
    _ARCTIC_UNION_CACHE.clear()
    yield
    _ARCTIC_UNION_CACHE.clear()


@pytest.fixture(autouse=True)
def _stage_coverage_absent_unless_stubbed(monkeypatch):
    """Default-deny the config-I7214 stage-coverage primitive in tests.

    Every weekly-SF-backed handler in ``lambda/`` calls
    ``krepis.stage_coverage.assert_stage_coverage`` immediately before it
    returns. That function builds its OWN boto3 S3 and CloudWatch clients
    and reads a registry object out of S3 — it does not inherit a test's
    ``patch("boto3.client")``. So the moment krepis 0.59.4 actually
    published the module, EVERY handler test that runs a handler end to
    end began issuing live AWS calls; on the CI runner (no credentials,
    no route) those retried until the job wedged and the runner was killed
    (run 31817900321 — cancelled after 2m30s with no test output past
    collection, which reads as infrastructure flakiness rather than as the
    test defect it is).

    ``None`` in ``sys.modules`` forces the handler's lazy import to raise
    ``ImportError`` deterministically. That keeps the suite hermetic AND
    makes "the lib is missing" a SIMULATED precondition rather than an
    ambient property of whatever krepis is installed — the assumption that
    silently expired under this suite.

    ``stub_stage_coverage`` overrides this for tests that need to see a
    verdict land (monkeypatch reverts LIFO at teardown).
    """
    monkeypatch.setitem(sys.modules, "krepis.stage_coverage", None)
    yield


@pytest.fixture
def stub_stage_coverage(monkeypatch, _stage_coverage_absent_unless_stubbed):
    """Simulate ``krepis.stage_coverage`` being present.

    The primitive lives in krepis (relocated from an initial
    ``nousergon_lib.stage_coverage`` landing — nousergon-lib-PR314 merged,
    PR315 removes the duplicate — because half its callers are bash
    launchers invoking ``python -m krepis.stage_coverage`` and krepis is
    published rather than git-pinned). It ships from krepis 0.59.4, which
    this repo's ``requirements.txt`` floor now requires.

    Tests that need to see a verdict land in a handler's payload use this
    fixture to inject a fake submodule into ``sys.modules`` ahead of the
    handler's lazy import, so the import succeeds and returns a
    controllable mock — a fake rather than the real module because the
    real one talks to S3 and CloudWatch. Returns the mock so a test can
    assert on call args (stage name, run_date, window_start) or set
    ``side_effect``/``return_value``.
    """
    mock_assert = MagicMock(return_value={"status": "COVERED", "stage": "stub"})
    fake_module = types.ModuleType("krepis.stage_coverage")
    fake_module.assert_stage_coverage = mock_assert
    monkeypatch.setitem(sys.modules, "krepis.stage_coverage", fake_module)
    yield mock_assert


@pytest.fixture
def live_router_resolution():
    """Opt OUT of the autouse resolver stub below.

    Requested by tests whose subject IS ``resolve_group_spec`` — they fake
    krepis' registry read (``resolve_group_structured``) and assert on the spec
    the REAL resolver builds from it, which the stub would replace wholesale.
    Still hermetic: no network, no registry file, no credential.
    """
    return True


@pytest.fixture(autouse=True)
def _router_resolution_is_stubbed_unless_a_test_says_otherwise(monkeypatch, request):
    """No unit test performs a LIVE router group resolution.

    ``agents.langchain_utils.make_agent_llm`` resolves every capability class
    through ``krepis.router.resolve_group_spec`` (alpha-engine-config-I7005).
    That resolver health-probes the local egress proxy and the router edge, and
    on a CI runner neither exists and no consumer credential is resolvable — so
    it correctly FAILS CLOSED with ``ValueError: No model in group 'low' is
    reachable ...`` (model-router-policy R20). Nine tests that only ever wanted
    a stand-in chat model went red on that, and every one of them would have
    been asserting the runner's network rather than the code under test.

    So the resolution is stubbed for the whole suite, at the seam the factory
    actually imports. The fake returns a router-edge route, which is also what
    keeps ``_assert_routed_through_the_proxy`` satisfied from any declared
    context.

    This does NOT weaken the guards on the factory itself: every property of
    the real resolution — that the client is bound to the returned spec, that
    the context is declared and passed verbatim, that a direct-provider route
    raises, that a resolver failure propagates — is asserted in
    ``tests/test_agent_llm_factory.py``, which installs its OWN resolver per
    test. A later ``monkeypatch``/``patch`` of the same attribute wins over
    this one and is restored to it at teardown, so opting out is just patching.
    """
    if "live_router_resolution" in request.fixturenames:
        yield
        return

    try:
        import krepis.router as krepis_router
    except ImportError:
        yield
        return

    from krepis.llm_config import ModelSpec

    def _stub_spec(max_tokens):
        """A REAL ``ModelSpec``, not a duck type.

        ``agents.langchain_utils`` reads four attributes off it, but the same
        resolver serves ``producers/single_agent.py``, which hands the spec
        straight to ``krepis.llm.LLMClient`` — that reads
        ``supports_automatic_prefix_caching`` and would ``AttributeError`` on a
        hand-rolled stand-in. Constructing the real dataclass keeps the stub
        honest about the contract every consumer of the resolver actually has.
        """
        return ModelSpec(
            provider="litellm_proxy",
            model="stub-model",
            max_tokens=max_tokens if max_tokens is not None else 4096,
            base_url="https://router.invalid/v1",
            api_key_env=None,  # placeholder auth: no credential to resolve
            registry_id="litellm:group:stub",
        )

    def _stub_resolve(group, *, exec_context=None, wire=None, max_tokens=None, **kw):
        route = {
            "route": "litellm_proxy",
            "registry_id": f"litellm:group:{group}",
            "primary_registry_id": f"litellm:group:{group}",
            "deployment_id": f"{group}-stub",
            "provider": "litellm",
            "api_base_url": "https://router.invalid/v1",
            # Echo the DECLARED context (R29) rather than inventing one, so a
            # test that declares `lambda` still exercises the guard's
            # non-laptop branch.
            "exec_context": exec_context or "laptop",
            "skipped_entries": [],
        }
        return _stub_spec(max_tokens), route

    monkeypatch.setattr(krepis_router, "resolve_group_spec", _stub_resolve, raising=True)
    yield
