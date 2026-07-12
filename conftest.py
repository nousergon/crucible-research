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
