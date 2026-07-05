"""Strict-mode validation env-var helper.

Lives at repo root (alongside ``preflight.py``, ``retry.py``,
``data_manifest.py``) so both ``graph/research_graph.py`` (state-shape
validators) and the ``agents/`` LLM-extraction sites can import it
without a circular import. ``graph`` depends on ``agents``, so a shared
helper at either layer would create a one-way violation; root level is
neutral.

PR 2.5 Step F (2026-04-30 evening) flipped the default to **True** —
strict-by-default. Operators set ``STRICT_VALIDATION=false`` in the
Lambda env for the emergency override path; the 30-second console flip
takes effect on warm containers without redeploy because the helper
reads ``os.environ`` fresh on each call.

Pre-flip default was ``False`` during the PR 2.1–2.4 rollout so each
agent's ``with_structured_output`` migration could ship without
flipping behavior; STRICT_VALIDATION=true was set explicitly in prod
during the rollout window. After Step F, the default matches prod.
"""

from __future__ import annotations

import os


def is_strict_validation_enabled() -> bool:
    """Return ``True`` when typed-state validation should hard-fail
    on schema violations.

    Reads the ``STRICT_VALIDATION`` env var fresh on each call so a
    Lambda console flip takes effect on warm containers without
    redeploy. Truthy values: ``true``, ``1``, ``yes`` (case-insensitive).
    Default is ``true`` (strict-by-default); set ``STRICT_VALIDATION=false``
    to opt out.
    """
    return os.environ.get("STRICT_VALIDATION", "true").lower() in (
        "true", "1", "yes"
    )
