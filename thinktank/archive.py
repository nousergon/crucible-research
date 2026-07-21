"""Moat-assessment archive — per-ticker time series inside ``thinktank/``.

Port of ``archive/manager.py::save_moat_profile`` (ROADMAP L1650) onto
``ThinktankStore``, so the artifact keeps getting a live producer post
config#1580 (the qual/CIO graph that used to write
``archive/universe/{ticker}/moat_profile.json`` was removed from the
weekly SF). Writes to ``thinktank/moat_profile/{ticker}.json`` instead —
inside Think Tank's own namespace, per the writes-only-to-``thinktank/``
boundary in ``thinktank/__init__.py``'s module docstring.

Moats decay slowly; the time derivative is the real signal, so this is an
append-only series rather than a snapshot replaced each run.
"""

from __future__ import annotations

import logging

from thinktank import MOAT_PROFILE_KEY_TMPL
from thinktank.storage import ThinktankStore

logger = logging.getLogger(__name__)


def save_moat_profile(
    store: ThinktankStore,
    ticker: str,
    run_date: str,
    moat_assessment: dict,
) -> None:
    """Append a moat-assessment snapshot to the per-ticker time series.

    ``moat_assessment`` shape: dict-dump of
    ``nousergon_lib.pillars.MoatAssessment``. Append semantics: read the
    existing JSON list (or ``[]`` on miss), push ``{run_date,
    **moat_assessment}``, write back. Idempotent on ``(ticker, run_date)``
    — a second call for the same key replaces the prior row in place
    rather than duplicating.
    """
    if not moat_assessment or not isinstance(moat_assessment, dict):
        return
    key = MOAT_PROFILE_KEY_TMPL.format(ticker=ticker)
    existing = store.get_json(key)
    history: list[dict] = existing if isinstance(existing, list) else []

    new_entry = {"run_date": run_date, **moat_assessment}
    # Idempotency: replace any prior entry with the same run_date.
    history = [e for e in history if e.get("run_date") != run_date]
    history.append(new_entry)
    # Keep chronological order. run_date is ISO so lex sort works.
    history.sort(key=lambda e: str(e.get("run_date") or ""))

    store.put_json(key, history)
