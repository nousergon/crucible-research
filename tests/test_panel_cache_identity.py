"""Regression guard: the closes-panel cache must not key on an OBJECT ADDRESS.

WHY THIS FILE EXISTS
---------------------
``scoring/leaderboard_producers._PANEL_CACHE`` is a within-invocation dedup so
the three leaderboards built in one Scanner invocation pay for the ~904-symbol
ArcticDB slice once instead of three times (alpha-engine-config-I7584/I7587).
Its key was ``(bucket, entry_dates, horizon_days, symbols, id(loader))``.

``id()`` is unique only among objects alive AT THE SAME MOMENT. CPython re-uses
the address of a freed object, so a loader constructed after an earlier one was
collected routinely receives the same integer — measured here at 4 out of 5
attempts on CPython 3.12. Every other component of the key is identical between
two callers asking for the same bucket, cohort dates, horizon and universe, so
the address was the ONLY thing separating two different panels, and it stopped
separating them whenever the allocator recycled it.

Observed live, 2026-08-18, crucible-research CI: two consecutive tests in
``tests/test_leaderboard_scoring.py`` each build their own ``_Panel`` and pass
``panel.loader()``. The second received the FIRST test's panel — which carries
no ``SPY`` row — so ``topn_alpha_vs_benchmark`` had no benchmark return on any
date, collapsed to ``None``, and the assertion on it raised ``TypeError:
'NoneType' object is not subscriptable``. The give-away in the log is a
"reusing the in-process closes panel" line emitted for a loader that had never
been cached. Nothing about the failing test or the code under it had changed;
an unrelated PR added two module imports, the heap layout moved, and the
collision started landing. That is the defining property of this class: the
wrong answer is served or not served by allocator luck, and it is served
SILENTLY — the caller cannot tell a hit on its own panel from a hit on
somebody else's.

The fix holds the loader OBJECT in the key. A key that holds it keeps it alive,
and a live object's address cannot be handed to anything else, so the collision
is impossible by construction rather than merely unlikely. These tests assert
that property directly, so removing it cannot pass review by accident.
"""

from __future__ import annotations

import gc

import pytest

from scoring import leaderboard_producers as lp

_BUCKET = "alpha-engine-research"
_DATES = ["2026-06-01"]
_HORIZON = 21


def _loader(panel: dict):
    """A distinct loader object over a fixed panel — the shape every test and
    backfill caller uses (``closes_panel_loader=...``)."""
    return lambda bucket, entry_dates, horizon_days, symbols=None: panel


def _panel(**closes: float) -> dict:
    return {"2026-06-01": dict(closes), "2026-07-24": {k: v * 1.1 for k, v in closes.items()}}


@pytest.fixture(autouse=True)
def _clean_cache():
    lp._PANEL_CACHE.clear()
    yield
    lp._PANEL_CACHE.clear()


def test_the_cache_key_holds_the_loader_object_not_its_address():
    """The structural invariant, stated so it cannot be regressed silently."""
    panel = _panel(A=100.0, B=100.0, SPY=100.0)
    loader = _loader(panel)

    lp._load_closes_panel(_BUCKET, _DATES, _HORIZON, None, loader)

    (key,) = lp._PANEL_CACHE
    assert any(k is loader for k in key), (
        "the closes-panel cache key must contain the loader OBJECT — holding it "
        "is what keeps its address out of circulation"
    )
    assert not any(isinstance(k, int) and k == id(loader) for k in key), (
        "the cache key still carries id(loader); CPython re-uses freed addresses, "
        "so this silently serves one caller's panel to another"
    )


def test_two_simultaneously_live_loaders_get_their_own_panels():
    """Same bucket, dates, horizon and universe — only the loader differs."""
    with_spy = _panel(A=100.0, B=100.0, SPY=100.0)
    without_spy = _panel(A=100.0, B=100.0)

    first = lp._load_closes_panel(_BUCKET, _DATES, _HORIZON, None, _loader(without_spy))
    second = lp._load_closes_panel(_BUCKET, _DATES, _HORIZON, None, _loader(with_spy))

    assert first == without_spy
    assert second == with_spy, (
        "the second caller was served the first caller's panel — the exact "
        "failure that dropped SPY out of topn_alpha_vs_benchmark"
    )


def test_a_cached_loader_stays_alive_so_its_address_cannot_be_recycled():
    """The behavioural half: prove the collision is IMPOSSIBLE, not just absent.

    Under the old key the loader was referenced by nothing once the caller
    dropped it, so the very next lambda allocated could land on its address.
    Under the fix the key holds it, so no later object can. Asserted by trying
    hard to land on the address and failing every time.
    """
    loader = _loader(_panel(A=100.0, SPY=100.0))
    address = id(loader)
    lp._load_closes_panel(_BUCKET, _DATES, _HORIZON, None, loader)

    del loader
    gc.collect()

    later = [_loader({"d": {"x": 1.0}}) for _ in range(200)]
    assert not any(id(obj) == address for obj in later), (
        "a newly allocated loader landed on the cached loader's address — the "
        "cache is not holding a reference to it, so key collisions are live again"
    )


def test_the_id_reuse_this_guards_against_is_real_on_this_interpreter():
    """A guard whose premise nobody has checked is not a guard.

    If CPython ever stopped recycling addresses this whole file would be
    theatre, so measure it rather than asserting it from memory. Not a
    correctness assertion about the cache — an assertion that the hazard the
    cache is defended against still exists.
    """
    hits = 0
    for _ in range(20):
        first = _loader({})
        address = id(first)
        del first
        gc.collect()
        second = _loader({})
        hits += id(second) == address
        del second
    assert hits > 0, (
        "no address was recycled in 20 attempts — if this interpreter no longer "
        "recycles, re-derive whether the identity key is still a hazard before "
        "relaxing anything in this file"
    )
