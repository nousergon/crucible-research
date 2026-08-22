"""Write-site refusals for any PROMOTED research artifact.

Two structural guards, both asserted at the S3 write boundary and both
raising BEFORE the ``put_object`` so a bad artifact is never promoted.
They live here — beside ``scoring/signals_envelope.py``, the live
producer — rather than in ``graph/``, because the graph they were born
under is retired (``crucible-research-PR685``,
``alpha-engine-config-I7827``) and a guard whose home package is dead is
a guard nobody calls.

────────────────────────────────────────────────────────────────────────
1. STUB QUARANTINE — ``assert_no_stub_output``
────────────────────────────────────────────────────────────────────────
Ported from ``graph/stub_quarantine.py`` (deleted with this module's
arrival), whose sole importer was the retired ``graph/research_graph.py``.

The incident it closes: ``s3://alpha-engine-research/signals/2026-05-15/
signals.json`` shipped synthetic ``dry_run.py`` stub output PROMOTED AS
REAL — GOOG / AFL / AXP / ABT / APD / ADBE / AMD each carried a
``thesis_summary`` starting ``"[DRY-RUN] Strong fundamentals…"`` and the
morning email rendered them as real picks, on a run that was NOT
``dry_run_llm``. The mechanism was a stub-pass persisting synthetic
sector-team output that the subsequent real pass resumed from.

WHAT CHANGED IN THE PORT, AND WHY. The original guard had two halves:

  (a) a recursive scan for the ``DRY_RUN_MARKER`` substring across every
      promotable surface, and
  (b) an all-sector-teams-present / none-degraded completeness check.

Half (b) is NOT ported. Its subject is ``agents/sector_teams`` — retired
graph code whose deletion is ``alpha-engine-config-I7817``'s sweep — so
porting it would move a check onto the live path whose predicate can
never be non-vacuously true there. Half (a) is path-independent: it
asserts a property of the BYTES, not of the pipeline that made them, so
it transfers to any producer and any write site unchanged.

WHY THE LIVE PATH STILL NEEDS IT even though ``scoring/signals_envelope.py``
is pure-quant (no LLM calls, so no stub installer on its path):
``ArchiveManager.write_shadow_signals_json`` is the shadow-producer write
site, and Think Tank — a live, daily, LLM-backed challenger — writes
through it. A promoted or shadow artifact carrying a synthetic marker is
the same defect at either key, and the guard costs one recursive walk of
a payload we are about to JSON-serialize anyway.

────────────────────────────────────────────────────────────────────────
2. SECTOR RESOLUTION — ``assert_sectors_resolved``
────────────────────────────────────────────────────────────────────────
Restores the 2026-05-04 EOG/NVT guard that died with
``graph/research_graph.py::_validate_signals_payload`` /
``_validate_and_write_signals``. An actionable signal whose ``sector``
never resolved (absent, empty, or the literal ``"Unknown"`` placeholder)
is a data defect that reaches the executor as a real position with the
sector-exposure caps computed against a bucket that does not exist. The
original refused the write, so the executor fell back to the prior day's
signals — a stale-but-coherent artifact, which is strictly better than a
fresh incoherent one.

Scoped to ACTIONABLE rows only (``ENTER`` / ``BUY``). The envelope emits
``rating: "HOLD"`` for the whole universe under the current champion, so
a whole-universe assertion would fail the run on rows nothing acts on;
measured 2026-08-20 against ``signals/latest.json`` (903 rows, run_date
2026-08-14): zero ``Unknown`` sectors and zero actionable rows, so this
guard is armed on a currently-clean surface rather than retro-fitted onto
a breach.
"""

from __future__ import annotations

import logging

from dry_run import DRY_RUN_MARKER

logger = logging.getLogger(__name__)

#: Ratings that put real capital at risk. A ``HOLD``/``EXIT`` row with an
#: unresolved sector is a reporting blemish; an ``ENTER`` row with one is a
#: position sized against a phantom exposure bucket.
ACTIONABLE_RATINGS = frozenset({"ENTER", "BUY"})

#: The placeholder ``scoring/signals_envelope.py`` substitutes when the
#: scanner board carries no sector for a ticker. Treated as unresolved.
_UNRESOLVED_SECTORS = frozenset({"", "unknown", "none", "n/a"})


class StubQuarantineError(RuntimeError):
    """A promoted artifact contained synthetic stub output. The producer
    MUST hard-fail (no signals.json / email / DB write) rather than
    promote synthetic data as real."""


class UnresolvedSectorError(RuntimeError):
    """An actionable signal carried an unresolved sector. The write is
    refused so the executor falls back to the prior day's artifact rather
    than sizing a position against a phantom exposure bucket."""


def _contains_marker(value: object) -> str | None:
    """Recursively scan ``value`` for the ``DRY_RUN_MARKER`` substring.

    Returns a short location/excerpt string on the FIRST hit (so the
    error names where the synthetic text was), or None when clean.
    Walks dicts / lists / tuples / sets; only str leaves are matched.
    """
    if isinstance(value, str):
        if DRY_RUN_MARKER in value:
            return value[:160]
        return None
    if isinstance(value, dict):
        for k, v in value.items():
            hit = _contains_marker(v)
            if hit is not None:
                return f"{k!r}: {hit}"
        return None
    if isinstance(value, (list, tuple, set)):
        for i, item in enumerate(value):
            hit = _contains_marker(item)
            if hit is not None:
                return f"[{i}] {hit}"
        return None
    return None


def assert_no_stub_output(payload: object, *, surface: str) -> None:
    """Refuse to promote ``payload`` if it carries synthetic stub output.

    ``surface`` names the write site for the error message (e.g.
    ``"signals/2026-08-14/signals.json"``). Raises
    :class:`StubQuarantineError` on the first ``DRY_RUN_MARKER`` hit
    anywhere in the structure; a clean return is the ONLY way a write
    proceeds.
    """
    hit = _contains_marker(payload)
    if hit is None:
        return
    msg = (
        f"STUB-QUARANTINE: synthetic dry-run output detected in {surface} "
        f"({DRY_RUN_MARKER!r} marker) — REFUSING the write. A promoted "
        f"artifact may ONLY be produced by a fully-real pass. This is the "
        f"exact 2026-05-15 failure shape (stub thesis promoted as real). "
        f"First hit: {hit!r}"
    )
    logger.error("[promotion_guards] %s", msg)
    raise StubQuarantineError(msg)


def _rows(payload: object) -> list[dict]:
    """Every per-ticker row in an envelope or a signals.json payload.

    Derived from the payload rather than a declared key list: the two
    live producers name the collection ``universe`` (envelope) and
    ``stocks`` (archive writer), and a third naming it something else
    should not silently skip the guard — so both are read and any
    list-of-dicts under either key counts.
    """
    if not isinstance(payload, dict):
        return []
    out: list[dict] = []
    for key in ("universe", "stocks", "buy_candidates"):
        value = payload.get(key)
        if isinstance(value, list):
            out.extend(r for r in value if isinstance(r, dict))
    return out


def assert_sectors_resolved(payload: object, *, surface: str) -> None:
    """Refuse to write when any ACTIONABLE row has an unresolved sector.

    Raises :class:`UnresolvedSectorError` naming every offending ticker.
    Non-actionable rows (``HOLD``, ``EXIT``, unrated) are not checked —
    see the module docstring for why the scope is deliberate.
    """
    offenders = []
    for row in _rows(payload):
        rating = str(row.get("rating") or "").strip().upper()
        if rating not in ACTIONABLE_RATINGS:
            continue
        sector = str(row.get("sector") or "").strip()
        if sector.lower() in _UNRESOLVED_SECTORS:
            offenders.append((row.get("ticker") or "<no-ticker>", sector or None))
    if not offenders:
        return
    msg = (
        f"UNRESOLVED-SECTOR: {len(offenders)} actionable signal(s) in "
        f"{surface} carry no resolved sector: "
        f"{sorted(offenders)} — REFUSING the write. The executor's "
        f"sector-exposure caps would size these against a bucket that "
        f"does not exist; falling back to the prior day's signals is the "
        f"correct degradation (2026-05-04 EOG/NVT guard, restored from "
        f"the retired graph's _validate_signals_payload)."
    )
    logger.error("[promotion_guards] %s", msg)
    raise UnresolvedSectorError(msg)


def assert_promotable(payload: object, *, surface: str) -> None:
    """Both write-site refusals, in the order a breach should be reported.

    The single call site helper — every producer that writes a promoted or
    shadow signals artifact calls THIS, so a new producer cannot pick up
    one guard and miss the other.
    """
    assert_no_stub_output(payload, surface=surface)
    assert_sectors_resolved(payload, surface=surface)
