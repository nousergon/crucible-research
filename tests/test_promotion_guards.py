"""Write-site promotion guards — the two incidents they exist to close, and
the wiring that makes them reachable from the LIVE producer.

Replaces ``tests/test_stub_quarantine.py``. That file tested
``graph/stub_quarantine.py``, whose only importer — ``graph/research_graph.py``
— was deleted by ``crucible-research-PR685``. A guard with no caller passes its
own unit tests forever while protecting nothing, which is why
``alpha-engine-config-I7856`` treats "consumerless guard" as the finding rather
than the cleanup.

Two incidents, both restored onto the live path:

  1. **2026-05-15 promoted stub.** ``signals/2026-05-15/signals.json`` shipped
     synthetic ``dry_run.py`` output PROMOTED AS REAL — GOOG / AFL / AXP / ABT
     / APD / ADBE / AMD each with a ``thesis_summary`` starting
     ``"[DRY-RUN] Strong fundamentals…"``, rendered as real picks by the
     morning email on a run that was NOT ``dry_run_llm``.
  2. **2026-05-04 EOG/NVT.** Actionable signals whose ``sector`` never
     resolved reached the executor, which sized them against a
     sector-exposure bucket that does not exist. The retired graph refused the
     write; nothing did after it was deleted.

The wiring tests are the load-bearing half — they assert the guards are CALLED
at every write site, which is exactly what stopped being true.

Uses the ``monkeypatch`` fixture (NOT ``unittest.mock.patch``).
"""

from __future__ import annotations

import pytest

from dry_run import DRY_RUN_MARKER, install_dry_run_stubs
from scoring.promotion_guards import (
    StubQuarantineError,
    UnresolvedSectorError,
    assert_no_stub_output,
    assert_promotable,
    assert_sectors_resolved,
)

_STUB_THESIS = f"{DRY_RUN_MARKER}] Strong fundamentals, expanding margins."


# ── Part 1: the stub-pass still cannot persist resume keys ────────────────
#
# Retained from the predecessor file. ``install_dry_run_stubs`` is reachable
# from ``dry_run.py``, which is live, so this half keeps a real subject even
# though the graph that consumed the persisted keys is gone.


class _FakeArchive:
    def __init__(self):
        self.persisted: list[tuple] = []

    def upload_db(self, *a, **k):
        self.persisted.append(("upload_db",))

    def write_signals_json(self, *a, **k):
        self.persisted.append(("write_signals_json",))

    def save_sector_team_run(self, run_date, team_id, output):
        self.persisted.append(("save_sector_team_run", team_id))

    def save_agent_run(self, run_date, agent_id, output):
        self.persisted.append(("save_agent_run", agent_id))


def test_install_dry_run_stubs_noops_resume_persistence():
    """The precise 2026-05-15 leak fix: the stub-pass MUST NOT be able to
    write the resume keys a later pass would load and promote."""
    arch = _FakeArchive()
    restore = install_dry_run_stubs(arch)
    try:
        arch.save_sector_team_run("2026-05-15", "technology", {"x": 1})
        arch.save_agent_run("2026-05-15", "cio", {"x": 1})
        arch.write_signals_json("2026-05-15", "", {})
        arch.upload_db("2026-05-15")
    finally:
        restore()

    assert arch.persisted == [], (
        "stub-pass persisted state — the exact leak path that promoted "
        "synthetic [DRY-RUN] theses on 2026-05-15"
    )

    arch.save_sector_team_run("2026-05-15", "technology", {"x": 1})
    assert ("save_sector_team_run", "technology") in arch.persisted


# ── Part 2: the marker scan ───────────────────────────────────────────────


def test_clean_payload_passes():
    assert_no_stub_output(
        {"universe": [{"ticker": "AAPL", "thesis_summary": "Real narrative."}]},
        surface="signals/2026-08-14/signals.json",
    )


def test_marker_at_the_top_level_raises():
    with pytest.raises(StubQuarantineError, match="signals/2026-05-15"):
        assert_no_stub_output(
            {"consolidated_report": _STUB_THESIS},
            surface="signals/2026-05-15/signals.json",
        )


def test_marker_nested_in_a_list_of_rows_raises():
    """The literal 2026-05-15 shape: the marker is several levels down, in a
    per-ticker field, inside a list."""
    payload = {
        "date": "2026-05-15",
        "universe": [
            {"ticker": "MSFT", "thesis_summary": "Real."},
            {"ticker": "GOOG", "thesis_summary": _STUB_THESIS},
        ],
    }
    with pytest.raises(StubQuarantineError) as exc:
        assert_no_stub_output(payload, surface="k")
    assert "GOOG" in str(exc.value) or "thesis_summary" in str(exc.value)


def test_marker_in_a_tuple_and_set_is_reached():
    with pytest.raises(StubQuarantineError):
        assert_no_stub_output({"a": ({"b": _STUB_THESIS},)}, surface="k")
    with pytest.raises(StubQuarantineError):
        assert_no_stub_output({"a": {_STUB_THESIS}}, surface="k")


# ── Part 3: the unresolved-sector gate ────────────────────────────────────


def _row(ticker, rating, sector):
    return {"ticker": ticker, "rating": rating, "sector": sector}


def test_resolved_sectors_pass():
    assert_sectors_resolved(
        {"universe": [_row("AAPL", "ENTER", "Information Technology")]},
        surface="k",
    )


@pytest.mark.parametrize("sector", ["Unknown", "", None, "unknown", "  ", "N/A"])
def test_actionable_row_with_an_unresolved_sector_raises(sector):
    with pytest.raises(UnresolvedSectorError, match="EOG"):
        assert_sectors_resolved({"universe": [_row("EOG", "ENTER", sector)]}, surface="k")


def test_non_actionable_rows_are_not_checked():
    """The envelope emits ``rating: HOLD`` for the whole universe under the
    current champion. Failing the run on a HOLD row nothing acts on would
    make the guard's first production act a false halt."""
    assert_sectors_resolved(
        {"universe": [_row("EOG", "HOLD", "Unknown"), _row("NVT", "EXIT", None)]},
        surface="k",
    )


def test_every_row_collection_key_is_scanned():
    """Derived, not enumerated: the two live producers name the collection
    ``universe`` and ``stocks``, and the archive writer also carries
    ``buy_candidates``. A guard that read only one key would skip the
    producer that names it the other way."""
    for key in ("universe", "stocks", "buy_candidates"):
        with pytest.raises(UnresolvedSectorError):
            assert_sectors_resolved({key: [_row("EOG", "BUY", "Unknown")]}, surface="k")


def test_assert_promotable_runs_both_guards():
    with pytest.raises(StubQuarantineError):
        assert_promotable({"a": _STUB_THESIS}, surface="k")
    with pytest.raises(UnresolvedSectorError):
        assert_promotable({"universe": [_row("EOG", "ENTER", "Unknown")]}, surface="k")


# ── Part 4: the WIRING — the half that actually stopped being true ────────


def test_write_envelope_refuses_a_stub_payload_before_any_put():
    """``scoring/signals_envelope.py::write_envelope`` is the live producer's
    only write site. A guard that raises AFTER the put is not a guard."""
    from scoring import signals_envelope

    puts: list[str] = []

    class _S3:
        def put_object(self, **kw):
            puts.append(kw["Key"])

    with pytest.raises(StubQuarantineError):
        signals_envelope.write_envelope(
            {"universe": [{"ticker": "GOOG", "thesis_summary": _STUB_THESIS}]},
            "2026-08-14",
            target="production",
            bucket="bkt",
            s3_client=_S3(),
        )
    assert puts == [], f"the guard raised only AFTER writing {puts}"


def test_write_envelope_refuses_an_unresolved_sector_before_any_put():
    from scoring import signals_envelope

    puts: list[str] = []

    class _S3:
        def put_object(self, **kw):
            puts.append(kw["Key"])

    with pytest.raises(UnresolvedSectorError):
        signals_envelope.write_envelope(
            {"universe": [_row("EOG", "ENTER", "Unknown")]},
            "2026-08-14",
            target="production",
            bucket="bkt",
            s3_client=_S3(),
        )
    assert puts == []


def test_write_envelope_passes_a_clean_payload():
    from scoring import signals_envelope

    puts: list[str] = []

    class _S3:
        def put_object(self, **kw):
            puts.append(kw["Key"])

    dated, latest = signals_envelope.write_envelope(
        {"universe": [_row("AAPL", "HOLD", "Information Technology")]},
        "2026-08-14",
        target="shadow",
        bucket="bkt",
        s3_client=_S3(),
    )
    assert puts == [dated, latest]


@pytest.mark.parametrize(
    "method,args",
    [
        ("write_signals_json", ("2026-08-14", "2026-08-14T00:00:00Z")),
        ("write_shadow_signals_json", ("thinktank", "2026-08-14", "2026-08-14T00:00:00Z")),
    ],
)
def test_archive_manager_write_sites_are_guarded(monkeypatch, method, args):
    """Both ``ArchiveManager`` signals write sites — the champion key and the
    shadow key the live LLM-backed challengers (Think Tank) land on."""
    from archive.manager import ArchiveManager

    am = ArchiveManager.__new__(ArchiveManager)
    puts: list[str] = []
    monkeypatch.setattr(
        ArchiveManager, "_s3_put", lambda self, key, body: puts.append(key), raising=False
    )

    with pytest.raises(StubQuarantineError):
        getattr(am, method)(*args, {"buy_candidates": [{"t": _STUB_THESIS}]})
    assert puts == [], f"{method} wrote {puts} before the guard raised"


def test_the_retired_graph_guard_module_is_gone():
    """``graph/stub_quarantine.py`` was importable-but-uncalled after
    PR685. Its continued presence is what this issue names as the defect
    class; re-adding it would recreate a guard nothing calls."""
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("graph.stub_quarantine")
