"""
Tests for the Lambda dry-run gate (`dry_run.py` + handler.py auto-gate).

Coverage:
1. Module-level patch + restore round-trip leaves originals identical.
2. `install_dry_run_stubs(archive)` patches all expected agent + graph
   targets and the archive instance methods.
3. Restore is correct under warm-container reuse: a second
   install/restore cycle still recovers original references.
4. Decision-capture env var is force-disabled during stub-pass and
   restored to its prior value (including the unset case).
5. Graph local-name patches are skipped (with a warning) when the graph
   module isn't yet imported — verifies the module guard.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _import_dry_run():
    """Re-import dry_run cleanly each test (it doesn't carry state, but
    importing fresh sidesteps surprises if a sibling test mutated it)."""
    if "dry_run" in sys.modules:
        importlib.reload(sys.modules["dry_run"])
    else:
        import dry_run  # noqa: F401
    return sys.modules["dry_run"]


class TestStubFunctions:
    def test_macro_stub_returns_neutral_regime(self):
        dr = _import_dry_run()
        # config.ALL_SECTORS may not be importable here; stub must still
        # return a dict shape. Provide a minimal config module if missing.
        if "config" not in sys.modules:
            cfg = types.ModuleType("config")
            cfg.ALL_SECTORS = ["Technology", "Healthcare", "Financial"]
            sys.modules["config"] = cfg
        out = dr._stub_run_macro_agent_with_reflection(None, None, {})
        assert out["market_regime"] == "neutral"
        assert "sector_modifiers" in out
        assert all(v == 1.0 for v in out["sector_modifiers"].values())

    def test_quant_stub_returns_ranked_picks(self):
        dr = _import_dry_run()
        out = dr._stub_run_quant_analyst(
            "tech-team",
            ["AAPL", "MSFT", "NVDA"],
            "neutral",
            None,
            None,
            "2026-04-30",
        )
        assert out["team_id"] == "tech-team"
        assert len(out["ranked_picks"]) == 3
        assert all("ticker" in p and "quant_score" in p for p in out["ranked_picks"])

    def test_cio_stub_advances_within_open_slots(self):
        dr = _import_dry_run()
        candidates = [{"ticker": f"T{i}", "combined_score": 60} for i in range(10)]
        out = dr._stub_run_cio(
            candidates,
            None,
            None,
            None,
            open_slots=3,
            exits=[],
            run_date="2026-04-30",
        )
        advanced = [d for d in out["decisions"] if d["decision"] == "ADVANCE"]
        rejected = [d for d in out["decisions"] if d["decision"] == "REJECT"]
        assert len(advanced) == 3
        assert len(rejected) == 7
        assert len(out["entry_theses"]) == 3


class TestInstallRestore:
    def setup_method(self):
        # Build a fake `agents.macro_agent` etc. with sentinel originals
        # so we can verify save/restore without importing real agents.
        # Track real attributes that already exist so teardown can
        # restore them — without this, sentinel MagicMocks leak into
        # the parent package namespace and break subsequent tests that
        # expect the real module attribute (caught 2026-04-30 by
        # tests/test_macro_agent_structured.py interaction).
        self._sentinels = {}
        self._created_modules = []
        self._saved_real_attrs: dict[tuple[str, str], object] = {}
        # Save real ``sys.modules`` entries we're about to overwrite with
        # fakes so teardown can RESTORE them (not just evict). Evicting a
        # real, already-imported module forces a fresh re-import in a
        # later test — that rebinds module globals like
        # ``graph.research_graph.SECTOR_COHERENCE_GATE_ENABLED`` from
        # config, while functions imported by other test modules still
        # close over the OLD module object, so their monkeypatches no
        # longer take effect (order-dependent gate-flag leak — see
        # tests/test_regime_stage_b_graph_topology.py docstring).
        self._saved_sys_modules: dict[str, object] = {}
        for mod_path, attr in [
            ("agents.macro_agent", "run_macro_agent_with_reflection"),
            ("agents.macro_agent", "run_macro_agent"),
            ("agents.sector_teams.quant_analyst", "run_quant_analyst"),
            ("agents.sector_teams.qual_analyst", "run_qual_analyst"),
            ("agents.sector_teams.peer_review", "run_peer_review"),
            ("agents.sector_teams.sector_team", "run_sector_team"),
            ("agents.investment_committee.ic_cio", "run_cio"),
        ]:
            # Ensure parent packages exist
            parts = mod_path.split(".")
            for i in range(1, len(parts) + 1):
                sub_path = ".".join(parts[:i])
                if sub_path not in sys.modules:
                    sys.modules[sub_path] = types.ModuleType(sub_path)
                    self._created_modules.append(sub_path)
            mod = sys.modules[mod_path]
            # Save the existing real attribute (if any) so teardown
            # can restore it. Sentinel MagicMocks must not leak into
            # the real module namespace once the test exits.
            if hasattr(mod, attr) and (mod_path, attr) not in self._saved_real_attrs:
                self._saved_real_attrs[(mod_path, attr)] = getattr(mod, attr)
            sentinel = MagicMock(name=f"{mod_path}.{attr}.original")
            setattr(mod, attr, sentinel)
            self._sentinels[(mod_path, attr)] = sentinel

        # The fake ``graph.research_graph`` shell that used to be built here
        # is gone with the module itself (alpha-engine-config-I7827):
        # ``dry_run._GRAPH_NAME_PATCHES`` is now empty, so there are no
        # late-bound names to sentinel. The agent-module patches above are
        # unchanged and are what the stub installer still acts on.

    def teardown_method(self):
        # Restore real-module attributes BEFORE popping the shells —
        # if the parent module persists in sys.modules with a sentinel
        # MagicMock as one of its attributes, subsequent tests in the
        # session will see the sentinel instead of the real function.
        for (mod_path, attr), original in self._saved_real_attrs.items():
            mod = sys.modules.get(mod_path)
            if mod is not None:
                setattr(mod, attr, original)
        for mod_path in reversed(self._created_modules):
            sys.modules.pop(mod_path, None)
        # Restore any REAL sys.modules entries we shadowed with fakes so
        # later tests keep the same module object their imported
        # functions close over (prevents the order-dependent gate-flag
        # monkeypatch leak).
        for mod_path, original in self._saved_sys_modules.items():
            sys.modules[mod_path] = original

    def test_install_replaces_targets(self):
        dr = _import_dry_run()
        archive = MagicMock()
        archive.upload_db = MagicMock(name="orig_upload_db")
        archive.write_signals_json = MagicMock(name="orig_write_signals_json")
        archive.write_shadow_signals_json = MagicMock(name="orig_write_shadow")

        restore = dr.install_dry_run_stubs(archive)

        # All agent module attrs replaced
        for (mod_path, attr), sentinel in self._sentinels.items():
            current = getattr(sys.modules[mod_path], attr)
            assert current is not sentinel, (
                f"{mod_path}.{attr} not patched"
            )

        # Archive methods patched — including the LIVE producer path's write
        # method (``write_shadow_signals_json``, added alpha-engine-config-I7827:
        # the stub set previously covered only what the champion graph wrote
        # through, leaving ``producers/`` unsuppressed under stub install).
        assert archive.upload_db.__name__ == "<lambda>"
        assert archive.write_signals_json.__name__ == "<lambda>"
        assert archive.write_shadow_signals_json.__name__ == "<lambda>"

        restore()

        # All restored
        for (mod_path, attr), sentinel in self._sentinels.items():
            current = getattr(sys.modules[mod_path], attr)
            assert current is sentinel, (
                f"{mod_path}.{attr} not restored"
            )

    def test_restore_handles_warm_container_replay(self):
        """Two consecutive install/restore cycles both recover originals.

        Models the warm-container Lambda case: stub-pass install + restore,
        then real-pass arrives, then a subsequent invocation does another
        stub-pass + restore. Originals must be intact each time.
        """
        dr = _import_dry_run()
        archive = MagicMock()
        archive.upload_db = MagicMock(name="orig_upload_db")
        archive.write_signals_json = MagicMock(name="orig_write_signals_json")

        # Cycle 1
        restore1 = dr.install_dry_run_stubs(archive)
        restore1()
        for (mod_path, attr), sentinel in self._sentinels.items():
            assert getattr(sys.modules[mod_path], attr) is sentinel

        # Cycle 2 — fresh install on already-restored state
        restore2 = dr.install_dry_run_stubs(archive)
        restore2()
        for (mod_path, attr), sentinel in self._sentinels.items():
            assert getattr(sys.modules[mod_path], attr) is sentinel

    def test_decision_capture_env_force_disabled_then_restored(self):
        dr = _import_dry_run()

        # Case 1: env var was set
        os.environ["ALPHA_ENGINE_DECISION_CAPTURE_ENABLED"] = "true"
        restore = dr.install_dry_run_stubs(None)
        try:
            assert os.environ["ALPHA_ENGINE_DECISION_CAPTURE_ENABLED"] == "false"
        finally:
            restore()
        assert os.environ["ALPHA_ENGINE_DECISION_CAPTURE_ENABLED"] == "true"

        # Case 2: env var was unset
        os.environ.pop("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", None)
        restore = dr.install_dry_run_stubs(None)
        try:
            assert os.environ["ALPHA_ENGINE_DECISION_CAPTURE_ENABLED"] == "false"
        finally:
            restore()
        assert "ALPHA_ENGINE_DECISION_CAPTURE_ENABLED" not in os.environ

    def test_archive_instance_unaffected_when_methods_missing(self):
        """ArchiveManager without the expected methods is tolerated."""
        dr = _import_dry_run()
        archive = object()  # no upload_db / write_signals_json attrs
        restore = dr.install_dry_run_stubs(archive)
        # Should not raise
        restore()


class TestBuildAfterInstallContract:
    """Regression test for the 2026-04-30 dry_run_llm:true bug.

    LangGraph's ``add_node(name, fn)`` captures the function reference at
    build time. Patches to the source module's attribute applied AFTER
    build_graph() do NOT redirect already-bound graph nodes. So the
    contract enforced by `install_dry_run_stubs` is: callers MUST invoke
    `build_graph()` AFTER installing stubs, otherwise direct-bound nodes
    (archive_writer, email_sender) keep their pre-install bindings and
    real S3 writes / email sends fire from the stub-pass.

    This test uses a minimal real StateGraph to assert both halves of the
    contract — build-before-install captures real refs, build-after-install
    captures stubs — so the asymmetry is codified, not just documented.
    """

    def test_build_after_patch_captures_stubs_build_before_does_not(self):
        from typing import TypedDict

        from langgraph.graph import END, START, StateGraph

        class _DemoState(TypedDict, total=False):
            calls: list

        # Module-as-namespace: simulates research_graph.py exposing
        # archive_writer at module scope. We patch via setattr the same
        # way install_dry_run_stubs does on graph.research_graph.
        ns = types.ModuleType("dry_run_demo_module")

        def _real_node(state):
            return {"calls": state.get("calls", []) + ["real"]}

        def _stub_node(state):
            return {"calls": state.get("calls", []) + ["stub"]}

        ns.archive_writer = _real_node

        def _build_demo_graph():
            g = StateGraph(_DemoState)
            g.add_node("aw", ns.archive_writer)
            g.add_edge(START, "aw")
            g.add_edge("aw", END)
            return g.compile()

        # Build BEFORE patching → graph holds real_node reference
        graph_pre = _build_demo_graph()
        ns.archive_writer = _stub_node
        out_pre = graph_pre.invoke({"calls": []})
        assert out_pre["calls"] == ["real"], (
            "Pre-install build must capture original — patch AFTER build "
            "must not affect already-bound nodes (this is the bug class)"
        )

        # Build AFTER patching → graph holds stub_node reference
        graph_post = _build_demo_graph()
        out_post = graph_post.invoke({"calls": []})
        assert out_post["calls"] == ["stub"], (
            "Post-install build must capture the stub — this is the "
            "contract install_dry_run_stubs callers must respect"
        )


# ── TestGraphModuleGuard DELETED (alpha-engine-config-I7827) ──────────────
# It asserted that a late-bound patch whose target module is absent from
# ``sys.modules`` logs a warning instead of raising. The only such target was
# ``graph.research_graph``, deleted with the retired champion arm, so
# ``_GRAPH_NAME_PATCHES`` is empty and the warning can no longer fire for any
# input — the test could only have passed by re-creating the very module whose
# absence it was asserting. The loop and its warning are kept in dry_run.py
# because the mechanism is still correct for the next module that binds an
# agent name at import time; the first such module re-earns this test.
