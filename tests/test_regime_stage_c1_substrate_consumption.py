"""Pin the regime-v3 Stage C.1 wiring — macro agent consumes the
quantitative regime substrate as a strong prior in its ReAct prompt.

Three layers tested:

1. ``ArchiveManager.load_regime_substrate`` resolves the canonical
   ``regime/latest.json`` sidecar → dated artifact, and returns
   ``None`` gracefully when the substrate is unavailable.
2. ``agents/macro_agent.py::_format_regime_substrate`` renders the
   substrate as a structured prompt block + emits a fallback message
   when None.
3. The graph topology adds ``load_regime_substrate_node`` between
   ``fetch_data`` and ``macro_economist_node`` (the substrate flows
   into the macro agent via state).

Topology assertions use static AST inspection (per Stage B's lesson —
calling ``build_graph()`` has side effects that pollute other tests).
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = REPO_ROOT / "graph" / "research_graph.py"


def _force_real_module(module_name: str):
    """Force-reload a module if some upstream test patched attributes
    on it (mocked out functions, etc) and never restored them.

    Aggressive — deletes the module from ``sys.modules`` AND reloads.
    Without this, sibling tests that ``patch("agents.macro_agent.X")``
    leave a MagicMock attribute behind, and signature-inspection here
    sees ``(*args, **kwargs)`` instead of the real signature.
    """
    import importlib
    import sys
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


# ---------------------------------------------------------------------------
# ArchiveManager.load_regime_substrate
# ---------------------------------------------------------------------------


def _make_am_with_s3_responses(responses: dict[str, bytes | None]):
    """Build an ArchiveManager with an in-memory S3 client.

    The real ArchiveManager.load_regime_substrate now delegates to
    alpha_engine_lib.eval_artifacts.load_latest_eval_artifact, which
    uses self.s3.get_object directly (boto3-like interface). So we
    install a stub on self.s3 that handles get_object."""
    import io

    from archive.manager import ArchiveManager

    am = ArchiveManager.__new__(ArchiveManager)
    am.bucket = "test-bucket"
    am.db_conn = None

    class _StubS3:
        def get_object(self, *, Bucket: str, Key: str):
            raw = responses.get(Key)
            if raw is None:
                raise KeyError(f"no object at {Bucket}/{Key}")
            body_bytes = raw if isinstance(raw, bytes) else raw.encode("utf-8")
            return {"Body": io.BytesIO(body_bytes)}

    am.s3 = _StubS3()
    return am


def test_load_regime_substrate_resolves_via_sidecar() -> None:
    """latest.json points at a dated artifact; loader returns the dated artifact."""
    sidecar = {
        "run_id": "2605170230",
        "artifact_key": "regime/2605170230.json",
        "calendar_date": "2026-05-17",
        "trading_day": "2026-05-15",
        "schema_version": 1,
        "hmm_argmax": "neutral",
    }
    artifact = {
        "calendar_date": "2026-05-17",
        "trading_day": "2026-05-15",
        "run_id": "2605170230",
        "schema_version": 1,
        "hmm": {"argmax": "neutral", "probs": {"bear": 0.2, "neutral": 0.6, "bull": 0.2}},
        "composite": {"intensity_z": 0.1},
        "bocpd": {"change_signal": False},
        "features": {"vix_level": 17.4},
        "guardrails": {},
    }
    am = _make_am_with_s3_responses({
        "regime/latest.json": json.dumps(sidecar).encode("utf-8"),
        "regime/2605170230.json": json.dumps(artifact).encode("utf-8"),
    })
    result = am.load_regime_substrate()
    assert result == artifact


def test_load_regime_substrate_returns_none_when_sidecar_missing() -> None:
    am = _make_am_with_s3_responses({})  # empty
    assert am.load_regime_substrate() is None


def test_load_regime_substrate_returns_none_when_sidecar_lacks_artifact_key() -> None:
    """Defensive — a malformed sidecar without artifact_key shouldn't
    crash; loader returns None and the macro agent falls back."""
    sidecar = {"run_id": "2605170230"}  # missing artifact_key
    am = _make_am_with_s3_responses({
        "regime/latest.json": json.dumps(sidecar).encode("utf-8"),
    })
    assert am.load_regime_substrate() is None


def test_load_regime_substrate_returns_none_when_artifact_missing() -> None:
    """Sidecar points at a key that doesn't exist (transient S3 hiccup
    or partial publish). Loader returns None."""
    sidecar = {
        "run_id": "2605170230",
        "artifact_key": "regime/2605170230.json",
    }
    am = _make_am_with_s3_responses({
        "regime/latest.json": json.dumps(sidecar).encode("utf-8"),
        # 2605170230.json intentionally missing
    })
    assert am.load_regime_substrate() is None


# ---------------------------------------------------------------------------
# _format_regime_substrate prompt-block rendering
# ---------------------------------------------------------------------------


def test_format_regime_substrate_none_emits_fallback_message() -> None:
    """When substrate is None, the prompt block tells the LLM to
    proceed with its prior behavior (LLM judgment + post-LLM guardrails)."""
    from agents.macro_agent import _format_regime_substrate
    block = _format_regime_substrate(None)
    assert "not available this run" in block
    # Tells the LLM what to do without the substrate
    assert "Proceed" in block or "proceed" in block
    # Macro agent's post-LLM guardrails are still applied — must say so
    assert "guardrail" in block.lower()


def test_format_regime_substrate_renders_full_block() -> None:
    """A populated substrate produces a structured prompt block with
    HMM posteriors, intensity, change signal, guardrail flags, and the
    strong-prior framing instruction."""
    from agents.macro_agent import _format_regime_substrate
    substrate = {
        "run_id": "2605170230",
        "calendar_date": "2026-05-17",
        "trading_day": "2026-05-15",
        "schema_version": 1,
        "hmm": {
            "argmax": "bear",
            "weeks_in_current_state": 3,
            "probs": {"bear": 0.65, "neutral": 0.25, "bull": 0.10},
        },
        "composite": {
            "intensity_z": -1.8,
            "implied_severity": "risk_off",
            "per_feature_z": {
                "vix_level": 2.1,
                "hy_oas_bps": 1.7,
                "spy_20d_return": -1.5,
            },
        },
        "bocpd": {
            "change_signal": True,
            "max_runlength_prob": 0.42,
            "change_confidence": 0.61,
        },
        "guardrails": {
            "vix_bear_breached": True,
            "spy_30d_bear_breached": True,
            "vix_caution_breached": False,
            "spy_30d_caution_breached": False,
            "hy_oas_caution_breached": False,
            "active_severity_floor": "bear",
        },
    }
    block = _format_regime_substrate(substrate)
    # Substantive content
    assert "P(bear)" in block and "0.65" in block
    assert "intensity_z" in block and "-1.80" in block  # signed format
    assert "change_signal" in block.lower()
    assert "True" in block  # bocpd change_signal value
    assert "BEAR threshold breached" in block or "VIX BEAR" in block
    assert "active_severity_floor" in block and "bear" in block
    assert "2605170230" in block  # run_id surfaced
    # Framing instruction — strong prior, not authority
    assert "STRONG PRIOR" in block
    assert "FINAL authority" in block


# ---------------------------------------------------------------------------
# Drawdown de-risk leg (3rd ensemble leg, predictor #176/#179)
# ---------------------------------------------------------------------------


def _substrate_with_drawdown(*, excess_available: bool) -> dict:
    excess = (
        {"available": True, "tier": "alpha_bleed",
         "nav_drawdown": -0.118, "excess_depth": 0.041,
         "regime_contribution": "bear"}
        if excess_available else
        {"available": False, "tier": "risk_on", "nav_drawdown": None,
         "excess_depth": None, "regime_contribution": None}
    )
    return {
        "run_id": "2605190230", "calendar_date": "2026-05-19",
        "trading_day": "2026-05-19", "schema_version": 1,
        "hmm": {"argmax": "neutral", "weeks_in_current_state": 50,
                "probs": {"bear": 0.0, "neutral": 1.0, "bull": 0.0}},
        "composite": {"intensity_z": 0.3, "implied_severity": "neutral",
                      "per_feature_z": {"vix_level": 0.4}},
        "bocpd": {"change_signal": False},
        "guardrails": {"active_severity_floor": None},
        "drawdown": {
            "spy": {"tier": "caution", "drawdown": -0.072, "peak": 600.0,
                    "regime_contribution": "caution"},
            "excess": excess,
        },
        "effective_regime": {
            "effective_regime": "bear" if excess_available else "caution",
            "drivers": {"hmm": "neutral", "drawdown_spy": "caution",
                        "drawdown_excess": (
                            "bear" if excess_available else None)},
        },
    }


def test_format_drawdown_leg_renders_continuous_statement() -> None:
    """The drawdown leg reframes the misleading HMM run-length into a
    continuous, market-grounded statement + composed effective regime."""
    from agents.macro_agent import _format_regime_substrate
    block = _format_regime_substrate(
        _substrate_with_drawdown(excess_available=True))
    # Run-length explicitly demoted to a diagnostic, not market duration.
    assert "filter run-length" in block
    assert "NOT a market-duration statement" in block
    # Continuous SPY + book statement.
    assert "7.2% off its trailing peak" in block
    assert "tier = caution" in block
    assert "11.8% off NAV high-water mark" in block
    assert "4.1pp deeper than the market" in block
    # Composed effective regime + drivers surfaced.
    assert "composed effective_regime = bear" in block
    assert "DETERMINISTIC DRAWDOWN DE-RISK LEG" in block


def test_format_drawdown_leg_portfolio_unavailable_fallback() -> None:
    """NAV unavailable ⇒ SPY leg still renders; excess line states it is
    unavailable; never raises."""
    from agents.macro_agent import _format_regime_substrate
    block = _format_regime_substrate(
        _substrate_with_drawdown(excess_available=False))
    assert "7.2% off its trailing peak" in block
    assert "book-vs-market excess: UNAVAILABLE" in block
    assert "the SPY leg still acts" in block
    assert "composed effective_regime = caution" in block


def test_format_drawdown_leg_absent_key_is_byte_identical() -> None:
    """Pre-#176/#179 substrate (no drawdown key) ⇒ the drawdown block is
    omitted entirely — zero behavior change vs the HMM-only path."""
    from agents.macro_agent import (
        _format_drawdown_leg,
        _format_regime_substrate,
    )
    no_dd = {
        "run_id": "2605190230", "trading_day": "2026-05-19",
        "schema_version": 1,
        "hmm": {"argmax": "neutral", "weeks_in_current_state": 3,
                "probs": {"bear": 0.1, "neutral": 0.8, "bull": 0.1}},
        "composite": {"intensity_z": 0.2, "implied_severity": "neutral",
                      "per_feature_z": {}},
        "bocpd": {"change_signal": False},
        "guardrails": {"active_severity_floor": None},
    }
    assert _format_drawdown_leg(no_dd) == ""
    block = _format_regime_substrate(no_dd)
    assert "DETERMINISTIC DRAWDOWN DE-RISK LEG" not in block
    # The run-length is still reframed as a diagnostic even without the
    # drawdown leg (the misleading framing is fixed regardless).
    assert "filter run-length" in block


# ---------------------------------------------------------------------------
# Macro agent threads substrate through reflection wrapper
# ---------------------------------------------------------------------------


def test_run_macro_agent_with_reflection_accepts_regime_substrate_kwarg() -> None:
    """The reflection wrapper must accept ``regime_substrate`` so the
    graph node can thread it through without conditional plumbing."""
    import inspect
    macro_agent = _force_real_module("agents.macro_agent")
    sig = inspect.signature(macro_agent.run_macro_agent_with_reflection)
    assert "regime_substrate" in sig.parameters, (
        "run_macro_agent_with_reflection must accept regime_substrate kwarg "
        "(Stage C.1 graph node passes it from state)."
    )
    assert sig.parameters["regime_substrate"].default is None, (
        "regime_substrate must default to None so Stage A pre-deploy + "
        "non-blocking-Catch failures don't break the call site."
    )


def test_run_macro_agent_accepts_regime_substrate_kwarg() -> None:
    """The primary entry point must accept the substrate too — the
    reflection wrapper passes it through on both initial + retry calls."""
    import inspect
    macro_agent = _force_real_module("agents.macro_agent")
    sig = inspect.signature(macro_agent.run_macro_agent)
    assert "regime_substrate" in sig.parameters
    assert sig.parameters["regime_substrate"].default is None


# ---------------------------------------------------------------------------
# Graph topology — load_regime_substrate_node wired between fetch_data
# and macro_economist_node
# ---------------------------------------------------------------------------


def _build_graph_source() -> str:
    """Extract the body of ``build_graph()`` as source text."""
    source = GRAPH_PATH.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "build_graph":
            return ast.get_source_segment(source, node) or ""
    raise AssertionError("build_graph function not found in research_graph.py")


def test_build_graph_adds_load_regime_substrate_node() -> None:
    body = _build_graph_source()
    assert (
        'graph.add_node("load_regime_substrate_node", load_regime_substrate_node)'
        in body
    ), "build_graph must register load_regime_substrate_node as a graph node."


def test_build_graph_edges_fetch_data_to_load_regime_substrate() -> None:
    body = _build_graph_source()
    assert 'graph.add_edge("fetch_data", "load_regime_substrate_node")' in body, (
        "fetch_data must edge to load_regime_substrate_node (the substrate "
        "fetch sits between data load and macro consumption)."
    )


def test_build_graph_edges_load_regime_substrate_to_macro() -> None:
    body = _build_graph_source()
    # Phase 2.A.3 (scorecard) splices load_scorecard_node between the
    # regime substrate loader and the macro economist. The substrate-
    # before-macro invariant is preserved through the chain
    # load_regime_substrate_node → load_scorecard_node → macro_economist_node.
    assert (
        'graph.add_edge("load_regime_substrate_node", "load_scorecard_node")'
        in body
        and 'graph.add_edge("load_scorecard_node", "macro_economist_node")'
        in body
    ), (
        "load_regime_substrate_node must edge through load_scorecard_node "
        "to macro_economist_node so the substrate (and the prior-cycle "
        "scorecard text) are both in state before macro reads them."
    )


def test_build_graph_no_longer_directly_edges_fetch_to_macro() -> None:
    """Pre-Stage-C the topology was fetch_data → macro_economist_node
    directly. Stage C inserts load_regime_substrate_node between them.
    The old direct edge must NOT remain (would skip substrate load)."""
    body = _build_graph_source()
    assert (
        'graph.add_edge("fetch_data", "macro_economist_node")' not in body
    ), (
        "Stale fetch_data → macro_economist_node edge — Stage C inserts "
        "load_regime_substrate_node between them. Direct edge bypasses "
        "the substrate fetch."
    )


# ---------------------------------------------------------------------------
# load_regime_substrate_node behavior
# ---------------------------------------------------------------------------


def test_load_regime_substrate_node_graceful_when_no_archive_manager() -> None:
    """Defensive — if state somehow lacks archive_manager, the node
    must not crash; it returns None substrate so macro falls back."""
    from graph.research_graph import load_regime_substrate_node
    result = load_regime_substrate_node({})
    assert result == {"regime_substrate": None}


def test_load_regime_substrate_node_returns_substrate_from_archive_manager() -> None:
    """The happy path: archive_manager.load_regime_substrate returns
    a dict; node packages it into the state update."""
    from graph.research_graph import load_regime_substrate_node

    am = MagicMock()
    am.load_regime_substrate.return_value = {
        "run_id": "2605170230",
        "hmm": {"argmax": "bull", "probs": {"bear": 0.1, "neutral": 0.2, "bull": 0.7}},
        "composite": {"intensity_z": 1.4},
        "bocpd": {"change_signal": False},
    }
    state = {"archive_manager": am}
    result = load_regime_substrate_node(state)
    assert result["regime_substrate"]["hmm"]["argmax"] == "bull"
    am.load_regime_substrate.assert_called_once()


def test_load_regime_substrate_node_returns_none_when_loader_returns_none() -> None:
    """When the loader returns None (substrate not yet published), the
    node must propagate None through — NOT raise. Stage C is observe-
    only at the substrate layer; macro agent falls back gracefully."""
    from graph.research_graph import load_regime_substrate_node

    am = MagicMock()
    am.load_regime_substrate.return_value = None
    state = {"archive_manager": am}
    result = load_regime_substrate_node(state)
    assert result == {"regime_substrate": None}
