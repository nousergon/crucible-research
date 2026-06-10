"""L4520 slice 4 — consumer contracts for the two backtester-tuned configs
research reads (cross-repo).

Pins the research side of the PIPELINE_CONTRACT.yaml `scoring_weights` and
`research_params` boundaries (producer side: backtester
tests/test_config_writers_producer_contract.py, #315). Both loaders filter
to the keys they know and silently ignore the rest, so a producer key this
side doesn't understand is a tuned param that never applies in live scoring
— the silent-drop class the contract exists to catch at PR time.

Declared sets are hard-coded mirrors of the YAML (per-repo CI can't import
the config repo — the test_scanner_consumer_contract.py precedent).
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_REPO = Path(__file__).resolve().parent.parent

# ── declared producer param sets (mirror PIPELINE_CONTRACT.yaml) ─────────────
SCORING_WEIGHT_PARAMS = {"quant", "qual"}
RESEARCH_PARAMS = {
    "short_interest_buy_threshold_pct", "short_interest_high_threshold_pct",
    "short_interest_buy_boost", "short_interest_high_boost",
    "institutional_min_funds", "institutional_boost",
    "consistency_bullish_dominance", "consistency_bearish_dominance",
    "consistency_low_score", "consistency_high_score",
}


def test_rp_defaults_cover_every_tuned_param():
    # config._load_research_params_from_s3 applies `{k: data[k] for k in
    # _RP_DEFAULTS if k in data}` — a tuned param absent from _RP_DEFAULTS is
    # SILENTLY DROPPED. Every backtester SAFE_PARAMS key must be known here.
    import config as research_config

    missing = RESEARCH_PARAMS - set(research_config._RP_DEFAULTS)
    assert not missing, (
        f"_RP_DEFAULTS is missing backtester-tuned research param(s) "
        f"{sorted(missing)} — the optimizer's update would silently never "
        f"apply. Add the default(s) + plumb the consumer, or remove the key "
        f"from the producer SAFE_PARAMS + PIPELINE_CONTRACT.yaml."
    )


def test_weights_loader_reads_exactly_the_declared_weight_keys():
    # scoring/aggregator._load_weights_from_s3 builds its weights dict from
    # explicit data[...] subscripts — extract them from source and pin against
    # the declared producer weight set, so a producer-side weight rename/add
    # (e.g. a third pillar) fails here instead of silently zeroing the blend.
    tree = ast.parse((_REPO / "scoring" / "aggregator.py").read_text())
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_load_weights_from_s3"
    )
    read_keys = {
        node.slice.value
        for node in ast.walk(fn)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name) and node.value.id == "data"
        and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)
    }
    assert read_keys == SCORING_WEIGHT_PARAMS, (
        f"scoring-weights loader reads {sorted(read_keys)} but the contract "
        f"declares {sorted(SCORING_WEIGHT_PARAMS)} — update PIPELINE_CONTRACT"
        f".yaml + the backtester producer together."
    )


def test_weights_loader_requires_both_keys_before_applying():
    # The loader applies S3 weights only when BOTH declared keys are present
    # (a partial payload falls through to defaults instead of half-applying).
    src = (_REPO / "scoring" / "aggregator.py").read_text()
    assert '"quant" in data and "qual" in data' in src, (
        "weights loader no longer gates on both quant+qual presence — a "
        "partial S3 payload could half-apply and skew the blend."
    )
