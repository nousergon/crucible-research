"""config#1039 — factor composite definitions load from the experiment package.

Pins: (a) the baseline stays the validated 6-composite set; (b) absent
config yields the baseline; (c) a package override is honored with tuple
shape restored.
"""
import config as cfg
from scoring import factor_scoring as fs


def test_baseline_composites_are_the_validated_set():
    assert set(fs._BASELINE_COMPOSITE_DEFS) == {
        "quality_score", "momentum_score", "low_vol_score",
        "value_score", "growth_score", "stewardship_score",
    }
    # spot-pin one composite's full recipe
    assert fs._BASELINE_COMPOSITE_DEFS["low_vol_score"] == [
        ("realized_vol_20d", 0.50, True),
        ("vol_ratio_10_60", 0.30, True),
        ("atr_14_pct", 0.20, True),
    ]


def test_active_defs_resolve_from_config_or_baseline():
    if cfg.FACTOR_COMPOSITES_CFG:
        expected = {
            k: [tuple(c) for c in v] for k, v in cfg.FACTOR_COMPOSITES_CFG.items()
        }
    else:
        expected = fs._BASELINE_COMPOSITE_DEFS
    assert fs._COMPOSITE_DEFS == expected


def test_override_restores_tuple_shape(monkeypatch):
    monkeypatch.setattr(
        "config.FACTOR_COMPOSITES_CFG",
        {"quality_score": [["roe", 1.0, False]]},
    )
    resolved = fs._resolve_composite_defs()
    assert resolved == {"quality_score": [("roe", 1.0, False)]}
