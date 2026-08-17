"""The momentum-horizon challenger arm (alpha-engine-config-I7538).

The champion attractiveness composite puts 40% of its momentum pillar at
horizons of one month or less — the short-term-reversal window — and carries
no 12-1 skip-month term at all. This arm re-ranks the same universe with the
momentum pillar respecified at the 12-1 horizon, changing nothing else.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring.factor_scoring import (  # noqa: E402
    _CHALLENGER_MOMENTUM_DEF,
    _COMPOSITE_DEFS,
    challenger_composite_defs,
    compute_factor_composites,
)
from scoring.universe_membership import (  # noqa: E402
    CHALLENGER_CUT_PREFIX,
    UniverseMembershipError,
    assert_cut_invariants,
    build_universe_membership,
)

# ── The definition itself ────────────────────────────────────────────────────


def test_challenger_drops_every_sub_month_momentum_term():
    """The reversal-window terms are removed, not merely down-weighted.

    A reduced weight on a term pointing the wrong way still points the wrong
    way; the experiment is only clean if the arms disagree about the horizon.
    """
    cols = {c for c, _, _ in _CHALLENGER_MOMENTUM_DEF}
    assert "momentum_5d" not in cols
    assert "momentum_20d" not in cols
    assert "mom_12_1_pct" in cols, "the 12-1 term is the entire point of the arm"


def test_challenger_momentum_weights_sum_to_one():
    assert sum(w for _, w, _ in _CHALLENGER_MOMENTUM_DEF) == pytest.approx(1.0)


def test_challenger_differs_from_champion_in_momentum_only():
    """§4 — hold everything constant except the thing under test.

    An arm that varied two pillars at once could not attribute its own result.
    """
    champ = _COMPOSITE_DEFS
    chal = challenger_composite_defs()
    assert set(champ) == set(chal), "the challenger must not add or drop a pillar"
    differing = [name for name in champ if list(champ[name]) != list(chal[name])]
    assert differing == ["momentum_score"], differing


def test_challenger_defs_track_champion_changes_to_other_pillars(monkeypatch):
    """Derived at call time, never frozen as a literal.

    A frozen copy would silently become a two-variable experiment the first
    time someone retuned value_score.
    """
    import scoring.factor_scoring as fs

    patched = {k: list(v) for k, v in fs._COMPOSITE_DEFS.items()}
    patched["value_score"] = [("pe_ratio", 1.0, True)]
    monkeypatch.setattr(fs, "_COMPOSITE_DEFS", patched)

    assert challenger_composite_defs()["value_score"] == [("pe_ratio", 1.0, True)]


# ── The composite actually reranks ───────────────────────────────────────────


def _panel():
    """Four tickers, one sector, where 12-1 and short-horizon momentum DISAGREE.

    Deliberately anti-correlated: on the live 2026-08-14 cross-section
    mom_12_1_pct is Spearman -0.14 against momentum_20d, so a fixture where
    they agree would let a broken override pass by reproducing the champion.
    """
    return pd.DataFrame({
        "ticker": ["AA", "BB", "CC", "DD"],
        "momentum_20d": [0.40, 0.30, 0.20, 0.10],
        "momentum_5d": [0.40, 0.30, 0.20, 0.10],
        "return_60d": [0.40, 0.30, 0.20, 0.10],
        "return_120d": [0.10, 0.20, 0.30, 0.40],
        "dist_from_52w_high": [-0.30, -0.20, -0.10, -0.05],
        "mom_12_1_pct": [0.05, 0.15, 0.45, 0.90],
        "realized_vol_20d": [0.2, 0.2, 0.2, 0.2],
        "vol_ratio_10_60": [1.0, 1.0, 1.0, 1.0],
        "atr_14_pct": [2.0, 2.0, 2.0, 2.0],
    })


def _fundamentals():
    return pd.DataFrame({
        "ticker": ["AA", "BB", "CC", "DD"],
        "roe": [0.1, 0.1, 0.1, 0.1],
        "debt_to_equity": [1.0, 1.0, 1.0, 1.0],
        "gross_margin": [0.4, 0.4, 0.4, 0.4],
        "current_ratio": [1.5, 1.5, 1.5, 1.5],
        "pe_ratio": [20.0, 20.0, 20.0, 20.0],
        "pb_ratio": [3.0, 3.0, 3.0, 3.0],
        "fcf_yield": [0.05, 0.05, 0.05, 0.05],
        "revenue_growth_3y": [0.1, 0.1, 0.1, 0.1],
        "eps_growth_3y": [0.1, 0.1, 0.1, 0.1],
        "payout_ratio": [0.3, 0.3, 0.3, 0.3],
        "capex_growth_5y": [0.1, 0.1, 0.1, 0.1],
    })


def _momentum_ranking(defs):
    sector_map = dict.fromkeys(["AA", "BB", "CC", "DD"], "Tech")
    out = compute_factor_composites(
        _panel(), _fundamentals(), sector_map, composite_defs=defs
    ).set_index("ticker")
    return list(out["momentum_score"].sort_values(ascending=False).index)


def test_challenger_momentum_ranking_inverts_the_champion_on_this_panel():
    """The two definitions must produce materially different orderings.

    If they did not, the leaderboard would be scoring the champion twice and
    reporting it as a tie.
    """
    champion = _momentum_ranking(_COMPOSITE_DEFS)
    challenger = _momentum_ranking(challenger_composite_defs())
    assert champion[0] == "AA", champion
    assert challenger[0] == "DD", challenger
    assert champion != challenger


def test_champion_ranking_is_unchanged_by_the_override_being_available():
    """Passing no composite_defs must reproduce the pre-change behaviour."""
    sector_map = dict.fromkeys(["AA", "BB", "CC", "DD"], "Tech")
    default = compute_factor_composites(_panel(), _fundamentals(), sector_map)
    explicit = compute_factor_composites(
        _panel(), _fundamentals(), sector_map, composite_defs=_COMPOSITE_DEFS
    )
    pd.testing.assert_frame_equal(default, explicit)


# ── The cut, and its guards ──────────────────────────────────────────────────


def _attr(n=80):
    return {f"T{i:03d}": float(1000 - i) for i in range(n)}


def _challenger_attr(n=80):
    # Reversed ordering — a genuinely different arm.
    return {f"T{i:03d}": float(i) for i in range(n)}


def test_challenger_cuts_are_emitted_count_matched():
    m = build_universe_membership(
        "2026-08-17",
        [f"T{i:03d}" for i in range(60)],
        _attr(),
        challenger_attractiveness=_challenger_attr(),
    )
    assert m["cuts"][f"{CHALLENGER_CUT_PREFIX}20"]["size"] == 20
    assert m["cuts"][f"{CHALLENGER_CUT_PREFIX}60"]["size"] == 60
    assert m["cuts"][f"{CHALLENGER_CUT_PREFIX}20"]["basis"] == "attractiveness_rank_mom121"


def test_challenger_cuts_absent_when_profiles_unavailable():
    """An absent arm is a recorded miss, never a champion fallback.

    Substituting champion scores would enter the leaderboard as a challenger
    that is actually the champion and read as a legitimate tie forever.
    """
    m = build_universe_membership(
        "2026-08-17", [f"T{i:03d}" for i in range(60)], _attr(), challenger_attractiveness=None
    )
    assert not [k for k in m["cuts"] if k.startswith(CHALLENGER_CUT_PREFIX)]


def test_vacuous_challenger_is_a_red_run():
    """§4 vacuity guard — identical membership means the override did not apply.

    Every other assertion still passes in that case, which is exactly why this
    one has to exist.
    """
    with pytest.raises(UniverseMembershipError, match="vacuous challenger"):
        build_universe_membership(
            "2026-08-17",
            [f"T{i:03d}" for i in range(60)],
            _attr(),
            challenger_attractiveness=_attr(),  # identical ranking
        )


def test_short_challenger_table_is_a_red_run():
    """A narrowed arm answers a breadth question, not a horizon one."""
    cuts = {
        "attractiveness_top_20": {"basis": "attractiveness_rank", "size": 20, "tickers": [f"T{i:03d}" for i in range(20)]},
        f"{CHALLENGER_CUT_PREFIX}20": {"basis": "attractiveness_rank_mom121", "size": 12, "tickers": [f"X{i:03d}" for i in range(12)]},
    }
    with pytest.raises(UniverseMembershipError, match="count-match broken"):
        assert_cut_invariants({"cuts": cuts}, "2026-08-17")
