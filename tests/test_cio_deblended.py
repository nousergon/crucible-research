"""De-blended CIO orchestration (L4564 Phase B).

Locks the deterministic sector-neutral stock-quality construction (rubric
per-sector bias stripped in code via a trailing-sector baseline) + the
flag-gated prompt path. The OFF path must be byte-identical to the legacy
prompt so the merge is inert; the ON path renders the sector-neutral rank and
loads the v1.5.0 template.
"""
from __future__ import annotations

import pytest

from agents.investment_committee.ic_cio import (
    _build_cio_prompt,
    _pct_rank,
    build_sector_neutral_quality_map,
    compute_sector_neutral_quality,
)


def _candidate(ticker, team, quant=70.0, qual=70.0, conv=70):
    return {
        "ticker": ticker, "team_id": team,
        "quant_score": quant, "qual_score": qual, "conviction": conv,
        "bull_case": "b", "bear_case": "r", "catalysts": ["c1"],
        "sector": team,
    }


# ── pure helpers ─────────────────────────────────────────────────────


def test_pct_rank_ties_averaged():
    r = _pct_rank({"a": 10.0, "b": 10.0, "c": 30.0})
    # a,b tie at ranks 1,2 → avg 1.5/3 = 0.5; c rank 3 → 1.0
    assert r["a"] == pytest.approx(0.5)
    assert r["b"] == pytest.approx(0.5)
    assert r["c"] == pytest.approx(1.0)


def test_sector_neutral_zscore_when_enough_prior():
    # Tech has 6 prior scores → z-score; q=85 vs prior mean/std.
    prior = {"Tech": [50.0, 55.0, 60.0, 65.0, 70.0, 75.0]}
    cq = {"AAA": ("Tech", 85.0)}
    out = compute_sector_neutral_quality(cq, prior, k_min=6)
    s = [50.0, 55.0, 60.0, 65.0, 70.0, 75.0]
    mu = sum(s) / len(s)
    sd = (sum((x - mu) ** 2 for x in s) / (len(s) - 1)) ** 0.5
    assert out["AAA"] == pytest.approx((85.0 - mu) / sd)


def test_sector_neutral_cold_start_falls_back_to_pool_rank():
    # Defensives has < k_min prior → both candidates fall back to the
    # within-pool percentile rank of their raw quality.
    prior = {"Tech": [50.0] * 6}
    cq = {"HI": ("Defensives", 90.0), "LO": ("Defensives", 40.0)}
    out = compute_sector_neutral_quality(cq, prior, k_min=6)
    assert out["HI"] == pytest.approx(1.0)   # top of pool
    assert out["LO"] == pytest.approx(0.5)   # bottom of 2 → 1/2


def test_sector_neutral_mixed_pool_and_zscore():
    prior = {"Tech": [60.0, 62.0, 64.0, 66.0, 68.0, 70.0]}  # ≥ k_min
    cq = {"T1": ("Tech", 80.0), "D1": ("Defensives", 99.0)}  # D1 cold-start
    out = compute_sector_neutral_quality(cq, prior, k_min=6)
    assert out["T1"] > 1.0          # z-scored, well above the sector mean
    assert 0.0 < out["D1"] <= 1.0   # pool-rank fallback


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, *_a, **_k):
        self._last = self._rows
        return self

    def fetchall(self):
        return self._rows


def test_build_map_sources_weighted_base_and_sector():
    candidates = [_candidate("AAA", "Tech"), _candidate("BBB", "Tech")]
    theses = {
        "AAA": {"weighted_base": 80.0, "sector": "Tech"},
        "BBB": {"weighted_base": 60.0, "sector": "Tech"},
    }
    # 6 prior Tech scores → z-score path exercised.
    conn = _FakeConn([("Tech", float(x)) for x in (50, 55, 60, 65, 70, 75)])
    out = build_sector_neutral_quality_map(candidates, theses, conn, "2026-06-08")
    assert set(out) == {"AAA", "BBB"}
    assert out["AAA"] > out["BBB"]  # higher weighted_base → higher neutral score


def test_build_map_empty_when_no_weighted_base():
    candidates = [_candidate("AAA", "Tech")]
    out = build_sector_neutral_quality_map(candidates, {}, _FakeConn([]), "2026-06-08")
    assert out == {}


def test_load_prior_sector_scores_db_error_is_graceful():
    from agents.investment_committee.ic_cio import load_prior_sector_scores

    class _Boom:
        def execute(self, *_a, **_k):
            raise RuntimeError("db gone")

    assert load_prior_sector_scores(_Boom(), "2026-06-08") == {}
    assert load_prior_sector_scores(None, "2026-06-08") == {}


# ── prompt rendering (flag-gated path) ───────────────────────────────
#
# These mock ``load_prompt`` so they're HERMETIC: the de-blended template
# (``ic_cio_evaluation_deblended``) lives in the private alpha-engine-config
# repo, which research CI checks out at MAIN — the v1.5.0 file isn't there
# until its config PR merges. The candidate-line rendering (the part under
# test) is built in code BEFORE ``.format()``, so a stub template that echoes
# ``{candidates_text}`` exercises it fully while also pinning which template
# name the flag selects.


def _prompt_kwargs():
    return {
        "macro_context": {"market_regime": "neutral"},
        "sector_ratings": {"Tech": {"rating": "overweight", "modifier": 1.1}},
        "population": [],
        "open_slots": 3,
        "exits": [],
        "run_date": "2026-06-08",
    }


class _FakePrompt:
    """Minimal stand-in for the loaded Prompt — echoes candidates_text and
    records the template name it was loaded as."""

    def __init__(self, name):
        self.name = name

    def format(self, **kw):
        return f"[template:{self.name}]\n{kw['candidates_text']}"


@pytest.fixture
def fake_load_prompt(monkeypatch):
    names: list[str] = []

    def _fake(name):
        names.append(name)
        return _FakePrompt(name)

    monkeypatch.setattr(
        "agents.investment_committee.ic_cio.load_prompt", _fake,
    )
    return names


def test_prompt_off_path_has_no_sector_neutral_line(fake_load_prompt):
    candidates = [_candidate("AAA", "Tech")]
    out = _build_cio_prompt(candidates, **_prompt_kwargs(), deblended=False)
    assert "Sector-Neutral Quality" not in out
    assert "AAA [Tech]" in out
    assert fake_load_prompt == ["ic_cio_evaluation"]  # legacy template selected


def test_prompt_deblended_renders_rank_and_loads_v15_template(fake_load_prompt):
    candidates = [_candidate("AAA", "Tech"), _candidate("BBB", "Tech")]
    snq = {"AAA": 2.5, "BBB": -1.0}  # AAA strongly above sector, BBB below
    out = _build_cio_prompt(
        candidates, **_prompt_kwargs(),
        deblended=True, sector_neutral_quality=snq,
    )
    # The de-blended template is selected and the per-candidate rank renders.
    assert fake_load_prompt == ["ic_cio_evaluation_deblended"]
    assert "Sector-Neutral Quality:" in out
    assert "100/100" in out  # AAA is top of the 2-name pool → pct rank 1.0
    # AAA's rank line precedes BBB's (AAA is the stronger neutral quality).
    assert out.index("AAA [Tech]") < out.index("BBB [Tech]")


def test_prompt_deblended_handles_missing_quality(fake_load_prompt):
    # A candidate absent from the map renders "n/a", never raises.
    candidates = [_candidate("AAA", "Tech"), _candidate("ZZZ", "Tech")]
    out = _build_cio_prompt(
        candidates, **_prompt_kwargs(),
        deblended=True, sector_neutral_quality={"AAA": 1.0},
    )
    assert "ZZZ [Tech] — Sector-Neutral Quality: n/a" in out


def test_flag_default_off():
    # No env var + no scoring.yaml entry → the flag defaults False (inert merge).
    import config
    assert config.CIO_DEBLENDED_ORCHESTRATION is False
