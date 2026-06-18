"""Tests for scoring/neutralize.py — the Phase-2 cross-sectional neutralization
mechanism (config#1142). Verifies it removes a configured factor's influence
from the score ranking while staying fail-soft + identity-by-default."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring.neutralize import NeutralizationConfig, neutralize_scores  # noqa: E402


def _spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def _cross_section(n=40, seed=0):
    rng = np.random.RandomState(seed)
    tickers = [f"T{i}" for i in range(n)]
    momentum = rng.normal(size=n)
    idio = rng.normal(size=n)
    return tickers, momentum, idio


def test_disabled_is_identity():
    scores = {"A": 10.0, "B": 20.0}
    exp = {"A": {"momentum": 1.0}, "B": {"momentum": -1.0}}
    out = neutralize_scores(scores, exp, NeutralizationConfig(enabled=False, factors=("momentum",)))
    assert out == scores


def test_empty_factors_is_identity():
    scores = {"A": 10.0, "B": 20.0}
    out = neutralize_scores(scores, {}, NeutralizationConfig(enabled=True, factors=()))
    assert out == scores


def test_pure_momentum_score_is_neutralized_out():
    # score == momentum exactly → after neutralization the ranking should no
    # longer track momentum (residual ~ 0 up to rescale).
    tickers, momentum, _ = _cross_section()
    scores = {t: float(m) for t, m in zip(tickers, momentum)}
    exp = {t: {"momentum": float(m)} for t, m in zip(tickers, momentum)}
    out = neutralize_scores(scores, exp, NeutralizationConfig(enabled=True, factors=("momentum",), rescale=False))
    out_vec = np.array([out[t] for t in tickers])
    assert abs(_spearman(out_vec, momentum)) < 0.2  # momentum influence removed


def test_recovers_idiosyncratic_ranking():
    # score = momentum + idiosyncratic. Neutralizing momentum should make the
    # ranking track the idiosyncratic component, not momentum.
    tickers, momentum, idio = _cross_section(seed=3)
    raw = momentum + idio
    scores = {t: float(v) for t, v in zip(tickers, raw)}
    exp = {t: {"momentum": float(m)} for t, m in zip(tickers, momentum)}
    out = neutralize_scores(scores, exp, NeutralizationConfig(enabled=True, factors=("momentum",), rescale=False))
    out_vec = np.array([out[t] for t in tickers])
    assert _spearman(out_vec, idio) > 0.8           # tracks idiosyncratic alpha
    assert abs(_spearman(out_vec, momentum)) < 0.25  # momentum scrubbed


def test_multifactor_momentum_beta_size():
    tickers, momentum, idio = _cross_section(seed=7)
    rng = np.random.RandomState(11)
    beta = rng.normal(size=len(tickers))
    size = rng.normal(size=len(tickers))
    raw = 2 * momentum + beta - size + idio
    scores = {t: float(v) for t, v in zip(tickers, raw)}
    exp = {
        t: {"momentum": float(m), "beta": float(b), "size": float(s)}
        for t, m, b, s in zip(tickers, momentum, beta, size)
    }
    out = neutralize_scores(
        scores, exp,
        NeutralizationConfig(enabled=True, factors=("momentum", "beta", "size"), rescale=False),
    )
    out_vec = np.array([out[t] for t in tickers])
    assert _spearman(out_vec, idio) > 0.8
    for f in (momentum, beta, size):
        assert abs(_spearman(out_vec, f)) < 0.3


def test_rescale_preserves_level_and_spread():
    tickers, momentum, idio = _cross_section(seed=1)
    raw = 5 + 3 * (momentum + idio)
    scores = {t: float(v) for t, v in zip(tickers, raw)}
    exp = {t: {"momentum": float(m)} for t, m in zip(tickers, momentum)}
    out = neutralize_scores(scores, exp, NeutralizationConfig(enabled=True, factors=("momentum",), rescale=True))
    out_vec = np.array([out[t] for t in tickers])
    raw_vec = np.array(list(scores.values()))
    # Tolerance accounts for the 4-decimal rounding of output scores; the
    # rescale preserves mean/std exactly pre-rounding.
    assert abs(out_vec.mean() - raw_vec.mean()) < 1e-3
    assert abs(out_vec.std() - raw_vec.std()) < 1e-3


def test_below_min_names_is_identity():
    scores = {f"T{i}": float(i) for i in range(5)}
    exp = {f"T{i}": {"momentum": float(i)} for i in range(5)}
    out = neutralize_scores(scores, exp, NeutralizationConfig(enabled=True, factors=("momentum",), min_names=20))
    assert out == scores


def test_missing_exposures_keep_original_score():
    tickers, momentum, idio = _cross_section(seed=5)
    raw = momentum + idio
    scores = {t: float(v) for t, v in zip(tickers, raw)}
    # Drop exposures for two names — they must survive with their original score.
    exp = {t: {"momentum": float(m)} for t, m in zip(tickers, momentum)}
    del exp[tickers[0]]
    exp[tickers[1]] = {"momentum": float("nan")}
    out = neutralize_scores(scores, exp, NeutralizationConfig(enabled=True, factors=("momentum",), rescale=False))
    assert set(out) == set(scores)
    assert out[tickers[0]] == scores[tickers[0]]
    assert out[tickers[1]] == scores[tickers[1]]


def test_constant_factor_is_identity():
    tickers, _, idio = _cross_section(seed=9)
    scores = {t: float(v) for t, v in zip(tickers, idio)}
    exp = {t: {"momentum": 1.0} for t in tickers}  # constant → uninformative
    out = neutralize_scores(scores, exp, NeutralizationConfig(enabled=True, factors=("momentum",)))
    assert out == scores


def test_from_dict_defaults_to_disabled():
    cfg = NeutralizationConfig.from_dict(None)
    assert cfg.enabled is False and cfg.factors == ()
    cfg2 = NeutralizationConfig.from_dict({"enabled": True, "factors": ["momentum", "beta"]})
    assert cfg2.enabled is True and cfg2.factors == ("momentum", "beta")
