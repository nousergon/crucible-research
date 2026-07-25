"""Cross-sectional score neutralization — Phase-2 mechanism (config#1142).

Generic, factor-AGNOSTIC residualizer. Given a cross-section of composite scores
and a matrix of per-name factor exposures, regress score on the exposures and
return the residual — the score's *idiosyncratic* component after removing
intended-neutral factor tilts (momentum, beta, size, ...). This removes the
unintended factor bet that the config#1060 diagnosis pinned as the cause of the
negative research edge (the funnel was implicitly a short-momentum bet that
inverted in narrow-breadth regimes).

SCOPE — this file is the public MECHANISM only:
  * WHICH factors to neutralize against, and the recipe for SOURCING their
    exposures (e.g. computing per-name beta vs SPY, log-market-cap for size),
    are alpha-bearing and land PRIVATE-FIRST in alpha-engine-config/strategy/
    per the divergence policy. Beta and size exposures do not yet exist in the
    research pipeline — sourcing them is part of that recipe phase.
  * The DEFAULT is IDENTITY (``enabled=False`` / no factors), so wiring this
    into the scoring path is a zero-behaviour-change no-op until the private
    recipe turns it on. Every degenerate case (too few names, missing/constant
    exposures, singular design, NaNs) also falls back to identity, fail-soft —
    neutralization must never corrupt or drop a name's score.

Integration contract (recipe phase): call ``neutralize_scores`` on the full
scored cross-section for a run_date AFTER per-ticker ``compute_composite_score``
and BEFORE ranking/threshold application, passing exposures keyed by the same
tickers. Validate via the OBSERVE shadow + the momentum_regime_ic metric
(config#1140) before cutover.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Minimum cross-section size for a stable cross-sectional regression. Below this
# the residual is dominated by fit noise, so we pass scores through unchanged.
_DEFAULT_MIN_NAMES = 20


@dataclass(frozen=True)
class NeutralizationConfig:
    """Config for cross-sectional neutralization (read from the private recipe).

    enabled:  master switch. False -> identity passthrough (the public default).
    factors:  exposure column names to residualize against (e.g. ("momentum",
              "beta", "size")). Empty -> identity.
    rescale:  re-center + re-scale the residuals back to the input score's mean
              and std so downstream BUY/SELL thresholds keep their meaning
              (neutralization changes the RANKING, not the score's units).
    min_names: cross-section floor below which we passthrough.
    """

    enabled: bool = False
    factors: tuple[str, ...] = ()
    rescale: bool = True
    min_names: int = _DEFAULT_MIN_NAMES

    @classmethod
    def from_dict(cls, d: dict | None) -> NeutralizationConfig:
        d = d or {}
        return cls(
            enabled=bool(d.get("enabled", False)),
            factors=tuple(d.get("factors", ()) or ()),
            rescale=bool(d.get("rescale", True)),
            min_names=int(d.get("min_names", _DEFAULT_MIN_NAMES)),
        )


def neutralize_scores(
    scores: dict[str, float],
    exposures: dict[str, dict[str, float]],
    config: NeutralizationConfig,
) -> dict[str, float]:
    """Residualize ``scores`` against the configured factor ``exposures``.

    Args:
        scores: ticker -> composite score (the cross-section for one run_date).
        exposures: ticker -> {factor_name: exposure_value}. Names absent here, or
            with a missing/NaN value for a configured factor, are excluded from
            the regression FIT but still returned with their ORIGINAL score
            (never dropped).
        config: NeutralizationConfig.

    Returns:
        ticker -> neutralized score. Identity copy of ``scores`` whenever
        disabled or any precondition fails (fail-soft).
    """
    out = dict(scores.items())  # identity baseline (never lose a name)

    if not config.enabled or not config.factors:
        return out

    try:
        factors = list(config.factors)
        # Build the fit set: names with a finite score AND a finite value for
        # every configured factor.
        fit_tickers: list[str] = []
        rows: list[list[float]] = []
        y: list[float] = []
        for t, s in scores.items():
            if s is None or not np.isfinite(s):
                continue
            ex = exposures.get(t) or {}
            vals = [ex.get(f) for f in factors]
            if any(v is None or not np.isfinite(v) for v in vals):
                continue
            fit_tickers.append(t)
            rows.append([float(v) for v in vals])
            y.append(float(s))

        if len(fit_tickers) < config.min_names:
            logger.warning(
                "[neutralize] %d usable names < min_names=%d — identity passthrough "
                "(factors=%s)", len(fit_tickers), config.min_names, factors,
            )
            return out

        X = np.asarray(rows, dtype=float)
        yv = np.asarray(y, dtype=float)

        # Standardize each factor cross-sectionally; drop constant factors (zero
        # std would make the column uninformative / blow up the solve).
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        keep = sd > 1e-12
        if not keep.any():
            logger.warning("[neutralize] all factors constant — identity passthrough")
            return out
        Xz = (X[:, keep] - mu[keep]) / sd[keep]

        # Design matrix with intercept; least-squares fit; residual = y - yhat.
        A = np.column_stack([np.ones(len(yv)), Xz])
        coef, _res, rank, _sv = np.linalg.lstsq(A, yv, rcond=None)
        if rank < A.shape[1]:
            logger.warning(
                "[neutralize] rank-deficient design (rank=%d < %d) — identity passthrough",
                rank, A.shape[1],
            )
            return out
        resid = yv - A @ coef

        if config.rescale:
            # Preserve the original score's level + spread so downstream
            # thresholds still apply; only the RANKING within the cross-section
            # changes. Guard against a degenerate (all-equal residual) std.
            r_sd = resid.std()
            if r_sd > 1e-12:
                resid = (resid - resid.mean()) / r_sd * yv.std() + yv.mean()
            else:
                resid = resid - resid.mean() + yv.mean()

        for t, r in zip(fit_tickers, resid, strict=True):
            out[t] = float(round(r, 4))
        logger.info(
            "[neutralize] residualized %d/%d names against %s (rescale=%s)",
            len(fit_tickers), len(scores), [f for f, k in zip(factors, keep, strict=True) if k],
            config.rescale,
        )
        return out
    except Exception as e:  # fail-soft: neutralization must never break scoring
        logger.warning("[neutralize] failed (%s) — identity passthrough", e)
        return dict(scores.items())
