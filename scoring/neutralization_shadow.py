"""OBSERVE shadow for cross-sectional score neutralization — Phase-2 of the
research-edge recovery (config#1142).

The pure MECHANISM (``scoring/neutralize.py``) is wired here into the research
scoring path as an UNCONDITIONAL observability sidecar: whenever the per-name
Barra factor loadings are available for the scored cross-section, we compute the
neutralized ranking and persist a shadow artifact to S3
``decision_artifacts/_neutralization_shadow/{run_date}.json``. This is pure
observation — it NEVER alters the live ``signals.json`` / population / ENTER
selection. It is the validation substrate measured against ``momentum_regime_ic``
(config#1140) before the gated LIVE cutover.

Design contract (per the observe-mode discipline + no-silent-fails carve-out):
  * The OBSERVE computation + S3 write are UNCONDITIONAL — observation is never
    flag-gated. Only the LIVE cutover (using the neutralized score as the live
    ranking) is gated, default OFF.
  * The whole sidecar is fail-soft: any failure (missing loadings, S3 error,
    neutralize edge case) logs a WARN and is swallowed. This hangs off the
    PRIMARY research path AFTER signals.json is already persisted, so a shadow
    failure cannot fail the run — the recording surface is the WARN log.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from scoring.neutralize import NeutralizationConfig, neutralize_scores

logger = logging.getLogger(__name__)

# The Barra factor set the composite is residualized against (config#1142).
# These are the per-name *_zscore loadings emitted by alpha-engine-data's
# feature store (group ``factor_loading``): MOMENTUM (short + medium), BETA,
# SIZE — the unintended factor bet the config#1060 diagnosis pinned.
SHADOW_FACTORS: tuple[str, ...] = (
    "momentum_20d_zscore",
    "return_60d_zscore",
    "beta_60d_zscore",
    "size_zscore",
)

SHADOW_PREFIX = "decision_artifacts/_neutralization_shadow"


def build_shadow_artifact(
    signals: dict[str, dict],
    factor_loadings: dict[str, dict[str, float]],
    *,
    run_date: str,
    factors: tuple[str, ...] = SHADOW_FACTORS,
) -> dict[str, Any]:
    """Compute the neutralized ranking and assemble the shadow artifact (pure).

    Args:
        signals: the live ``signals`` block of the signals payload (ticker ->
            signal dict carrying a ``score``). This is the exact live
            cross-section — read-only here.
        factor_loadings: ticker -> {factor_name: exposure}.
        run_date: trading day key for the artifact.
        factors: factor columns to neutralize against.

    Returns:
        The artifact dict (see module docstring for the schema). Pure — does
        no I/O, mutates nothing.
    """
    # Extract the live composite cross-section (finite scores only — a None
    # score carries no ranking information and neutralize_scores skips it too).
    composite_scores: dict[str, float] = {}
    for ticker, sig in signals.items():
        sc = sig.get("score") if isinstance(sig, dict) else None
        if sc is None:
            continue
        try:
            scf = float(sc)
        except (TypeError, ValueError):
            continue
        composite_scores[ticker] = scf

    cfg = NeutralizationConfig(enabled=True, factors=factors)
    neutralized = neutralize_scores(composite_scores, factor_loadings, cfg)

    per_ticker: dict[str, dict[str, Any]] = {}
    n_with_full = 0
    for ticker, live in composite_scores.items():
        ex = factor_loadings.get(ticker, {}) or {}
        used = {f: ex[f] for f in factors if f in ex}
        if len(used) == len(factors):
            n_with_full += 1
        per_ticker[ticker] = {
            "live_score": round(live, 4),
            "neutralized_score": round(float(neutralized.get(ticker, live)), 4),
            "exposures": {f: round(float(v), 6) for f, v in used.items()},
        }

    return {
        "run_date": run_date,
        "factors_used": list(factors),
        "n_names": len(per_ticker),
        "n_with_full_exposures": n_with_full,
        "tickers": per_ticker,
        "note": (
            "OBSERVE shadow only — does NOT affect live signals.json / "
            "population / ENTER selection (config#1142)."
        ),
    }


def run_neutralization_shadow(
    am: Any,
    run_date: str,
    signals_payload: dict,
    factor_loadings: dict[str, dict[str, float]] | None,
    *,
    factors: tuple[str, ...] = SHADOW_FACTORS,
) -> dict | None:
    """Compute + persist the OBSERVE neutralization shadow (fail-soft).

    Hung off the PRIMARY research path AFTER signals.json is already written.
    Any failure is swallowed with a WARN — the research run's primary deliverable
    must be unaffected (no-silent-fails secondary-observability carve-out: the
    recording surface is the WARN log).

    Returns the artifact dict (for tests / callers) or ``None`` if nothing was
    written (no loadings, no scores, or a swallowed failure).
    """
    try:
        if not factor_loadings:
            logger.warning(
                "[neutralization_shadow] no factor loadings available for %s — "
                "skipping shadow (this is expected until the feature store's "
                "factor_loading group has a snapshot).",
                run_date,
            )
            return None

        signals = (signals_payload or {}).get("signals", {}) or {}
        if not signals:
            logger.warning(
                "[neutralization_shadow] signals payload has no signals for %s "
                "— skipping shadow.", run_date,
            )
            return None

        artifact = build_shadow_artifact(
            signals, factor_loadings, run_date=run_date, factors=factors,
        )
        if artifact["n_names"] == 0:
            logger.warning(
                "[neutralization_shadow] no scored names for %s — skipping write.",
                run_date,
            )
            return None

        body = json.dumps(artifact, indent=2, default=str)
        am._s3_put(f"{SHADOW_PREFIX}/{run_date}.json", body)
        logger.info(
            "[neutralization_shadow] wrote shadow for %s: %d names "
            "(%d with full exposures) against %s",
            run_date, artifact["n_names"], artifact["n_with_full_exposures"],
            list(factors),
        )
        return artifact
    except Exception as e:  # fail-soft: shadow must never break the research run
        logger.warning(
            "[neutralization_shadow] failed for %s (%s) — swallowed; "
            "signals.json is already persisted and unaffected.",
            run_date, e,
        )
        return None
