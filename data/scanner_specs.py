"""Scanner champion/challenger spec registry + shadow-artifact builder
(config#1221 + config#1186).

The scanner is the first funnel stage (candidate generation, ~900 → ~60). To
evaluate and refine it INDEPENDENTLY of the research selection layer, we run it
as a champion/challenger OBSERVE substrate — exactly the pattern the predictor
model-zoo uses for the M slot, and the standing pattern for every
refinement-target module (champion serves live, >=1 challenger runs in shadow,
both scored on realized outcomes, promotion manual + evidence-gated).

- **Champion** (``momentum_sleeve``): the LIVE scanner ranking factor. Ranks the
  liquidity-eligible universe by ``mean(z(momentum_20d), z(return_60d))`` and
  takes the top-N. This is the candidate-gen the config#1186 reconciliation
  found beats the previous tech_score composite on the scanner's OWN long-only
  objective with date-clustered significance (lift +0.080, p=0.013). Promoted
  from shadow champion/challenger to live champion via Operator decision
  2026-07-22 (Option 1) per config#1186 closes-when.
- **Challenger** (``tech_score_momentum``): the PREVIOUS champion, now running
  in shadow for comparison. Ranks by ``tech_score`` (RSI/MACD/MA/momentum
  composite) over the same liquidity-eligible universe, count-matched top-N.

A challenger reuses the live scanner's own gate decisions (the per-ticker
``_last_eval_log`` stashed by ``run_quant_filter``) — so the hard gates
(liquidity, volatility) are held CONSTANT across specs with zero gate
duplication, and only the RANKING signal varies. The sleeve inputs
(``*_zscore``) come already cross-sectionally normalized from
``factor_loading.parquet`` (same source the #1142 shadow uses).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)


def _rank_momentum_sleeve(
    eval_log: list[dict],
    factor_loadings: dict[str, dict[str, float]] | None,
    params: dict,
) -> list[str]:
    """Rank the liquidity-eligible universe by mean(z(momentum_20d),
    z(return_60d)) and return the top-N tickers (count-matched to the live
    scanner's ``momentum_top_n``).

    Eligibility reuses the live scanner's gate decision: any ticker that cleared
    the liquidity floor (``liquidity_pass == 1``) is eligible — we do NOT re-gate
    on ``tech_score`` (that IS the champion's ranking signal, held out of the
    comparison). Names without a factor loading are dropped (can't be scored).
    """
    top_n = params.get("momentum_top_n") or 60
    if not factor_loadings:
        return []
    eligible = [r["ticker"] for r in eval_log if r.get("liquidity_pass") == 1]
    scored: list[tuple[str, float]] = []
    for ticker in eligible:
        fl = factor_loadings.get(ticker)
        if not fl:
            continue
        vals = [
            v for v in (fl.get("momentum_20d_zscore"), fl.get("return_60d_zscore"))
            if v is not None
        ]
        if not vals:
            continue
        scored.append((ticker, sum(vals) / len(vals)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in scored[:top_n]]


@dataclass(frozen=True)
class ScannerSpec:
    """A named candidate-generation build. ``rank`` is ``None`` for the champion
    (the live path is authoritative); challengers carry a pure ranking function
    ``(eval_log, factor_loadings, params) -> ordered top-N ticker list``."""

    name: str
    kind: str  # "champion" | "challenger"
    version: str
    description: str
    rank: Callable[[list[dict], dict | None, dict], list[str]] | None = None


def _rank_tech_score(
    eval_log: list[dict],
    factor_loadings: dict[str, dict[str, float]] | None,
    params: dict,
) -> list[str]:
    """Rank the liquidity-eligible universe by ``tech_score`` and return the
    top-N tickers (count-matched to ``momentum_top_n``).

    Mirrors the pre-cutover ``run_quant_filter`` sorting for the legacy
    momentum path — RSI/MACD/MA/momentum composite score. Now runs in shadow
    as a challenger for comparison against the momentum-sleeve champion.

    Eligibility reuses the live scanner's gate decisions (``liquidity_pass``)
    so the hard gates are held constant across specs. ``factor_loadings`` is
    accepted for API compatibility but unused — ``tech_score`` is a single
    composite, not a z-score blend.
    """
    top_n = params.get("momentum_top_n") or 60
    if not eval_log:
        return []
    eligible = [r for r in eval_log if r.get("liquidity_pass") == 1]
    scored: list[tuple[str, float]] = []
    for rec in eligible:
        ts = rec.get("tech_score")
        if ts is None:
            continue
        scored.append((rec["ticker"], ts))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in scored[:top_n]]


# The registry. Add new candidate-gen builds here as challengers; they are
# scored forever in shadow with no further plumbing (config#1221).
SCANNER_SPECS: dict[str, ScannerSpec] = {
    "momentum_sleeve": ScannerSpec(
        name="momentum_sleeve",
        kind="champion",
        version="v1",
        description="z(momentum_20d)+z(return_60d) over the liquidity-eligible "
        "universe, count-matched top-N (config#1186)",
        rank=_rank_momentum_sleeve,
    ),
    "tech_score_momentum": ScannerSpec(
        name="tech_score_momentum",
        kind="challenger",
        version="v1.0",
        description="momentum-only tech_score composite (RSI/MACD/MA) over the "
        "liquidity-eligible universe, count-matched top-N (config#1186 shadow)",
        rank=_rank_tech_score,
    ),
}


def challenger_specs() -> list[ScannerSpec]:
    return [s for s in SCANNER_SPECS.values() if s.kind == "challenger"]


def champion_spec() -> ScannerSpec | None:
    """Return the single champion spec (kind == "champion")."""
    for s in SCANNER_SPECS.values():
        if s.kind == "champion":
            return s
    return None


def _shadow_artifact(
    spec: ScannerSpec,
    scanner_tickers: list[str],
    live_artifact: dict,
    n_eligible: int,
    n_scored: int,
) -> dict:
    """Build a shadow candidates artifact for ``spec`` parallel to the live
    schema, so a downstream leaderboard can read live + every shadow uniformly.
    population is spec-independent (carried from live); agent_input_set follows
    the live ``population ∪ spec_picks[:50]`` convention."""
    population_tickers = list(live_artifact.get("population_tickers", []))
    agent_input_set = list(
        dict.fromkeys(population_tickers + scanner_tickers[:50])
    )
    return {
        "run_date": live_artifact["run_date"],
        "scanner_version": f"{spec.name}-{spec.version}",
        "spec": {
            "name": spec.name,
            "kind": spec.kind,
            "ranking": spec.description,
        },
        "generated_at": live_artifact.get("generated_at"),
        "population_tickers": population_tickers,
        "scanner_tickers": scanner_tickers,
        "agent_input_set": agent_input_set,
        "filters_applied": live_artifact.get("filters_applied", {}),
        "stats": {
            "universe_size": live_artifact.get("stats", {}).get("universe_size"),
            "post_scanner": len(scanner_tickers),
            "population_size": len(population_tickers),
            "agent_input_size": len(agent_input_set),
            "eligible_universe": n_eligible,
            "spec_scored": n_scored,
        },
    }


def build_shadow_artifacts(
    live_artifact: dict,
    eval_log: list[dict],
    factor_loadings: dict[str, dict[str, float]] | None,
    params: dict,
) -> dict[str, dict]:
    """Build shadow candidate artifacts for every CHALLENGER spec.

    Fail-soft PER SPEC (§61 alarmed carve-out, config#1684): the shadow build
    runs in the scanner Lambda alongside the live candidates artifact; a single
    challenger spec that raises must not take out the live champion or the other
    shadows, so it is omitted PER SPEC — but the failure now lands on an ALARMED
    surface with a consumer (observe_alerts → SNS + flow-doctor forum), not a
    bare WARN that let the empty-``candidates_shadow/`` class hide for weeks
    (config#1403). Returns ``{spec_name: artifact}``.
    """
    n_eligible = sum(1 for r in eval_log if r.get("liquidity_pass") == 1)
    out: dict[str, dict] = {}
    for spec in challenger_specs():
        try:
            tickers = spec.rank(eval_log, factor_loadings, params)
            out[spec.name] = _shadow_artifact(
                spec, tickers, live_artifact, n_eligible, len(tickers)
            )
        except Exception as exc:  # noqa: BLE001 — shadow is best-effort observability
            logger.warning(
                "[scanner_specs] shadow spec %s failed (non-fatal, live "
                "unaffected): %s", spec.name, exc,
            )
            try:
                from observe_alerts import publish_observe_alert
                publish_observe_alert(
                    f"scanner shadow spec {spec.name} FAILED to emit "
                    f"(non-fatal, live candidates unaffected): {exc}",
                    source=f"scanner:shadow_spec:{spec.name}",
                    dedup_key=f"scanner_shadow_spec_fail:{spec.name}",
                )
            except Exception:  # noqa: BLE001 — alerting is secondary; WARN above is the backstop
                logger.warning(
                    "[scanner_specs] observe_alert publish unavailable for %s "
                    "(WARN log is the backstop)", spec.name,
                )
    return out
