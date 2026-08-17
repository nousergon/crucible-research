"""Scanner champion/challenger spec registry + shadow-artifact builder
(config#1221 + config#1186).

The scanner is the first funnel stage (candidate generation, ~900 → ~60). To
evaluate and refine it INDEPENDENTLY of the research selection layer, we run it
as a champion/challenger OBSERVE substrate — exactly the pattern the predictor
model-zoo uses for the M slot, and the standing pattern for every
refinement-target module (champion serves live, >=1 challenger runs in shadow,
both scored on realized outcomes, promotion manual + evidence-gated).

- **Champion** (`tech_score_momentum`): the live momentum-only ``tech_score``
  scanner. Its candidates are emitted by the live path
  (``candidates/{date}/candidates.json``) — authoritative, untouched here.
- **Challenger** (`momentum_sleeve`): ranks the SAME liquidity-eligible universe
  by ``mean(z(momentum_20d), z(return_60d))`` and takes the top-N. This is the
  candidate-gen the config#1186 reconciliation found beats the live scanner on
  the scanner's OWN long-only objective with date-clustered significance
  (lift +0.080, p=0.013) while being flat on the cross-sectional rank-IC #1142
  neutralizes — i.e. the scanner SHOULD keep momentum even though the composite
  does not. Emitted to ``candidates_shadow/{spec}/{date}/candidates.json`` and
  scored forward; never touches the live pool until manually promoted.
- **Challenger** (`mom_12_1_sleeve`): the HORIZON challenger. `momentum_sleeve`
  established that the scanner should KEEP momentum; it did not establish at
  which horizon to READ it, and every momentum input the scanner currently
  uses — ``tech_score``'s ``momentum_20d`` term and `momentum_sleeve`'s
  ``mean(z(momentum_20d), z(return_60d))`` — sits at 1 to 3 months. That is
  the short-term-REVERSAL window (Jegadeesh 1990), not the 12-1 skip-month
  window the Jegadeesh-Titman momentum premium is defined over, and the
  scanner's objective is names attractive over ~1 year. This arm ranks on
  ``z(mom_12_1_pct)`` alone, holding eligibility, width and clock constant,
  so the leaderboard isolates the horizon question from the keep-momentum
  question `momentum_sleeve` already answered (alpha-engine-config-I7544).

A challenger reuses the live scanner's own gate decisions (the per-ticker
``_last_eval_log`` stashed by ``run_quant_filter``) — so the hard gates
(liquidity, volatility) are held CONSTANT across specs with zero gate
duplication, and only the RANKING signal varies. The sleeve inputs
(``*_zscore``) come already cross-sectionally normalized from
``factor_loading.parquet`` (same source the #1142 shadow uses).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

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
        vals = [v for v in (fl.get("momentum_20d_zscore"), fl.get("return_60d_zscore")) if v is not None]
        if not vals:
            continue
        scored.append((ticker, sum(vals) / len(vals)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in scored[:top_n]]


def _rank_mom_12_1_sleeve(
    eval_log: list[dict],
    factor_loadings: dict[str, dict[str, float]] | None,
    params: dict,
) -> list[str]:
    """Rank the liquidity-eligible universe by z(mom_12_1_pct) and return the
    top-N tickers (count-matched to the live scanner's ``momentum_top_n``).

    The horizon challenger to ``momentum_sleeve``. That arm ranks on
    mean(z(momentum_20d), z(return_60d)) — 1-month and 3-month returns. Both
    sit inside the window where cross-sectional momentum REVERSES rather than
    persists (Jegadeesh 1990); neither is the 12-1 skip-month horizon the
    Jegadeesh-Titman momentum premium is defined over. This arm holds
    everything else constant (same eligibility, same width, same clock) and
    varies ONLY the momentum horizon, so the leaderboard answers exactly one
    question: at which horizon should the scanner read momentum?

    That the two arms genuinely differ is measured, not assumed — on the
    2026-08-14 snapshot over 901 names, ``mom_12_1_pct`` is Spearman -0.14
    against ``momentum_20d`` and -0.03 against ``return_60d``. The vacuity
    guard (champion-challenger-policy.md §4) would fire if they resolved to
    the same membership; they do not come close.

    Eligibility mirrors ``_rank_momentum_sleeve`` exactly — ``liquidity_pass
    == 1``, no re-gate on ``tech_score`` (that IS the champion's ranking
    signal, held out of the comparison). Names without the loading are
    dropped rather than imputed: a name whose 12-1 momentum is unknown has no
    place in a ranking BY 12-1 momentum, and a neutral fill would quietly
    seed the middle of the cut with names that were never scored.
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
        z = fl.get("mom_12_1_pct_zscore")
        if z is None:
            continue
        scored.append((ticker, z))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in scored[:top_n]]


# Factor-loading columns the SHADOW substrate reads. Deliberately a superset
# of the reader's default tuple (which serves the live, non-fail-soft
# attractiveness path): a column only the challenger arms need is requested
# only where a missing column is survivable. Adding an arm that reads a new
# loading means adding it here, not widening the reader's default.
SHADOW_FACTOR_LOADING_COLS: tuple[str, ...] = (
    "momentum_20d_zscore",
    "return_60d_zscore",
    "mom_12_1_pct_zscore",
    "beta_60d_zscore",
    "size_zscore",
)


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


# The registry. Add new candidate-gen builds here as challengers; they are
# scored forever in shadow with no further plumbing (config#1221).
SCANNER_SPECS: dict[str, ScannerSpec] = {
    "tech_score_momentum": ScannerSpec(
        name="tech_score_momentum",
        kind="champion",
        version="v1.0",
        description="live momentum-only tech_score scanner (run_quant_filter)",
        rank=None,
    ),
    "momentum_sleeve": ScannerSpec(
        name="momentum_sleeve",
        kind="challenger",
        version="v1",
        description="z(momentum_20d)+z(return_60d) over the liquidity-eligible "
        "universe, count-matched top-N (config#1186)",
        rank=_rank_momentum_sleeve,
    ),
    "mom_12_1_sleeve": ScannerSpec(
        name="mom_12_1_sleeve",
        kind="challenger",
        version="v1",
        description="z(mom_12_1_pct) — 12-1 skip-month momentum — over the "
        "liquidity-eligible universe, count-matched top-N. Horizon "
        "challenger to momentum_sleeve (alpha-engine-config-I7544)",
        rank=_rank_mom_12_1_sleeve,
    ),
}


def challenger_specs() -> list[ScannerSpec]:
    return [s for s in SCANNER_SPECS.values() if s.kind == "challenger"]


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
    agent_input_set = list(dict.fromkeys(population_tickers + scanner_tickers[:50]))
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
) -> tuple[dict[str, dict], dict[str, str]]:
    """Build shadow candidate artifacts for every CHALLENGER spec.

    Fail-soft PER SPEC (§61 alarmed carve-out, config#1684): the shadow build
    runs in the scanner Lambda alongside the live candidates artifact; a single
    challenger spec that raises must not take out the live champion or the other
    shadows, so it is omitted PER SPEC — but the failure now lands on an ALARMED
    surface with a consumer (observe_alerts → SNS + flow-doctor forum), not a
    bare WARN that let the empty-``candidates_shadow/`` class hide for weeks
    (config#1403).

    Returns ``(artifacts, errors)``:

    - ``artifacts`` is ``{spec_name: artifact}`` for every spec that ranked
      successfully this cycle (unchanged contract).
    - ``errors`` is ``{spec_name: reason}`` for every spec whose ``rank()``
      raised this cycle — mirrors ``producers/runner.py::run_challengers``'s
      own ``(written, errors)`` return shape.

    ``errors`` is the input the orchestration layer
    (``data/scanner_orchestrator.py`` / ``lambda/scanner_handler.py``) uses to
    write an explicit ``scanner_shadow_status.v1`` MISS record via
    :func:`build_shadow_status_record`, mirroring
    ``producers/experiment_record.py``'s ``experiment_record.v1`` pattern
    (config#6428, champion-challenger-policy.md §3): "A cycle where an arm
    produces no output is recorded as a miss, not omitted. Silent absence and
    a genuine zero must never render identically." This WARN + alarm path is
    unchanged and remains the primary paging surface; the status record is
    ADDITIVE — a durable record alongside it, never a replacement.
    """
    n_eligible = sum(1 for r in eval_log if r.get("liquidity_pass") == 1)
    out: dict[str, dict] = {}
    errors: dict[str, str] = {}
    for spec in challenger_specs():
        try:
            tickers = spec.rank(eval_log, factor_loadings, params)
            out[spec.name] = _shadow_artifact(spec, tickers, live_artifact, n_eligible, len(tickers))
        except Exception as exc:  # noqa: BLE001 — shadow is best-effort observability
            logger.warning(
                "[scanner_specs] shadow spec %s failed (non-fatal, live unaffected): %s",
                spec.name,
                exc,
            )
            errors[spec.name] = str(exc)
            try:
                from observe_alerts import publish_observe_alert

                publish_observe_alert(
                    f"scanner shadow spec {spec.name} FAILED to emit (non-fatal, live candidates unaffected): {exc}",
                    source=f"scanner:shadow_spec:{spec.name}",
                    dedup_key=f"scanner_shadow_spec_fail:{spec.name}",
                )
            except Exception:  # noqa: BLE001 — alerting is secondary; WARN above is the backstop
                logger.warning(
                    "[scanner_specs] observe_alert publish unavailable for %s (WARN log is the backstop)",
                    spec.name,
                )
    return out, errors


def build_shadow_status_record(
    spec: ScannerSpec,
    run_date: str,
    *,
    shadow_candidates_key: str | None = None,
    error: str | None = None,
) -> dict:
    """Explicit per-spec, per-cycle status record for the scanner shadow slot
    — ``scanner_shadow_status.v1`` — mirroring
    ``producers/experiment_record.py::build_challenger_experiment_record``'s
    vocabulary (config#6428, champion-challenger-policy.md §3).

    Called by the orchestration layer (``lambda/scanner_handler.py``) AFTER
    the real ``candidates_shadow/`` write has been attempted for this spec —
    exactly when the producer slot's own ``run_challengers`` calls
    ``build_challenger_experiment_record`` — so ``status`` reflects the TRUE
    final outcome (ranked AND persisted, or not) rather than asserting an
    action that never happened. ``shadow_candidates_key`` is the S3 key
    ``write_shadow_candidates_artifact`` returned for a successful write
    (``None`` when ``rank()`` raised OR the subsequent S3 write failed —
    ``error`` then carries why, becoming the artifact's honest ``absent``
    reason)."""
    if shadow_candidates_key is not None:
        artifact = {
            "name": "shadow_candidates",
            "status": "emitted",
            "key": shadow_candidates_key,
        }
        status = "complete"
    else:
        artifact = {
            "name": "shadow_candidates",
            "status": "absent",
            "reason": error or "challenger spec did not emit a candidate set this cycle",
        }
        status = "failed"
    return {
        "schema_version": 1,
        "spec": spec.name,
        "kind": spec.kind,
        "run_date": run_date,
        "status": status,
        "artifacts": [artifact],
    }
