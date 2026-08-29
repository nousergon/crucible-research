"""Scanner champion/challenger spec registry + shadow-artifact builder
(config#1221 + config#1186).

The scanner is the first funnel stage (candidate generation, ~900 → ~60). To
evaluate and refine it INDEPENDENTLY of the research selection layer, we run it
as a champion/challenger OBSERVE substrate — exactly the pattern the predictor
model-zoo uses for the M slot, and the standing pattern for every
refinement-target module (champion serves live, >=1 challenger runs in shadow,
both scored on realized outcomes, promotion manual + evidence-gated).

- **Champion** (`momentum_sleeve`): the LIVE candidate ranking —
  ``mean(z(momentum_20d), z(return_60d))`` over the liquidity-eligible
  universe, count-matched to ``momentum_top_n``. Live since the 2026-07-22
  ``config#1186`` Option A cutover, which promoted it over the ``tech_score``
  gate on measured lift (+0.080, p=0.013, date-clustered) on the scanner's own
  long-only objective. Its candidates are emitted by the live path
  (``candidates/{date}/candidates.json``), which applies
  ``SCANNER_SPECS[LIVE_CHAMPION].rank`` — this registry entry IS the live
  ranking, not a description of one (alpha-engine-config-I7808).
- **Challenger** (`tech_score_gate`): the DISPLACED INCUMBENT — ``tech_score``
  (RSI / MACD / MA50 / MA200 / 20-day momentum, equally weighted) over the
  rows the momentum path admitted, count-matched. This is what the live
  scanner ranked on before 2026-07-22. It is registered as a scored arm so
  that promotion stays MEASURED forward rather than resting on a single
  backtest run under a benchmark convention that has since been corrected.
- **Challenger** (`mom_12_1_sleeve`): the HORIZON challenger. The champion
  established that the scanner should KEEP momentum; it did not establish at
  which horizon to READ it, and every momentum input the scanner currently
  uses — ``tech_score``'s ``momentum_20d`` term and the champion's
  ``mean(z(momentum_20d), z(return_60d))`` — sits at 1 to 3 months. That is
  the short-term-REVERSAL window (Jegadeesh 1990), not the 12-1 skip-month
  window the Jegadeesh-Titman momentum premium is defined over, and the
  scanner's objective is names attractive over ~1 year. This arm ranks on
  ``z(mom_12_1_pct)`` alone, holding eligibility, width and clock constant,
  so the leaderboard isolates the horizon question from the keep-momentum
  question the champion already answered (alpha-engine-config-I7544).

**The champion is never also a challenger.** Between 2026-07-22 and
2026-08-20 this registry declared a champion named ``tech_score_momentum``
whose description named a ranking the live path had stopped using, while
``momentum_sleeve`` — the ranking the live path had actually adopted — was
still registered as a challenger. The scanner leaderboard loads the champion
from the live artifact, so it scored an arm against itself for four weeks and
alerted daily. ``assert_registry_coherent`` now makes that state unreachable
at import time. See ``SCANNER_CONTRACT.md`` §3.

A challenger reuses the live scanner's own gate decisions (the per-ticker
``_last_eval_log`` stashed by ``run_quant_filter``) — so the hard gates
(liquidity, volatility) are held CONSTANT across specs with zero gate
duplication, and only the RANKING signal varies. The sleeve inputs
(``*_zscore``) come already cross-sectionally normalized from
``factor_loading.parquet`` (same source the #1142 shadow uses).
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, replace

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


def _rank_tech_score(
    eval_log: list[dict],
    factor_loadings: dict[str, dict[str, float]] | None,
    params: dict,
) -> list[str]:
    """Rank the momentum-path-eligible universe by ``tech_score`` descending
    and return the top-N tickers (count-matched to ``momentum_top_n``).

    The DISPLACED INCUMBENT. This reproduces exactly what
    ``data.scanner.run_quant_filter`` emitted as the live cut before the
    2026-07-22 ``config#1186`` cutover: ``momentum_candidates`` sorted by
    ``tech_score`` descending, sliced at ``momentum_top_n``.

    Eligibility is ``scan_path == "momentum"`` — the rows that cleared the
    momentum path's own gates (liquidity floor, price floor, ``tech_score_min``,
    the MA200 floor and the momentum-path volatility ceiling). That is a
    NARROWER set than the sibling sleeve arms' ``liquidity_pass == 1``, and
    deliberately so: those arms hold out ``tech_score`` because it is the
    signal under test for THIS arm, whereas this arm is the incumbent rule in
    full, gates included. Re-deriving the gates here instead would make the
    arm a reconstruction of the incumbent rather than the incumbent.

    Ties break on ticker so the ranking is deterministic across runs — an
    arbitrary tie order would show up on the leaderboard as membership churn
    the arm did not actually produce.

    ``factor_loadings`` is unused: ``tech_score`` comes from the eval log, not
    the factor store. The parameter is part of the ``ScannerSpec.rank``
    signature, which every arm shares so the substrate can call them uniformly.
    """
    top_n = params.get("momentum_top_n") or 60
    scored: list[tuple[str, float]] = []
    for row in eval_log:
        if row.get("scan_path") != "momentum":
            continue
        score = row.get("tech_score")
        ticker = row.get("ticker")
        if not ticker or not isinstance(score, (int, float)):
            continue
        scored.append((str(ticker), float(score)))
    scored.sort(key=lambda x: (-x[1], x[0]))
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


# The registry. Exactly one entry is ``kind="champion"`` and ``LIVE_CHAMPION``
# names it; the orchestrator applies THAT entry's ``rank`` rather than
# importing a ranking function directly, so the register and the live path
# cannot drift apart (alpha-engine-config-I7808, SCANNER_CONTRACT.md §3). Add
# new candidate-gen builds here as challengers; they are scored forever in
# shadow with no further plumbing (config#1221).
LIVE_CHAMPION = "momentum_sleeve"

SCANNER_SPECS: dict[str, ScannerSpec] = {
    "momentum_sleeve": ScannerSpec(
        name="momentum_sleeve",
        kind="champion",
        version="v1",
        description="LIVE candidate ranking: z(momentum_20d)+z(return_60d) over "
        "the liquidity-eligible universe, count-matched top-N. Promoted from "
        "challenger to champion by the 2026-07-22 config#1186 cutover; the "
        "registry caught up 2026-08-20 (alpha-engine-config-I7808)",
        rank=_rank_momentum_sleeve,
    ),
    "tech_score_gate": ScannerSpec(
        name="tech_score_gate",
        kind="challenger",
        version="v1",
        description="tech_score (RSI/MACD/MA50/MA200/momentum_20d, equally "
        "weighted) over the momentum-path-eligible rows, count-matched top-N. "
        "The incumbent the 2026-07-22 cutover displaced, restored as a scored "
        "arm (alpha-engine-config-I7808)",
        rank=_rank_tech_score,
    ),
    "mom_12_1_sleeve": ScannerSpec(
        name="mom_12_1_sleeve",
        kind="challenger",
        version="v1",
        description="z(mom_12_1_pct) - 12-1 skip-month momentum - over the "
        "liquidity-eligible universe, count-matched top-N. Horizon "
        "challenger to the champion (alpha-engine-config-I7544)",
        rank=_rank_mom_12_1_sleeve,
    ),
}

# Arms retired from the register, kept so a leaderboard reading historical
# ``candidates_shadow/`` objects can say what a name USED to mean rather than
# reporting an unknown spec. Never scored forward.
RETIRED_SPEC_NAMES: dict[str, str] = {
    "tech_score_momentum": (
        "champion label 2026-06 to 2026-08-20. Named the tech_score gate but "
        "resolved to the live artifact, which has been the momentum sleeve "
        "since 2026-07-22 — the vacuous-comparison defect "
        "(alpha-engine-config-I7808). Superseded by champion 'momentum_sleeve' "
        "and challenger 'tech_score_gate'."
    ),
}


# ── The champion pointer (alpha-engine-config-I9273) ─────────────────────────
#
# The spec slot's champion used to move ONLY by a human editing
# ``LIVE_CHAMPION``. That is how the four-week vacuous leaderboard happened
# (alpha-engine-config-I7808): a hand-moved pointer leaves no artifact saying
# when it moved or why, so nothing could compare the register against the live
# path. ``scoring/spec_promotion.py`` is now the writer of this key and
# ``LIVE_CHAMPION`` is demoted to the DEFAULT the pointer falls back to —
# exactly the shape ``scoring/universe_membership.py``'s
# ``CUT_CHAMPION_POINTER_KEY`` / ``live_cut_champion`` already has for the cut
# slot.
SPEC_CHAMPION_POINTER_KEY = "config/scanner_spec_champion.json"

DEFAULT_SPEC_CHAMPION = LIVE_CHAMPION
"""The champion served when the pointer has never been written.

``LIVE_CHAMPION`` remains the register's own declaration and is still what
``assert_registry_coherent`` checks the ``kind`` fields against; it is no
longer the MECHANISM by which a champion changes. A promotion writes the
pointer; a deploy is not required and a hand edit is not the path.
"""

PROMOTABLE_SPECS: tuple[str, ...] = tuple(SCANNER_SPECS)
"""Every arm of the spec slot is eligible to hold the live ranking.

**Brian's ruling 2026-08-29, verbatim: "for the research arm, we should make
all arms promote eligible, including think tank."** The general principle he
stated twice that day is that *an arm that is scored must be able to win* —
measurement without a path to promotion is a leaderboard that decides nothing,
which is the condition alpha-engine-config-I9273 was filed about.

So this tuple is derived from :data:`SCANNER_SPECS` rather than restated: a new
arm is promotable by being registered, and there is no second list to forget.
A genuine exclusion is a DECLARED property carrying a reason —
:data:`DECLARED_INELIGIBLE_SPECS` — never an absence from a list
(ARCHITECTURE §140: a disposition that is only an inference is not a state).
"""

DECLARED_INELIGIBLE_SPECS: dict[str, str] = {}
"""Arms that are SCORED every cycle and may never hold the live ranking.

Empty, and that emptiness is the ruling above rendered as data. An entry here
is a name mapped to the REASON it cannot be promoted, and the promotion engine
copies both onto every decision record, so a reader of the artifact learns that
an arm is excluded and why without opening this file. Asserted at import to be
a subset of :data:`SCANNER_SPECS` — an exclusion naming an arm that does not
exist is a stale rule that reads as a live one.
"""


class ScannerSpecPointerError(RuntimeError):
    """``config/scanner_spec_champion.json`` names something the register cannot
    serve.

    Raised rather than degrading to :data:`DEFAULT_SPEC_CHAMPION`: quietly
    serving the default would mean the promotion engine believes one arm ranks
    the live cut while the scanner applies another — the exact drift
    ``SCANNER_CONTRACT.md`` exists to prevent, and the reason
    ``universe_membership.live_cut_champion`` raises on the same condition.
    """


def register_for_champion(
    champion: str, register: dict[str, ScannerSpec] | None = None
) -> dict[str, ScannerSpec]:
    """A NEW register in which ``champion`` is the champion and every other arm
    is a challenger.

    A promotion CONSTRUCTS the register state; it never mutates one entry's
    ``kind``. Mutating a field can leave two champions (or none) in a module
    whose import-time guard has already run, and the guarantee
    ``assert_registry_coherent`` provides would then hold only for the state the
    process started in. The constructed state is validated here, before any
    caller can see it.
    """
    base = register if register is not None else SCANNER_SPECS
    if champion not in base:
        raise ScannerSpecPointerError(
            f"champion {champion!r} is not a registered scanner spec "
            f"({sorted(base)}) — refusing to construct a register around an arm "
            "that does not exist"
        )
    built = {
        name: replace(spec, kind=("champion" if name == champion else "challenger"))
        for name, spec in base.items()
    }
    assert_register_coherent(built, champion)
    return built


def read_champion_pointer(
    *, bucket: str | None = None, s3_client=None, strict: bool = True
) -> str | None:
    """The arm named by the written pointer, or ``None`` when none is written.

    ``strict`` governs a READ FAILURE only, never a bad value: a pointer that
    exists and names an unregistered arm raises
    :class:`ScannerSpecPointerError` under both settings.

    * ``strict=True`` — the promotion engine's setting. Any read failure other
      than a missing object propagates: an engine that cannot establish which
      arm is currently live must not write a decision claiming it displaced one.
    * ``strict=False`` — the live scanner path's setting, and a deliberate
      fail-soft carve-out under the AGENTS.md rule. (a) The swallowed failure is
      a transient S3 or credential error on a READER. (b) The primary
      deliverable survives because the register default is what the pointer
      names in steady state, and the alternative — raising — stops the trading
      day's candidate ranking for an observability pointer, which
      ``SCANNER_CONTRACT.md`` §5 already rules out for this path. (c) The
      recording surface is the ERROR log below plus
      ``config/apply_audit/scanner_spec_champion/latest.json``: the WRITER does
      not degrade, so a pointer the engine cannot read stops the decision series
      and shows up as a stale audit record.
    """
    b = bucket or os.environ.get("S3_BUCKET")
    if s3_client is None and not b:
        # No bucket is declared, so there is no pointer to read and the register
        # default IS the answer. A configuration fact, not a swallowed failure —
        # and it is what keeps the serving path free of I/O in any context that
        # has not been pointed at a bucket.
        return None
    try:
        if s3_client is None:
            import boto3

            s3_client = boto3.client("s3")
        body = s3_client.get_object(Bucket=b, Key=SPEC_CHAMPION_POINTER_KEY)["Body"].read()
    except Exception as exc:  # noqa: BLE001 — absence is a legitimate state
        if "NoSuchKey" in str(exc) or "NoSuchBucket" in str(exc) or "404" in str(exc):
            return None
        if strict:
            raise
        logger.error(
            "[scanner_specs] %s unreadable (%s) — the live cut keeps the register "
            "default %r this cycle; the promotion engine does NOT degrade and its "
            "audit record is what shows this",
            SPEC_CHAMPION_POINTER_KEY,
            exc,
            DEFAULT_SPEC_CHAMPION,
        )
        return None
    champion = json.loads(body).get("champion")
    if champion not in SCANNER_SPECS:
        raise ScannerSpecPointerError(
            f"{SPEC_CHAMPION_POINTER_KEY} names champion {champion!r}, which is not "
            f"a registered scanner spec ({sorted(SCANNER_SPECS)}). Refusing to "
            "resolve the live ranking from an unvalidated pointer."
        )
    if champion in DECLARED_INELIGIBLE_SPECS:
        raise ScannerSpecPointerError(
            f"{SPEC_CHAMPION_POINTER_KEY} names {champion!r}, which is declared "
            f"ineligible to hold the live ranking: "
            f"{DECLARED_INELIGIBLE_SPECS[champion]}"
        )
    return champion


def live_champion_name(
    *, bucket: str | None = None, s3_client=None, strict: bool = False
) -> str:
    """The arm the live candidate path ranks by this cycle."""
    return read_champion_pointer(
        bucket=bucket, s3_client=s3_client, strict=strict
    ) or DEFAULT_SPEC_CHAMPION


def live_champion_spec(*, bucket: str | None = None, s3_client=None) -> ScannerSpec:
    """The spec whose ``rank`` the live candidate path applies.

    The ONLY supported way for the orchestrator to obtain the live ranking.
    Importing a ``_rank_*`` function directly is what let the live path and this
    register disagree for four weeks (alpha-engine-config-I7808) —
    ``tests/test_scanner_contract.py`` asserts the orchestrator does not do it.

    Resolves the WRITTEN pointer, with the register entry as the default
    (alpha-engine-config-I9273). When the pointer names the default the register
    object itself is returned, identity intact; when it names another arm the
    returned spec comes from a freshly CONSTRUCTED coherent register, so the
    live ranking follows a promotion with no code edit and the "exactly one
    champion" guarantee is re-established rather than assumed.
    """
    name = live_champion_name(bucket=bucket, s3_client=s3_client)
    if name == DEFAULT_SPEC_CHAMPION:
        return SCANNER_SPECS[name]
    return register_for_champion(name)[name]


def challenger_specs_for(champion: str) -> list[ScannerSpec]:
    """Every arm that is not ``champion``, as challengers of a coherent register."""
    return [s for n, s in register_for_champion(champion).items() if n != champion]


def assert_register_coherent(
    register: dict[str, ScannerSpec], live_champion: str
) -> None:
    """Raise ``ValueError`` unless ``register`` can express a real experiment.

    Generalised from the module-global check so a register CONSTRUCTED for a
    promoted champion is held to exactly the same invariants as the declared
    one (alpha-engine-config-I9273) — a promotion must not be able to reach a
    state the import-time guard would have refused.

    Every condition here is one that produced, or would reproduce, the
    four-week vacuous-leaderboard defect:

    * exactly one champion, and ``live_champion`` names it — otherwise
      ``live_champion_spec()`` and the leaderboard's champion can differ;
    * the champion carries a ``rank`` — the live path has to be able to apply
      it, and a ``None`` here is the old "the live path is authoritative,
      trust me" arrangement that had no way to be checked;
    * no challenger shares the champion's ranking callable — that is the
      vacuous comparison itself, and it is cheaper to refuse here than to
      alert on daily forever (champion-challenger-policy.md §4);
    * a retired name is not also live, so the two meanings of a name cannot
      overlap in one leaderboard;
    * every declared-ineligible arm is a registered arm, so an exclusion rule
      cannot go stale while still reading as live.
    """
    champions = [s for s in register.values() if s.kind == "champion"]
    if len(champions) != 1:
        raise ValueError(
            f"SCANNER_SPECS must declare exactly one champion, found {len(champions)}: "
            f"{[s.name for s in champions]}"
        )
    champion = champions[0]
    if champion.name != live_champion:
        raise ValueError(
            f"LIVE_CHAMPION={live_champion!r} does not name the champion entry "
            f"({champion.name!r}) — the live path and the leaderboard would rank differently"
        )
    if champion.rank is None:
        raise ValueError(
            f"champion {champion.name!r} carries no rank function; the live path "
            "applies SCANNER_SPECS[LIVE_CHAMPION].rank and cannot fall back to a description"
        )
    for spec in register.values():
        if spec.name != champion.name and spec.rank is champion.rank:
            raise ValueError(
                f"challenger {spec.name!r} shares the champion's ranking function — "
                "the comparison is vacuous by construction "
                "(champion-challenger-policy.md §4, alpha-engine-config-I7808)"
            )
    overlap = set(RETIRED_SPEC_NAMES) & set(register)
    if overlap:
        raise ValueError(f"spec name(s) both live and retired: {sorted(overlap)}")
    unknown = set(DECLARED_INELIGIBLE_SPECS) - set(register)
    if unknown:
        raise ValueError(
            f"DECLARED_INELIGIBLE_SPECS names unregistered arm(s) {sorted(unknown)} — "
            "an exclusion rule that outlives its arm reads as a live rule and is not one"
        )


def assert_registry_coherent() -> None:
    """The module-global register, checked at import and by the contract tests."""
    assert_register_coherent(SCANNER_SPECS, LIVE_CHAMPION)


assert_registry_coherent()


def challenger_specs() -> list[ScannerSpec]:
    """Challengers of the DECLARED register — the shadow-write set.

    Deliberately not pointer-resolved: shadow artifacts must be written for
    every arm that is not the one the LIVE artifact already covers, and the
    live artifact is produced by ``live_champion_spec()``. Use
    :func:`challenger_specs_for` when a promotion has moved the pointer and the
    shadow set must follow it.
    """
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
