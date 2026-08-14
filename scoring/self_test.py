"""self_test.py — the research module's published known-answer SELF-TEST (config-I7262).

WHY THIS EXISTS
---------------
The weekly pipeline's reliability work establishes that each stage **runs**. None
of it establishes that any stage's output is **right**. `sf-pipeline-policy.md`
§2.3a names this as an independent axis: a missing *data artifact* makes a
consumer fail visibly, while a missing *correctness verdict* makes every consumer
succeed **as though the check had passed**.

Counted at `origin/main` on 2026-08-13, `crucible-evaluator` carried three modules
with a numeric self-test and `crucible-backtester` one. `crucible-research` — the
R slot, the signal set everything downstream ranks, sizes and grades — carried
**zero**. That is what this module closes.

The equivalent battery on the backtester/evaluator found a real defect on first
contact: a headline Sharpe annualised by sqrt(365) against a fleet reference using
sqrt(252), a 20.3% divergence between two numbers on one Report Card
(`alpha-engine-config-I7236`). The pre-existing test asserted only `sharpe > 0`.

WHY IT DRIVES THE PRODUCTION FUNCTIONS ON THE DEPLOYED IMAGE
-------------------------------------------------------------
CI proves the code is correct **on a runner**. It does not prove the **deployed
instrument** is. `requirements.txt` pins `nousergon-lib` at a tag and resolves
`numpy`/`pandas` transitively, all baked into the Lambda image at **build** time.
The cross-sectional attractiveness blend — the chokepoint every name's score
passes through — lives in `nousergon_lib.quant.attractiveness`, whose pin moves
independently of this repo. A changed winsorization clip, ddof or percentile tie
rule would move **every** name's score at once: coherently, plausibly, and
entirely invisibly, with CI green throughout.

So every case below calls the **production** function (`nousergon_lib.quant.
attractiveness`, `scoring.composite`, `scoring.aggregator`,
`scoring.leaderboard_scoring`, `scoring.factor_scoring`,
`scoring.attractiveness_trajectory`) on the image's own site-packages, and the
artifact records the resolved version of every numeric distribution actually
loaded. **The library versions in the header are the point** — they are what makes
this an instrument check rather than a code check.

Every `expected` is derived **on paper from the function's definition** and
recomputed here in plain arithmetic — never by calling the code under test, which
would agree with whatever that code ever does.

THE FOUR CASE CLASSES
---------------------
``*_closed_form``   analytically derivable, asserted to 1e-9.
``*_metamorphic``   relations that must hold regardless of the data — affine
                    invariance of a z-score, permutation invariance of a rank,
                    monotonicity of a score in its driving pillar, IC sign flip.
                    They catch what closed-form cannot, because they do not
                    require knowing the right answer.
``*_degenerate``    zero variance, single observation, all-NaN column, empty
                    input — the class that produced `I7237`.
``*_convention``    every lookback, ddof, clip bound and weight asserted against
                    the value ACTUALLY IN USE, with its source of truth named.

THE FINDING THIS BATTERY PINS: THREE REPRESENTATIONS OF "UNDEFINED"
--------------------------------------------------------------------
Measured 2026-08-13 across this repo's scoring surface, an undefined value is
reported in **three mutually incompatible ways**:

===================================================  ==========  ==============
Function                                             Degenerate  Reports
===================================================  ==========  ==============
``nousergon_lib.quant.attractiveness._zscore``       std <= 0    ``0.0``
``scoring/attractiveness_trajectory.py::_zmap``      sd == 0     ``0.0``
``scoring/factor_scoring.py`` composite              all-NaN     ``NaN``
``scoring/leaderboard_scoring.py::_pearson``         zero var    ``None``
===================================================  ==========  ==============

Only the last is safe. A z-score of exactly ``0.0`` is also what a genuinely
at-the-mean name produces, so "this pillar had no cross-sectional spread" and
"this name sits exactly at the median" are indistinguishable to every consumer —
and the blend then treats a *measured nothing* as a *measured average*. This is
the `I7237` class ("a measured-looking zero where the value is undefined") at the
signal layer, and it is filed as **alpha-engine-config-I7272**.

PIN, DO NOT FIX — AND WHAT WAS FIXED
-------------------------------------
Of the three degenerate-input sites this module originally pinned, TWO —
``trajectory_zmap_zero_variance_degenerate`` and
``trajectory_zmap_single_observation_degenerate``, both in
``scoring/attractiveness_trajectory.py::_zmap`` in THIS repo — are now FIXED:
``_zmap`` reports undefined cross-sections as ``None``, and every downstream
consumer in ``build_trajectory`` (the z-scores, the orthogonalized residual,
the percentile cut, both rank orders, the final sort) was audited to drop or
skip ``None`` rather than treat it as a fabricated 0.0. See config-I7272.

One case remains a pinned, NOT-corrected known gap:
``attractiveness_zero_variance_degenerate`` pins
``nousergon_lib.quant.attractiveness._zscore``, which lives in a DIFFERENT
repo — out of scope for this change. Changing it moves every historical
attractiveness score and every gate keyed to one across the whole fleet,
which is Brian's decision, not a patch — the binding precedent is `I7236`,
pinned and filed rather than silently "fixed". Its PASS means "still behaves
as recorded", never "this is correct", and the artifact says so in words so no
reader has to infer it from a green row. ``undefined_representation_divergence_
convention`` remains ``known_gap`` too — it counts the remaining
nousergon_lib gap and pins the new 2-of-3 state.

CONTRACT
--------
``run_self_test()`` **never raises**, and the caller writes its output
unconditionally. A case that DISAGREED is ``FAIL`` (evidence the numbers are
wrong); a case that could not RUN is ``UNKNOWN`` (absence of evidence).
Collapsing the two would make a broken image read as a correctness regression.
Per Brian's ruling 2026-08-13, **a case that exceeds its time budget is FAIL,
never UNKNOWN**.

This module introduces no hard-fail path, no new SF state and no topology change.
An accuracy instrument that can take down the pipeline is a worse defect than the
one it detects.
"""

from __future__ import annotations

import logging
import math
import os
import platform
import signal
import threading
import time
from collections.abc import Callable
from datetime import UTC
from pathlib import Path
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)

SCHEMA = "research_self_test-1.0.0"
COMPONENT = "research"

# The fleet verdict vocabulary, shared with grading/attestation and the console
# so a reader never translates between dialects. (S105 reads "PASS" as a possible
# credential; it is a verdict constant.)
PASS = "PASS"  # noqa: S105
FAIL = "FAIL"
UNKNOWN = "UNKNOWN"

#: ``research/{run_date}/self_test.json`` — beside the run's signals.
_KEY_TEMPLATE = "research/{run_date}/self_test.json"

#: The console component id. Renders through the fleet `checks-envelope` adapter
#: (`nous-ergon-ops/nousergon-console/config.d/fleet-checks.yaml`), which
#: discovers producers by S3 prefix — so this row appears on the console on its
#: first successful publish with no console deploy.
CHECK_ID = "ae-research-self-test"

#: Weekly cadence: this runs with the weekly SF, so a row older than ~1.5 weeks
#: is the console's honest STALE, not a green. Declared, never guessed.
CADENCE_MINUTES = 7 * 24 * 60

_TRACKED_DISTRIBUTIONS = (
    "nousergon-lib",
    "numpy",
    "pandas",
    "krepis",
    "boto3",
)

CASE_TIMEOUT_SECONDS = 30.0

#: 1e-9 absolute, per `alpha-engine-config-I7262`. Not tuned to make the battery
#: pass — observed agreement is ~1e-16 on every unrounded case.
TOLERANCE = 1e-9

# ── the frozen fixtures ─────────────────────────────────────────────────────
# Small, hand-derivable, and deliberately NOT round: a fixture of equal values
# hides a weighting bug and a symmetric one hides a sign bug.

#: Six pillar observations. mean = 3.5, population variance = 35/12 — irrational
#: sd, so an accidental ddof change cannot coincide to within 1e-9.
_PILLAR_VALUES = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

#: Composite-score probe. quant/qual at the default 50/50 weights.
_QUANT = 80.0
_QUAL = 60.0
_SECTOR_MODIFIER = 1.15
_BOOSTS = {"insider": 3.0, "momentum": 2.0}

#: Rank fixture with a deliberate TIE — min-rank semantics are the documented
#: contract ([80, 75, 75, 60] -> [1, 2, 2, 4]) and a tie is the only input that
#: distinguishes min-rank from dense-rank or ordinal.
_RANK_SCORES = (80.0, 75.0, 75.0, 60.0)

#: Spearman fixture: 12 names, all-distinct ranks, ONE adjacent transposition, so
#: the IC is high but not 1.0 — a perfect-correlation fixture would pass under any
#: implementation that merely preserves order, including a broken one.
_IC_N = 12

#: Date-clustered stats probe. Five per-date ICs, deliberately asymmetric.
_PER_DATE_IC = (0.04, 0.02, -0.01, 0.05, 0.03)

#: Affine transform for metamorphic invariance. Both non-trivial.
_AFFINE_SCALE = 7.0
_AFFINE_SHIFT = -13.0


class _CaseTimeout(Exception):
    """Raised when a case exceeds :data:`CASE_TIMEOUT_SECONDS`."""


class Case(NamedTuple):
    """One known-answer, metamorphic, degenerate or convention check.

    ``expected`` is derived by hand from the function's definition; ``compute``
    drives the production code and returns the comparable observed number. They
    are kept apart — rather than ``compute`` returning a bool — so the artifact
    carries both numbers and a later divergence is diagnosable from the artifact
    alone. ``inputs`` is published verbatim: a reader must be able to re-derive
    ``expected`` on paper without opening this file.

    ``known_gap`` marks a case pinning behaviour believed WRONG, so a reader never
    mistakes its PASS for an endorsement. See the module docstring.
    """

    name: str
    description: str
    inputs: dict
    expected: float
    compute: Callable[[], float]
    tolerance: float = TOLERANCE
    known_gap: bool = False
    gap_issue: str | None = None


# ════════════════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════════════════

def _pillar_scores(scale: float = 1.0, shift: float = 0.0) -> dict[str, dict]:
    """Six tickers carrying ONE pillar each, so the blend's coverage
    renormalization runs over a single present pillar and the expected z is the
    raw z. Affine-transformable for the metamorphic case."""
    return {
        f"T{i}": {"quality": v * scale + shift}
        for i, v in enumerate(_PILLAR_VALUES)
    }


def _ic_fixture() -> tuple[list[float], list[float]]:
    signal = [float(i) for i in range(_IC_N)]
    realized = [float(i) * 0.001 for i in range(_IC_N)]
    realized[4], realized[5] = realized[5], realized[4]
    return signal, realized


def _rank_results() -> dict[str, dict]:
    return {f"T{i}": {"final_score": s} for i, s in enumerate(_RANK_SCORES)}


# ════════════════════════════════════════════════════════════════════════════
# The expectations — plain arithmetic from each function's definition
# ════════════════════════════════════════════════════════════════════════════

def _pop_mean_sd(values) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    return mean, math.sqrt(sum((v - mean) ** 2 for v in values) / n)  # ddof=0


def _expected_zscore(value: float) -> float:
    """(value - mean) / POPULATION sd, winsorized to +/-3.0.

    ddof=0 is the load-bearing half: the sample-sd alternative differs by
    sqrt(6/5) = 1.0954 (9.5%) on this fixture.
    """
    mean, sd = _pop_mean_sd(_PILLAR_VALUES)
    return max(-3.0, min(3.0, (value - mean) / sd))


def _expected_attractiveness_raw(value: float) -> float:
    """One present pillar => the coverage-renormalized blend IS the z, then the
    production path rounds it to 4dp."""
    return round(_expected_zscore(value), 4)


def _expected_avg_rank_pct(index: int) -> float:
    """Average-rank percentile: (avg_rank / n) * 100, rounded to 2dp by the
    production path. All six blends are distinct, so avg_rank == ordinal rank."""
    return round((index + 1) / len(_PILLAR_VALUES) * 100, 2)


def _expected_spearman_ic() -> float:
    """rho = 1 - 6*sum(d^2)/(n(n^2-1)); one adjacent transposition => sum(d^2)=2."""
    n = _IC_N
    return 1.0 - 12.0 / (n * (n * n - 1))


def _expected_clustered_se() -> float:
    """sd(ddof=1)/sqrt(n), rounded to 6dp by the production path."""
    vals = _PER_DATE_IC
    n = len(vals)
    mean = sum(vals) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1))  # ddof=1
    return round(sd / math.sqrt(n), 6)


def _expected_clustered_se_population() -> float:
    """The WRONG answer, kept so the convention case can assert it is not this."""
    vals = _PER_DATE_IC
    n = len(vals)
    mean = sum(vals) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / n)  # ddof=0
    return sd / math.sqrt(n)


def _expected_macro_shift() -> float:
    """(modifier - 1.0)/range * max_points = 0.15/0.30*10 = 5.0."""
    return (_SECTOR_MODIFIER - 1.0) / 0.30 * 10.0


def _expected_composite_final() -> float:
    """weighted_base + macro_shift + total_boost, clipped to [0, 100], 1dp.

    base = 80*0.5 + 60*0.5 = 70; macro = 5.0; boost = 3+2 = 5 (under the 10 cap).
    """
    base = _QUANT * 0.5 + _QUAL * 0.5
    total_boost = sum(_BOOSTS.values())
    return round(max(0.0, min(100.0, base + _expected_macro_shift() + total_boost)), 1)


def _expected_percentile(rank: int) -> float:
    """(n - rank) / max(n - 1, 1), rounded to 4dp by the production path."""
    n = len(_RANK_SCORES)
    return round((n - rank) / max(n - 1, 1), 4)


# ════════════════════════════════════════════════════════════════════════════
# Production drivers
# ════════════════════════════════════════════════════════════════════════════

def _attractiveness(scale: float = 1.0, shift: float = 0.0) -> dict:
    from nousergon_lib.quant.attractiveness import compute_cross_sectional_attractiveness

    return compute_cross_sectional_attractiveness(
        _pillar_scores(scale, shift), {"quality": 1.0},
    )


def _composite(**overrides) -> dict:
    from scoring.composite import compute_composite_score

    kwargs = {"quant_score": _QUANT, "qual_score": _QUAL,
              "sector_modifier": _SECTOR_MODIFIER, "boosts": dict(_BOOSTS)}
    kwargs.update(overrides)
    return compute_composite_score(**kwargs)


def _ranked(results: dict[str, dict] | None = None) -> dict[str, dict]:
    from scoring.aggregator import assign_cross_sectional_ranks

    payload = results if results is not None else _rank_results()
    assign_cross_sectional_ranks(payload)
    return payload


def _spearman(signal=None, realized=None):
    from scoring.leaderboard_scoring import spearman_ic

    if signal is None:
        signal, realized = _ic_fixture()
    return spearman_ic(signal, realized)


def _clustered(per_date=None):
    from scoring.leaderboard_scoring import date_clustered_stats

    return date_clustered_stats(_PER_DATE_IC if per_date is None else per_date)


def _zmap(values: dict[str, float]) -> dict[str, float]:
    from scoring.attractiveness_trajectory import _zmap as zmap

    return zmap(values)


def _is_undefined(value: Any) -> float:
    """1.0 iff ``value`` is reported as undefined (``None``/NaN/inf)."""
    if value is None:
        return 1.0
    try:
        return 0.0 if math.isfinite(float(value)) else 1.0
    except (TypeError, ValueError):
        return 1.0


def build_cases() -> list[Case]:
    """The battery. A callable (not a module constant) so no production import
    happens at import time of this module, and so a test can substitute it."""
    from nousergon_lib.quant import attractiveness as attr

    from scoring import composite as comp
    from scoring import leaderboard_scoring as lead

    cases: list[Case] = []

    # ── closed form ─────────────────────────────────────────────────────────
    cases += [
        Case(
            name="attractiveness_zblend_closed_form",
            description=(
                "nousergon_lib.quant.attractiveness.compute_cross_sectional_"
                "attractiveness over six names with ONE pillar: mean = 3.5, "
                "POPULATION sd (ddof=0) = sqrt(35/12), so the top name's raw "
                "blend is (6 - 3.5)/sqrt(35/12), rounded to 4dp. This is the "
                "chokepoint EVERY name's score passes through"
            ),
            inputs={"pillar_values": list(_PILLAR_VALUES), "pillar": "quality",
                    "weights": {"quality": 1.0}, "ddof": 0, "clip": 3.0,
                    "rounding": "4dp", "units": "z-score"},
            expected=_expected_attractiveness_raw(6.0),
            compute=lambda: float(_attractiveness()["T5"]["attractiveness_raw"]),
        ),
        Case(
            name="attractiveness_percentile_closed_form",
            description=(
                "The same call's attractiveness_score: average-rank percentile "
                "(avg_rank/n)*100 rounded to 2dp. Six distinct blends, so the top "
                "name is rank 6 of 6 => 100.0. Pins the 0-100 scale AND the "
                "direction (higher = better)"
            ),
            inputs={"n": len(_PILLAR_VALUES), "rank": 6, "rounding": "2dp",
                    "units": "percentile 0-100"},
            expected=_expected_avg_rank_pct(5),
            compute=lambda: float(_attractiveness()["T5"]["attractiveness_score"]),
        ),
        Case(
            name="attractiveness_winsorization_closed_form",
            description=(
                "A lone outlier in a 20-name cross-section (19 names at 0.0, one "
                "at 100.0) has raw z = 19/sqrt(19) = sqrt(19) = 4.3589, which the "
                "+3.0 winsorization bound must cut to EXACTLY 3.0. Without the "
                "clip this single name would dominate the weighted blend for "
                "every pillar it appears in. NOTE the fixture size is "
                "load-bearing: with a POPULATION sd the largest attainable z in "
                "an n-name cross-section is (n-1)/sqrt(n), which is 2.27 at n=7 "
                "and first crosses 3.0 at n=11 — so the clip is INERT on any "
                "pillar covered by fewer than 11 names, and a small fixture "
                "cannot exercise it at all (see I7272)"
            ),
            inputs={"clip": 3.0, "n_names": _CLIP_FIXTURE_N,
                    "unclipped_z": _unclipped_outlier_z(),
                    "max_attainable_z_formula": "(n-1)/sqrt(n)",
                    "clip_inert_below_n": 11,
                    "source_of_truth": "nousergon_lib.quant.attractiveness._ZSCORE_CLIP",
                    "units": "z-score"},
            expected=3.0,
            compute=_clipped_attractiveness,
        ),
        Case(
            name="composite_macro_shift_closed_form",
            description=(
                "scoring/composite.py::compute_composite_score — "
                "macro_shift = (sector_modifier - 1.0)/MACRO_MODIFIER_RANGE * "
                "MACRO_MAX_SHIFT_POINTS = 0.15/0.30*10 = 5.0 points"
            ),
            inputs={"sector_modifier": _SECTOR_MODIFIER, "macro_modifier_range": 0.30,
                    "macro_max_shift_points": 10.0, "units": "score points"},
            expected=_expected_macro_shift(),
            compute=lambda: float(_composite()["macro_shift"]),
        ),
        Case(
            name="composite_final_score_closed_form",
            description=(
                "Full composite: base = 80*0.5 + 60*0.5 = 70, macro_shift = +5.0, "
                "boosts = 3+2 = 5 (under the 10-point cap) => 80.0, clipped to "
                "[0,100] and rounded to 1dp. Pins the WHOLE assembly, not one term"
            ),
            inputs={"quant_score": _QUANT, "qual_score": _QUAL,
                    "w_quant": 0.50, "w_qual": 0.50,
                    "sector_modifier": _SECTOR_MODIFIER, "boosts": dict(_BOOSTS),
                    "max_aggregate_boost": 10.0, "units": "score 0-100"},
            expected=_expected_composite_final(),
            compute=lambda: float(_composite()["final_score"]),
        ),
        Case(
            name="composite_boost_cap_closed_form",
            description=(
                "Boosts summing to 40 must clamp to max_aggregate_boost = 10.0. "
                "The cap is the only thing standing between an enrichment bug and "
                "an unbounded score lever"
            ),
            inputs={"boosts": {"a": 20.0, "b": 20.0}, "max_aggregate_boost": 10.0,
                    "units": "score points"},
            expected=10.0,
            compute=lambda: float(_composite(boosts={"a": 20.0, "b": 20.0})["total_boost"]),
        ),
        Case(
            name="cross_sectional_rank_closed_form",
            description=(
                "scoring/aggregator.py::assign_cross_sectional_ranks — MIN-rank "
                "tie semantics: scores [80, 75, 75, 60] must rank [1, 2, 2, 4], "
                "NOT [1, 2, 2, 3] (dense) and NOT [1, 2, 3, 4] (ordinal). Sum of "
                "|rank - expected| = 0. The tie is the only input that "
                "distinguishes the three conventions"
            ),
            inputs={"scores": list(_RANK_SCORES), "expected_ranks": [1, 2, 2, 4],
                    "tie_semantics": "min", "units": "rank error sum"},
            expected=0.0,
            compute=_rank_error_sum,
        ),
        Case(
            name="cross_sectional_percentile_closed_form",
            description=(
                "Same call: percentile = (n - rank)/max(n-1, 1). The two TIED "
                "names at rank 2 must both get (4-2)/3 = 0.6666..., so a tie "
                "cannot be silently broken by the percentile step either"
            ),
            inputs={"n": len(_RANK_SCORES), "rank": 2, "rounding": "4dp",
                    "units": "percentile 0-1"},
            expected=_expected_percentile(2),
            compute=lambda: float(_ranked()["T1"]["percentile"]),
        ),
        Case(
            name="spearman_ic_closed_form",
            description=(
                "scoring/leaderboard_scoring.py::spearman_ic over 12 names with "
                "all-distinct ranks and ONE adjacent transposition: "
                "rho = 1 - 6*sum(d^2)/(n(n^2-1)) = 1 - 12/1716. Derived from "
                "Spearman's definition, not from any library"
            ),
            inputs={"n": _IC_N, "sum_d_squared": 2.0, "ties": "none",
                    "units": "rank correlation [-1,1]"},
            expected=_expected_spearman_ic(),
            compute=lambda: float(_spearman()),
        ),
        Case(
            name="date_clustered_se_closed_form",
            description=(
                "scoring/leaderboard_scoring.py::date_clustered_stats — "
                "se = sd(ddof=1)/sqrt(n) over 5 per-date ICs, rounded to 6dp. "
                "SAMPLE variance is the load-bearing half: the population "
                "alternative differs by sqrt(5/4) = 1.118 and would overstate "
                "every t-stat the cutover gate reads"
            ),
            inputs={"per_date_ic": list(_PER_DATE_IC), "n": len(_PER_DATE_IC),
                    "ddof": 1, "rounding": "6dp", "units": "standard error"},
            expected=_expected_clustered_se(),
            compute=lambda: float(_clustered()["se"]),
        ),
        Case(
            name="rankdata_average_ties_closed_form",
            description=(
                "scoring/leaderboard_scoring.py::_rankdata uses AVERAGE-rank ties "
                "(matching scipy rankdata method='average'), NOT the min-rank the "
                "aggregator uses. Values [10, 20, 20, 30] must rank "
                "[1, 2.5, 2.5, 4]. Two DIFFERENT tie conventions coexist in this "
                "repo by design — average for correlation, min for selection — and "
                "this case pins the correlation one so they cannot silently merge"
            ),
            inputs={"values": [10.0, 20.0, 20.0, 30.0],
                    "expected_ranks": [1.0, 2.5, 2.5, 4.0],
                    "tie_semantics": "average", "units": "rank error sum"},
            expected=0.0,
            compute=lambda: float(sum(
                abs(a - b) for a, b in
                zip(lead._rankdata([10.0, 20.0, 20.0, 30.0]), [1.0, 2.5, 2.5, 4.0],
                    strict=True)
            )),
        ),
    ]

    # ── metamorphic ─────────────────────────────────────────────────────────
    cases += [
        Case(
            name="attractiveness_affine_invariance_metamorphic",
            description=(
                "METAMORPHIC. A cross-sectional z-blend is invariant under a "
                "positive affine transform of EVERY pillar value: scoring "
                "(7x - 13) must equal scoring x. Catches a lost mean-centering or "
                "a units change without needing to know the right answer — the "
                "avg_volume_20d class of bug. Difference expected 0.0"
            ),
            inputs={"scale": _AFFINE_SCALE, "shift": _AFFINE_SHIFT,
                    "units": "z-score (difference)"},
            expected=0.0,
            compute=lambda: (
                float(_attractiveness(_AFFINE_SCALE, _AFFINE_SHIFT)["T5"]["attractiveness_raw"])
                - float(_attractiveness()["T5"]["attractiveness_raw"])
            ),
        ),
        Case(
            name="attractiveness_monotonicity_metamorphic",
            description=(
                "METAMORPHIC. The blend must be NON-DECREASING in its driving "
                "pillar: sorting the six names by pillar value must reproduce the "
                "order of their attractiveness scores exactly. Count of inversions "
                "expected 0"
            ),
            inputs={"pillar_values": list(_PILLAR_VALUES),
                    "relation": "v1 < v2 => score(v1) <= score(v2)",
                    "units": "inversion count"},
            expected=0.0,
            compute=_attractiveness_inversions,
            tolerance=0.0,
        ),
        Case(
            name="rank_permutation_invariance_metamorphic",
            description=(
                "METAMORPHIC. cross_sectional_rank must depend only on "
                "final_score, never on dict insertion order. The same four scores "
                "inserted in reverse must produce the same per-ticker ranks — "
                "including for the TIED pair, which is where an order-dependent "
                "implementation would show. Sum of |difference| expected 0.0"
            ),
            inputs={"scores": list(_RANK_SCORES), "permutation": "reverse insertion",
                    "units": "rank difference sum"},
            expected=0.0,
            compute=_rank_permutation_delta,
            tolerance=0.0,
        ),
        Case(
            name="spearman_ic_sign_flip_metamorphic",
            description=(
                "METAMORPHIC. Negating the signal reverses every rank, so the IC "
                "must flip sign exactly. Catches a rank direction inversion — the "
                "error that turns a real edge into a real anti-edge while the "
                "magnitude still looks right. Sum expected 0.0"
            ),
            inputs={"n": _IC_N, "relation": "IC(-s, r) + IC(s, r) == 0",
                    "units": "rank correlation (sum)"},
            expected=0.0,
            compute=_ic_sign_flip_sum,
        ),
        Case(
            name="spearman_ic_monotone_transform_invariance_metamorphic",
            description=(
                "METAMORPHIC. Spearman is a RANK correlation, so any strictly "
                "increasing transform of the signal (here exp) must leave it "
                "unchanged. An implementation that quietly became Pearson would "
                "pass the closed-form case on the near-linear fixture and fail "
                "this one. Difference expected 0.0"
            ),
            inputs={"n": _IC_N, "transform": "exp (strictly increasing)",
                    "units": "rank correlation (difference)"},
            expected=0.0,
            compute=_ic_monotone_transform_delta,
        ),
        Case(
            name="composite_monotonicity_metamorphic",
            description=(
                "METAMORPHIC. final_score must be NON-DECREASING in quant_score, "
                "all else equal, across 21 points spanning [0, 100]. Count of "
                "violations expected 0"
            ),
            inputs={"quant_range": [0.0, 100.0], "n_points": 21,
                    "relation": "q1 < q2 => final(q1) <= final(q2)",
                    "units": "violation count"},
            expected=0.0,
            compute=_composite_monotonicity_violations,
            tolerance=0.0,
        ),
    ]

    # ── degenerate inputs (the I7237 class) ─────────────────────────────────
    cases += [
        Case(
            name="spearman_ic_zero_variance_is_undefined_degenerate",
            description=(
                "A constant signal has UNDEFINED rank correlation (correlation of "
                "a constant with anything). _pearson returns None. 1.0 iff "
                "undefined. This is the CORRECT convention and is the reference "
                "the three known_gap cases below are measured against"
            ),
            inputs={"signal": "constant", "n": _IC_N,
                    "units": "boolean-encoded contract"},
            expected=1.0,
            compute=lambda: _is_undefined(_spearman([1.0] * _IC_N, _ic_fixture()[1])),
            tolerance=0.0,
        ),
        Case(
            name="spearman_ic_single_observation_is_undefined_degenerate",
            description=(
                "A single paired name cannot have a rank correlation. Must report "
                "None, never 0.0. 1.0 iff undefined"
            ),
            inputs={"n": 1, "units": "boolean-encoded contract"},
            expected=1.0,
            compute=lambda: _is_undefined(_spearman([1.0], [2.0])),
            tolerance=0.0,
        ),
        Case(
            name="date_clustered_single_date_se_is_undefined_degenerate",
            description=(
                "One date gives no dispersion, so the clustered SE and t-stat are "
                "UNDEFINED. Must report None for both — a t-stat of 0.0 or inf "
                "here would be read by the cutover gate as a real measurement. "
                "1.0 iff se is undefined"
            ),
            inputs={"per_date_ic": [0.04], "n": 1,
                    "units": "boolean-encoded contract"},
            expected=1.0,
            compute=lambda: _is_undefined(_clustered([0.04])["se"]),
            tolerance=0.0,
        ),
        Case(
            name="date_clustered_empty_is_undefined_degenerate",
            description=(
                "No dates at all => the whole stat block is None, not a zero-filled "
                "record. 1.0 iff the function returns None"
            ),
            inputs={"per_date_ic": [], "n": 0,
                    "units": "boolean-encoded contract"},
            expected=1.0,
            compute=lambda: 1.0 if _clustered([]) is None else 0.0,
            tolerance=0.0,
        ),
        Case(
            name="composite_both_scores_missing_is_flagged_degenerate",
            description=(
                "Both quant and qual missing => final_score None AND "
                "score_failed True. Pins that the failure is reported as a FLAG, "
                "not encoded as a low score — a 0.0 here would rank the name last "
                "instead of excluding it. 1.0 iff both hold"
            ),
            inputs={"quant_score": None, "qual_score": None,
                    "units": "boolean-encoded contract"},
            expected=1.0,
            compute=_composite_missing_is_flagged,
            tolerance=0.0,
        ),
        # ── the three known gaps: pinned at MEASURED, not corrected ─────────
        Case(
            name="attractiveness_zero_variance_degenerate",
            description=(
                "KNOWN GAP (alpha-engine-config-I7272), PINNED NOT FIXED. "
                "nousergon_lib.quant.attractiveness._zscore returns a finite 0.0 "
                "when a pillar has ZERO cross-sectional spread, where the z-score "
                "is UNDEFINED. 0.0 is also exactly what a genuinely at-the-mean "
                "name produces, so 'this pillar had no spread' and 'this name is "
                "average' are indistinguishable to every consumer — and the blend "
                "treats a measured NOTHING as a measured AVERAGE. This is the "
                "I7237 class at the signal layer, at the chokepoint every name "
                "passes through. Expected 0.0 records the MEASURED behaviour so "
                "further drift goes red; a PASS means 'unchanged', NOT 'correct'"
            ),
            inputs={"pillar_values": [4.0] * 6, "correct_behaviour": "None",
                    "measured_behaviour": 0.0, "units": "z-score"},
            expected=0.0,
            compute=_zero_variance_attractiveness,
            known_gap=True,
            gap_issue="alpha-engine-config-I7272",
        ),
        Case(
            name="trajectory_zmap_zero_variance_degenerate",
            description=(
                "FIXED (alpha-engine-config-I7272). "
                "scoring/attractiveness_trajectory.py::_zmap on a ZERO-VARIANCE "
                "cross-section now reports UNDEFINED (None) for every ticker "
                "rather than a finite 0.0 — 0.0 is also exactly what a genuinely "
                "at-the-mean ticker produces, so the two were indistinguishable "
                "downstream. 1.0 iff undefined."
            ),
            inputs={"values": {"A": 4.0, "B": 4.0, "C": 4.0}, "units": "z-score"},
            expected=1.0,
            compute=lambda: _is_undefined(_zmap({"A": 4.0, "B": 4.0, "C": 4.0})["A"]),
            tolerance=0.0,
        ),
        Case(
            name="trajectory_zmap_single_observation_degenerate",
            description=(
                "FIXED (alpha-engine-config-I7272). "
                "A single observation has no cross-section at all — _zmap now "
                "reports UNDEFINED (None) rather than the previous fabricated "
                "0.0. 1.0 iff undefined."
            ),
            inputs={"values": {"A": 4.0}, "n": 1, "units": "z-score"},
            expected=1.0,
            compute=lambda: _is_undefined(_zmap({"A": 4.0})["A"]),
            tolerance=0.0,
        ),
    ]

    # ── convention pinning ──────────────────────────────────────────────────
    cases += [
        Case(
            name="attractiveness_ddof_convention",
            description=(
                "CONVENTION. nousergon_lib.quant.attractiveness._mean_std uses "
                "POPULATION std (ddof=0). Source of truth: that module's "
                "`var = sum((v - mean) ** 2) / n`. Asserted as the DIFFERENCE from "
                "the sample-sd answer, which must be NON-zero — so this case fails "
                "if the convention ever silently flips. NOTE: "
                "scoring/leaderboard_scoring.py uses ddof=1 on its own series; two "
                "std conventions coexist by design (cross-sectional vs "
                "time-series) and neither is declared anywhere else"
            ),
            inputs={"ddof_in_use": 0, "alternative_ddof": 1,
                    "source_of_truth": "nousergon_lib.quant.attractiveness._mean_std",
                    "units": "z-score (difference)"},
            # Both sides carry the production path's 4dp rounding, so the
            # difference is exact and the 1e-9 band applies unchanged. An
            # expectation left unrounded here would need a 1e-4 tolerance —
            # 100,000x looser than the rest of the battery, and wide enough to
            # hide a real convention change.
            expected=round(_expected_zscore(6.0), 4) - round(_sample_zscore(6.0), 4),
            compute=lambda: (float(_attractiveness()["T5"]["attractiveness_raw"])
                             - round(_sample_zscore(6.0), 4)),
        ),
        Case(
            name="date_clustered_ddof_convention",
            description=(
                "CONVENTION. scoring/leaderboard_scoring.py::date_clustered_stats "
                "uses SAMPLE variance (ddof=1) — `var = sum(...)/(n-1)`. The "
                "population alternative differs by sqrt(5/4) = 1.118 on this "
                "fixture and would overstate every t-stat the cutover gate reads. "
                "Asserted against the hand-computed ddof=1 SE"
            ),
            inputs={"per_date_ic": list(_PER_DATE_IC), "ddof_in_use": 1,
                    "population_alternative": _expected_clustered_se_population(),
                    "source_of_truth": "scoring/leaderboard_scoring.py::date_clustered_stats",
                    "units": "standard error"},
            expected=_expected_clustered_se(),
            compute=lambda: float(_clustered()["se"]),
        ),
        Case(
            name="zscore_clip_convention",
            description=(
                "CONVENTION. _ZSCORE_CLIP = 3.0 — the winsorization bound on every "
                "pillar z-score. Widening it lets a single outlier dominate the "
                "blend; narrowing it compresses the whole cross-section. Read from "
                "the library, not restated"
            ),
            inputs={"constant": "nousergon_lib.quant.attractiveness._ZSCORE_CLIP",
                    "units": "z-score bound"},
            expected=3.0,
            compute=lambda: float(attr._ZSCORE_CLIP),
            tolerance=0.0,
        ),
        Case(
            name="horizon_days_convention",
            description=(
                "CONVENTION. DEFAULT_HORIZON_DAYS = 21 — the forward window every "
                "IC in this module is measured over. An IC computed over a "
                "different horizon than the one it is compared against is the "
                "quiet version of the sqrt(365)-vs-sqrt(252) defect"
            ),
            inputs={"constant": "scoring.leaderboard_scoring.DEFAULT_HORIZON_DAYS",
                    "units": "trading days"},
            expected=21.0,
            compute=lambda: float(lead.DEFAULT_HORIZON_DAYS),
            tolerance=0.0,
        ),
        Case(
            name="composite_weights_convention",
            description=(
                "CONVENTION. DEFAULT_W_QUANT = DEFAULT_W_QUAL = 0.50, and they "
                "must sum to 1.0. A pair that no longer sums to 1 still produces "
                "plausible scores on a different scale — silently rebased"
            ),
            inputs={"w_quant": 0.50, "w_qual": 0.50,
                    "source_of_truth": "scoring/composite.py", "units": "weight sum"},
            expected=1.0,
            compute=lambda: float(comp.DEFAULT_W_QUANT + comp.DEFAULT_W_QUAL),
        ),
        Case(
            name="macro_overlay_constants_convention",
            description=(
                "CONVENTION. MACRO_MODIFIER_RANGE = 0.30 and "
                "MACRO_MAX_SHIFT_POINTS = 10.0 together set the macro overlay's "
                "dose. Encoded as range*100 + max_points = 30 + 10 = 40 so a "
                "change to EITHER moves this number"
            ),
            inputs={"macro_modifier_range": 0.30, "macro_max_shift_points": 10.0,
                    "encoding": "100*range + max_points", "units": "composite constant"},
            expected=40.0,
            compute=lambda: float(100.0 * comp.MACRO_MODIFIER_RANGE
                                  + comp.MACRO_MAX_SHIFT_POINTS),
        ),
        Case(
            name="pillar_weights_sum_convention",
            description=(
                "CONVENTION. normalize_pillar_weights must always return weights "
                "summing to 1.0 over the six pillars, including on the "
                "empty/negative fallback path (equal weights, 1/6 each). Weights "
                "that do not sum to 1 are renormalized per-name at runtime, so a "
                "broken set would still produce plausible scores"
            ),
            inputs={"raw": None, "pillars": 6,
                    "source_of_truth": "nousergon_lib.quant.attractiveness."
                                       "normalize_pillar_weights",
                    "units": "weight sum"},
            expected=1.0,
            compute=lambda: float(sum(attr.normalize_pillar_weights(None).values())),
        ),
        Case(
            name="undefined_representation_divergence_convention",
            description=(
                "CONVENTION / CROSS-IMPLEMENTATION (alpha-engine-config-I7272). "
                "Counts how many of the three degenerate-input sites report an "
                "undefined value as UNDEFINED rather than as a measured-looking "
                "0.0. Measured 2026-08-13: 1 of 3 (leaderboard_scoring._pearson "
                "-> None; attractiveness._zscore -> 0.0; "
                "attractiveness_trajectory._zmap -> 0.0). "
                "config-I7272 (this change): attractiveness_trajectory._zmap "
                "FIXED in crucible-research -> None, moving the count to 2 of 3. "
                "attractiveness._zscore lives in nousergon_lib — OUT OF SCOPE for "
                "this change (a different repo/PR); it remains PINNED at its "
                "MEASURED 0.0. Expected 2.0 pins the new measured state. The day "
                "nousergon_lib is aligned too this case must be updated to 3.0 IN "
                "THE SAME CHANGE — which is the point: the fix cannot land "
                "silently"
            ),
            inputs={"sites": ["leaderboard_scoring._pearson",
                              "attractiveness._zscore",
                              "attractiveness_trajectory._zmap"],
                    "measured_safe_count": 2, "total_sites": 3,
                    "units": "count of sites reporting undefined honestly"},
            expected=2.0,
            compute=_undefined_representation_count,
            tolerance=0.0,
            known_gap=True,
            gap_issue="alpha-engine-config-I7272",
        ),
    ]

    return cases


# ── compute helpers used by the battery ─────────────────────────────────────

def _sample_zscore(value: float) -> float:
    """The ddof=1 z-score — the WRONG answer, for the convention case."""
    n = len(_PILLAR_VALUES)
    mean = sum(_PILLAR_VALUES) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in _PILLAR_VALUES) / (n - 1))
    return max(-3.0, min(3.0, (value - mean) / sd))


#: Cross-section size for the winsorization case. NOT arbitrary: with a
#: POPULATION standard deviation, the largest z any single name can attain in an
#: n-name cross-section is (n-1)/sqrt(n) — because the outlier inflates the very
#: sd it is measured against. That bound is 2.27 at n=7 and only crosses 3.0 at
#: n=11, so a small-cross-section fixture can never exercise the clip at all
#: (measured: a 7-name fixture tops out at z=2.368 and the +3.0 bound is
#: unreachable). 20 names with 19 at zero puts the outlier at z=sqrt(19)=4.359,
#: comfortably past the bound. The same arithmetic means the winsorization is
#: INERT on any pillar covered by fewer than 11 names — noted in I7272.
_CLIP_FIXTURE_N = 20


def _clipped_attractiveness() -> float:
    """A lone outlier in a 20-name cross-section must clip to exactly +3.0.

    19 names at 0.0 and one at 100.0: mean = 5.0, population sd =
    100*sqrt(19)/20, so the outlier's raw z is 19/sqrt(19) = sqrt(19) = 4.3589,
    which the +3.0 winsorization bound must cut to exactly 3.0.
    """
    from nousergon_lib.quant.attractiveness import compute_cross_sectional_attractiveness

    scores: dict[str, dict] = {
        f"F{i}": {"quality": 0.0} for i in range(_CLIP_FIXTURE_N - 1)
    }
    scores["OUTLIER"] = {"quality": 100.0}
    out = compute_cross_sectional_attractiveness(scores, {"quality": 1.0})
    return float(out["OUTLIER"]["attractiveness_raw"])


def _unclipped_outlier_z() -> float:
    """sqrt(19) — what the outlier's z would be WITHOUT the winsorization bound.
    Kept so the case's description is checkable rather than asserted."""
    return math.sqrt(_CLIP_FIXTURE_N - 1)


def _rank_error_sum() -> float:
    ranked = _ranked()
    observed = [ranked[f"T{i}"]["cross_sectional_rank"] for i in range(len(_RANK_SCORES))]
    return float(sum(abs(a - b) for a, b in zip(observed, [1, 2, 2, 4], strict=True)))


def _rank_permutation_delta() -> float:
    forward = _ranked()
    reverse_input = {f"T{i}": {"final_score": s}
                     for i, s in reversed(list(enumerate(_RANK_SCORES)))}
    reverse = _ranked(reverse_input)
    return float(sum(
        abs(forward[f"T{i}"]["cross_sectional_rank"]
            - reverse[f"T{i}"]["cross_sectional_rank"])
        for i in range(len(_RANK_SCORES))
    ))


def _attractiveness_inversions() -> float:
    out = _attractiveness()
    scores = [float(out[f"T{i}"]["attractiveness_score"]) for i in range(len(_PILLAR_VALUES))]
    return float(sum(1 for a, b in zip(scores, scores[1:], strict=False)
                     if b < a - 1e-12))


def _ic_sign_flip_sum() -> float:
    signal, realized = _ic_fixture()
    return float(_spearman(signal, realized)) + float(
        _spearman([-s for s in signal], realized))


def _ic_monotone_transform_delta() -> float:
    signal, realized = _ic_fixture()
    transformed = [math.exp(s / 4.0) for s in signal]
    return float(_spearman(transformed, realized)) - float(_spearman(signal, realized))


def _composite_monotonicity_violations() -> float:
    finals = [float(_composite(quant_score=q * 5.0)["final_score"]) for q in range(21)]
    return float(sum(1 for a, b in zip(finals, finals[1:], strict=False)
                     if b < a - 1e-12))


def _composite_missing_is_flagged() -> float:
    out = _composite(quant_score=None, qual_score=None)
    return 1.0 if (out["final_score"] is None and out["score_failed"] is True) else 0.0


def _zero_variance_attractiveness() -> float:
    from nousergon_lib.quant.attractiveness import compute_cross_sectional_attractiveness

    flat = {f"T{i}": {"quality": 4.0} for i in range(6)}
    out = compute_cross_sectional_attractiveness(flat, {"quality": 1.0})
    return float(out["T0"]["attractiveness_raw"])


def _undefined_representation_count() -> float:
    """How many of the three degenerate sites report UNDEFINED honestly."""
    safe = 0
    # 1. leaderboard_scoring._pearson via spearman_ic on a constant signal.
    if _is_undefined(_spearman([1.0] * _IC_N, _ic_fixture()[1])) == 1.0:
        safe += 1
    # 2. nousergon_lib.quant.attractiveness._zscore on a zero-spread pillar.
    if _is_undefined(_zero_variance_attractiveness()) == 1.0:
        safe += 1
    # 3. attractiveness_trajectory._zmap on a zero-spread cross-section.
    if _is_undefined(_zmap({"A": 4.0, "B": 4.0, "C": 4.0})["A"]) == 1.0:
        safe += 1
    return float(safe)


# ════════════════════════════════════════════════════════════════════════════
# Provenance header — the reason this is an INSTRUMENT check, not a code check
# ════════════════════════════════════════════════════════════════════════════

def resolved_library_versions(
    distributions: tuple[str, ...] = _TRACKED_DISTRIBUTIONS,
) -> dict[str, str]:
    """The installed version of every numeric distribution loaded at runtime.

    ``importlib.metadata.version`` reads the DISTRIBUTION metadata pip actually
    resolved into this image — the thing that moves between the CI runner and the
    deployed Lambda. A missing distribution is recorded explicitly, never
    omitted: an absent key and a missing library must not look the same.
    """
    from importlib.metadata import PackageNotFoundError, version

    resolved: dict[str, str] = {}
    for dist in distributions:
        try:
            resolved[dist] = version(dist)
        except PackageNotFoundError:
            resolved[dist] = "<not installed>"
        except Exception as exc:  # noqa: BLE001 — a version probe never blocks
            resolved[dist] = f"<unavailable: {type(exc).__name__}>"
    return resolved


def code_sha() -> str:
    """The SHA of the code that ran, without shelling out.

    Deploy-time stamps first (``GIT_SHA`` env, then ``/var/task/GIT_SHA.txt`` —
    the fleet's Lambda-image convention), then the checkout's own ``.git`` refs
    for a local run. ``unknown`` is a legitimate answer and is recorded as one — a
    fabricated SHA is worse than an absent one.
    """
    for env_key in ("GIT_SHA", "CODE_SHA", "GITHUB_SHA"):
        stamped = os.environ.get(env_key)
        if stamped:
            return stamped.strip()
    try:
        lambda_stamp = Path("/var/task/GIT_SHA.txt")
        if lambda_stamp.is_file():
            stamped = lambda_stamp.read_text().strip()
            if stamped:
                return stamped
    except Exception:  # noqa: BLE001,S110 — provenance never blocks the battery
        # Deliberately silent: this is the OPTIONAL Lambda stamp probe, and its
        # absence is the normal case off-Lambda. The failure mode swallowed is
        # "no deploy-time SHA stamp here"; the recording surface is the returned
        # `code_sha`, which falls through to the .git lookup and finally to the
        # literal "unknown" — a value the artifact carries honestly.
        pass
    try:
        git_dir = Path(__file__).resolve().parents[1] / ".git"
        head = (git_dir / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:].strip()
        ref_path = git_dir / ref
        if ref_path.is_file():
            return ref_path.read_text().strip()
        for line in (git_dir / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split(" ", 1)[0].strip()
        return "unknown"
    except Exception:  # noqa: BLE001 — provenance never blocks the battery
        return "unknown"


# ════════════════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════════════════

def _call_with_timeout(fn: Callable[[], float], seconds: float) -> float:
    """Run ``fn`` under a wall-clock budget.

    A SIGALRM budget is installed when this is the main thread of a platform that
    has one (a Lambda handler thread qualifies). Where it is not available the
    elapsed time is checked after the call instead — that cannot interrupt a hang,
    but it does catch an overrun, and the caller distinguishes neither: both FAIL.
    """
    can_interrupt = (
        hasattr(signal, "SIGALRM")
        and hasattr(signal, "setitimer")
        and threading.current_thread() is threading.main_thread()
    )
    if not can_interrupt:
        started = time.monotonic()
        value = fn()
        elapsed = time.monotonic() - started
        if elapsed > seconds:
            raise _CaseTimeout(
                f"case exceeded its {seconds:g}s budget "
                f"({elapsed:.1f}s, detected after the fact)"
            )
        return value

    def _fire(_signum, _frame):
        raise _CaseTimeout(f"case exceeded its {seconds:g}s budget")

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def run_self_test(
    run_date: str | None = None,
    *,
    case_provider: Callable[[], list[Case]] | None = None,
    component: str = COMPONENT,
    schema: str = SCHEMA,
    case_timeout_seconds: float = CASE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run the known-answer battery on the DEPLOYED instrument and return the body.

    Never raises — see the module docstring's CONTRACT section.
    """
    started = time.monotonic()
    header = {
        "schema": schema,
        "component": component,
        "run_date": run_date,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "code_sha": code_sha(),
        "libraries": resolved_library_versions(),
        "case_timeout_seconds": case_timeout_seconds,
    }

    provider = case_provider or build_cases
    try:
        # Materialised inside the try: a provider returning a lazy or broken
        # iterable would otherwise raise at the FOR loop below, outside every
        # handler, and take down the stage this must never be able to fail.
        cases = list(provider())
    except Exception as exc:  # noqa: BLE001 — see CONTRACT: this becomes UNKNOWN
        logger.error(
            "self-test: the battery could not be constructed (%s: %s) — verdict "
            "UNKNOWN. No correctness guarantee is granted this cycle.",
            type(exc).__name__, exc, exc_info=True,
        )
        return {**header, "status": "error", "verdict": UNKNOWN, "cases": [],
                "n_cases": 0, "n_failed": 0, "n_errored": 0, "n_known_gaps": 0,
                "error_class": type(exc).__name__, "error_msg": str(exc)[:500],
                "wall_clock_seconds": round(time.monotonic() - started, 3)}

    records: list[dict] = []
    for index, case in enumerate(cases):
        try:
            record: dict[str, Any] = {
                "case": case.name,
                "description": case.description,
                "inputs": case.inputs,
                "expected": case.expected,
                "actual": None,
                "abs_error": None,
                "tolerance": case.tolerance,
                "verdict": UNKNOWN,
            }
        except Exception as exc:  # noqa: BLE001 — see CONTRACT: never raises
            # A provider that returned a well-formed LIST of malformed items gets
            # past the materialisation guard above and would otherwise raise here,
            # outside every handler — taking down the stage this module must never
            # be able to fail. Recorded as UNKNOWN, never as a pass.
            logger.error(
                "self-test: case %d is not a Case (%s: %s) => UNKNOWN",
                index, type(exc).__name__, exc,
            )
            records.append({
                "case": f"<malformed case {index}>",
                "description": "the case provider returned a non-Case object",
                "inputs": {"units": "n/a"},
                "expected": None, "actual": None, "abs_error": None,
                "tolerance": None, "verdict": UNKNOWN, "errored": True,
                "error_class": type(exc).__name__, "error_msg": str(exc)[:500],
                "wall_clock_seconds": 0.0,
            })
            continue

        if case.known_gap:
            # Stated in the artifact, in words, so a reader never reads this
            # row's PASS as an endorsement of the behaviour it pins.
            record["known_gap"] = True
            record["gap_issue"] = case.gap_issue
            record["known_gap_note"] = (
                "This case PINS behaviour believed incorrect at its MEASURED "
                "value so further drift goes red. PASS means 'unchanged since "
                f"recorded', NOT 'this is correct'. Tracked in {case.gap_issue}."
            )
        case_started = time.monotonic()
        try:
            actual = float(_call_with_timeout(case.compute, case_timeout_seconds))
            error = abs(actual - case.expected)
            record["actual"] = actual
            record["abs_error"] = error
            record["verdict"] = PASS if error <= case.tolerance else FAIL
            if record["verdict"] == FAIL:
                logger.error(
                    "self-test case FAILED: %s expected=%r actual=%r abs_error=%r "
                    "tolerance=%r", case.name, case.expected, actual, error,
                    case.tolerance,
                )
        except _CaseTimeout as exc:
            # Brian ruling 2026-08-13: a timeout is FAIL, never UNKNOWN.
            record["verdict"] = FAIL
            record["timed_out"] = True
            record["error_class"] = type(exc).__name__
            record["error_msg"] = str(exc)[:500]
            logger.error("self-test case TIMED OUT (=> FAIL): %s (%s)", case.name, exc)
        except Exception as exc:  # noqa: BLE001 — a case that could not RUN is UNKNOWN
            record["verdict"] = UNKNOWN
            record["errored"] = True
            record["error_class"] = type(exc).__name__
            record["error_msg"] = str(exc)[:500]
            logger.error(
                "self-test case ERRORED (=> UNKNOWN): %s (%s: %s)",
                case.name, type(exc).__name__, exc, exc_info=True,
            )
        record["wall_clock_seconds"] = round(time.monotonic() - case_started, 3)
        records.append(record)

    n_failed = sum(1 for r in records if r["verdict"] == FAIL)
    n_errored = sum(1 for r in records if r["verdict"] == UNKNOWN)
    n_known_gaps = sum(1 for r in records if r.get("known_gap"))
    if n_failed:
        verdict = FAIL
    elif n_errored or not records:
        verdict = UNKNOWN
    else:
        verdict = PASS

    body = {**header, "status": "ok", "verdict": verdict, "cases": records,
            "n_cases": len(records), "n_failed": n_failed, "n_errored": n_errored,
            "n_known_gaps": n_known_gaps,
            "wall_clock_seconds": round(time.monotonic() - started, 3)}

    if verdict == PASS:
        logger.info(
            "self-test PASS — %d/%d known-answer cases agreed on %s (%s); "
            "%d of them PIN a known gap rather than endorse it",
            len(records), len(records), header["python"],
            ", ".join(f"{k} {v}" for k, v in header["libraries"].items()),
            n_known_gaps,
        )
    elif verdict == UNKNOWN:
        logger.error(
            "self-test UNKNOWN — %d/%d cases could not run. The correctness "
            "guarantee is WITHHELD this cycle (never granted by default).",
            n_errored, len(records),
        )
    else:
        logger.error(
            "self-test FAIL — %d/%d known-answer cases DISAGREE with their "
            "hand-derived expectation on the DEPLOYED libraries (%s). THIS "
            "CYCLE'S SIGNAL SET IS NOT TRUSTWORTHY.",
            n_failed, len(records),
            ", ".join(f"{k} {v}" for k, v in header["libraries"].items()),
        )
    return body


def verdict_is_pass(verdict: str | None) -> bool:
    """True only for an explicit PASS — ``None`` and ``"ok"`` withhold the guarantee."""
    return verdict == PASS


def self_test_key(run_date: str) -> str:
    """S3 key of the research module's published self-test for ``run_date``."""
    return _KEY_TEMPLATE.format(run_date=run_date)


def write_self_test(bucket: str, run_date: str, body: dict, s3_client=None) -> str:
    """Persist the artifact. Returns the key written.

    Raises on failure: this artifact IS the evidence the stage graded its own
    arithmetic, so a silent write failure would reproduce exactly the absence the
    self-test exists to remove. The caller isolates it so the signal set — the
    primary deliverable — survives regardless.
    """
    import json

    import boto3

    client = s3_client or boto3.client("s3")
    key = self_test_key(run_date)
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(body, indent=2, default=str).encode(),
        ContentType="application/json",
    )
    return key


# ════════════════════════════════════════════════════════════════════════════
# Console surface — `principles.md` §2.7: a check that reports nowhere is unobserved
# ════════════════════════════════════════════════════════════════════════════

#: The fleet check-result envelope contract, as published by
#: ``nousergon_lib.fleet_check_result`` and read by the console's
#: ``checks-envelope`` adapter
#: (``nous-ergon-ops/nousergon-console/config.d/fleet-checks.yaml``).
#:
#: WHY THIS IS BUILT HERE RATHER THAN IMPORTED FROM THE LIB
#: --------------------------------------------------------
#: The lib is the right home and ``fleet_check_result`` is already there — but it
#: first ships in **nousergon-lib v0.124.29**, and this repo pins **v0.124.3**
#: (``requirements.txt``). Bumping the pin to reach it would pull 26+ lib versions
#: into the research Lambda inside a PR whose entire purpose is to VERIFY that
#: image's arithmetic — and a lib bump moves ``quant.attractiveness``, which is
#: the chokepoint this battery measures. That makes the bump self-defeating here,
#: not merely large.
#:
#: So this is the same interim duplication the lib's own module docstring records
#: for ``alpha-engine-config`` ("still carries its own copy for now … staged
#: deliberately rather than silently, per ``principles.md`` §2.4, and the interim
#: duplication is held legitimate by a contract test asserting the two modules
#: still agree on the envelope they produce" — ``shared-code-policy.md`` §5.1's
#: fork-detection backstop). ``tests/test_self_test.py`` carries that contract
#: test; migration to the lib import is tracked in **alpha-engine-config-I7274**.
_ENVELOPE_SCHEMA_VERSION = 1

STATUS_OK = "ok"
STATUS_ATTENTION = "attention"
STATUS_ERROR = "error"

RESEARCH_BUCKET = "alpha-engine-research"
CHECKS_PREFIX = "ops/checks/"


def console_envelope_key(check_id: str = CHECK_ID) -> str:
    return f"{CHECKS_PREFIX}{check_id}/latest.json"


def console_envelope(body: dict, now=None) -> dict:
    """The fleet check-result envelope for this self-test.

    The status mapping is deliberately NOT an identity of the internal verdict —
    the envelope vocabulary is a published contract with the console adapter:

    * ``PASS``    -> ``ok``
    * ``FAIL``    -> ``error``    (evidence the numbers are wrong)
    * ``UNKNOWN`` -> ``attention`` (could not measure — `principles.md` §2.7
      forbids rendering this green, and it is not a defect either)

    ``ran_at`` + ``cadence_minutes`` are what let the console mark this check
    STALE when it stops publishing, whatever status it last wrote — the last
    thing a dying check writes is almost always "ok".
    """
    from datetime import datetime

    verdict = body.get("verdict")
    status = {PASS: STATUS_OK, FAIL: STATUS_ERROR}.get(verdict, STATUS_ATTENTION)

    n_cases = body.get("n_cases", 0)
    n_failed = body.get("n_failed", 0)
    n_errored = body.get("n_errored", 0)
    n_gaps = body.get("n_known_gaps", 0)

    findings = [
        {"key": c["case"], "detail": (
            f"expected={c.get('expected')!r} actual={c.get('actual')!r} "
            f"abs_error={c.get('abs_error')!r} tolerance={c.get('tolerance')!r}"
            + (f" [{c.get('error_class')}: {c.get('error_msg')}]"
               if c.get("error_class") else "")
        )}
        for c in body.get("cases", [])
        if c.get("verdict") != PASS
    ]

    return {
        "schema_version": _ENVELOPE_SCHEMA_VERSION,
        "check_id": CHECK_ID,
        "label": "Research numeric self-test (R slot)",
        "ran_at": (now or datetime.now(UTC)).isoformat(),
        "status": status,
        "summary": (
            f"research known-answer battery: {n_cases - n_failed - n_errored} "
            f"passed, {n_failed} failed, {n_errored} could not run "
            f"({n_gaps} of the passing cases PIN a known gap at its measured "
            f"value rather than endorse it). Scope: CORRECTNESS of the R slot's "
            f"scoring arithmetic on the deployed libraries — it says nothing "
            f"about whether the stage ran or its inputs were fresh, which is the "
            f"preflight sweep's axis (sf-pipeline-policy §2.3a)."
        ),
        "cadence_minutes": CADENCE_MINUTES,
        "deep_link": None,
        "findings": findings,
        # Beyond the base contract, carried so the console row is diagnosable
        # without opening the full artifact.
        "verdict": verdict,
        "n_cases": n_cases,
        "n_failed": n_failed,
        "n_errored": n_errored,
        "n_known_gaps": n_gaps,
        "code_sha": body.get("code_sha"),
        "libraries": body.get("libraries"),
    }


def publish_console_row(body: dict, *, dry_run: bool = False, s3_client=None) -> str | None:
    """Publish the console row. Returns the s3:// URI, or None if nothing written.

    NEVER raises. A check must not go red because its telemetry did, and this
    module must not be able to fail the pipeline at all. A missing envelope
    renders on the console as ``unreadable``, never ``ok``, so a failed publish
    degrades to a visible gap rather than a false all-clear.
    """
    import json

    envelope = console_envelope(body)
    key = console_envelope_key()
    uri = f"s3://{RESEARCH_BUCKET}/{key}"
    if dry_run:
        logger.info("[dry-run] would publish %s (%s)", uri, envelope["status"])
        return None
    try:
        import boto3

        client = s3_client or boto3.client("s3")
        client.put_object(
            Bucket=RESEARCH_BUCKET, Key=key,
            Body=json.dumps(envelope, indent=2, default=str).encode(),
            ContentType="application/json",
        )
    except Exception:  # noqa: BLE001 — a failed publish must never fail the check
        logger.warning(
            "could not publish the self-test console row to %s — the console will "
            "render this check as `unreadable` (never `ok`), so the gap is visible",
            uri, exc_info=True,
        )
        return None
    return uri
