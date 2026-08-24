"""Historical reconstruction of the weight-vector arms (alpha-engine-config-I8262).

Three of the universe-cut slot's five arms were first emitted in the week they
were written, so their first read at a matured horizon lands in 2027-02 — for
arms that are pure deterministic functions of an input S3 has retained since
2026-05-21. ``momzero`` and ``hard3`` read
``factors/profiles/{date}/by_ticker.json`` plus a weight vector and nothing
else, so their past picks are computable exactly.

What these tests defend is not the arithmetic — that is the live path's, shared.
It is the three things a backfill can destroy while every other assertion passes:

    the champion's archived picks       never recomputed, never modified
    a contemporaneous cut               never overwritten by a reconstruction
    a reconstructed cut's provenance    per CUT, derived, and carrying the
                                        contamination exposure of the dates it
                                        covers

Red-run evidence for each guard is recorded in the PR body (§7.4).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring.universe_membership import (  # noqa: E402
    ATTRACTIVENESS_FEED_TOP_N,
    BACKFILLABLE_ARM_PREFIXES,
    CHALLENGER_CUT_PREFIX,
    HARD3_CUT_PREFIX,
    HARD3_PILLAR_WEIGHTS,
    MOMZERO_CUT_PREFIX,
    MOMZERO_PILLAR_WEIGHTS,
    OBSERVE_ONLY_CUTS,
    PILLAR_ORDER_FOR_WEIGHTS,
    VENDOR_FUNDAMENTAL_PILLARS,
    VENDOR_FUNDAMENTALS_REPAIR_DATE,
    UniverseMembershipError,
    arm_cut_source,
    assert_backfill_preserved_live_cuts,
    build_backfilled_arm_cuts,
    build_universe_membership,
    merge_backfilled_cuts,
    vendor_fundamentals_exposure,
    weight_vector_cut_block,
)

HARD3_CUT = f"{HARD3_CUT_PREFIX}{ATTRACTIVENESS_FEED_TOP_N}"
MOMZERO_CUT = f"{MOMZERO_CUT_PREFIX}{ATTRACTIVENESS_FEED_TOP_N}"
CHAMPION_60 = f"attractiveness_top_{ATTRACTIVENESS_FEED_TOP_N}"
DATE_PRE_REPAIR = "2026-06-05"
DATE_POST_REPAIR = "2026-08-21"

_FACTOR_KEY = {
    "quality": "quality_score",
    "value": "value_score",
    "momentum": "momentum_score",
    "growth": "growth_score",
    "stewardship": "stewardship_score",
    "defensiveness": "low_vol_score",
}


def _profiles(n: int = 120, *, placeholder_vendor_half: bool = False) -> dict:
    """A factor-profile file in the live shape.

    ``placeholder_vendor_half`` reproduces the pre-2026-08-20 state
    (alpha-engine-config-I8255): quality / growth / stewardship carry ONE
    distinct value across every name, so they contribute no cross-sectional
    spread and the six-pillar champion is a three-pillar ranking by accident.
    """
    out = {}
    for i in range(n):
        t = f"T{i:03d}"
        vals = {
            "quality": 50.0 if placeholder_vendor_half else float((i * 7) % 100),
            "growth": 50.0 if placeholder_vendor_half else float((i * 11) % 100),
            "stewardship": 50.0 if placeholder_vendor_half else float((i * 13) % 100),
            "value": float((i * 17) % 100),
            "momentum": float((i * 23) % 100),
            "defensiveness": float((i * 29) % 100),
        }
        out[t] = {"sector": "Tech", **{_FACTOR_KEY[p]: v for p, v in vals.items()}}
    return out


def _membership(profiles: dict, run_date: str = DATE_PRE_REPAIR) -> dict:
    """An archived artifact for ``run_date``, champion cuts and ranks included."""
    names = sorted(profiles)
    attractiveness = {t: float(len(names) - i) for i, t in enumerate(names)}
    return build_universe_membership(
        run_date, names[:ATTRACTIVENESS_FEED_TOP_N], attractiveness,
    )


# ══ Provenance is derived, never a literal (§7.5) ═══════════════════════════


def test_mom121_source_names_the_prefix_its_reader_actually_opens():
    """RED on origin/main.

    The membership artifact annotated every mom121 cut
    ``scanner/factor_profiles_shadow/mom121/{date}/profiles.json`` — a prefix
    that has never existed on S3, in either its directory OR its filename,
    while ``challenger_attractiveness_for_run`` has always read
    ``factor_scoring.CHALLENGER_PROFILE_PREFIX`` +
    ``/{date}/by_ticker.json``. Reader and annotation now resolve one constant,
    so they cannot disagree again.

    I8262 read that as "nothing writes the shadow profiles". It is the opposite:
    they ARE written (`factor_scoring` line ~876, `write_factor_profiles_to_s3`
    with ``key_prefix=CHALLENGER_PROFILE_PREFIX``) and dated snapshots exist
    from 2026-08-18. Only the string was wrong.
    """
    from scoring.factor_scoring import CHALLENGER_PROFILE_PREFIX

    src = arm_cut_source(CHALLENGER_CUT_PREFIX, "2026-08-21")
    read_key = f"{CHALLENGER_PROFILE_PREFIX}/2026-08-21/by_ticker.json"
    assert src.split("::")[0] == read_key
    assert "scanner/factor_profiles_shadow" not in src


def test_weight_arm_source_names_the_pillars_the_vector_actually_zeroes():
    """A hand-written ablation annotation goes stale the moment the vector is
    retuned; this one is generated FROM the vector."""
    momzero = arm_cut_source(MOMZERO_CUT_PREFIX, DATE_PRE_REPAIR)
    hard3 = arm_cut_source(HARD3_CUT_PREFIX, DATE_PRE_REPAIR)
    assert momzero.endswith("@momentum=0")
    assert hard3.endswith("@quality=growth=stewardship=0")
    for src in (momzero, hard3):
        assert src.startswith(f"factors/profiles/{DATE_PRE_REPAIR}/by_ticker.json")


@pytest.mark.parametrize(
    ("prefix", "cut"),
    [(HARD3_CUT_PREFIX, HARD3_CUT), (MOMZERO_CUT_PREFIX, MOMZERO_CUT)],
)
def test_live_and_backfilled_cut_blocks_are_the_same_shape(prefix, cut):
    """RED on origin/main for ``momzero``: the live producer carried its own
    block literal per arm and momzero's annotated ``@momentum_weight=0`` — a
    key name, not a pillar name — so a reconstruction deriving the ablation
    from the vector could not be byte-comparable with it. One builder, two
    callers, and the annotation names the pillar the vector actually zeroes."""
    profiles = _profiles()
    ranked = sorted(profiles)
    live = build_universe_membership(
        DATE_POST_REPAIR,
        ranked[:ATTRACTIVENESS_FEED_TOP_N],
        {t: float(120 - i) for i, t in enumerate(ranked)},
        hard3_attractiveness={t: float(i) for i, t in enumerate(ranked)},
        momzero_attractiveness={t: float((i * 37) % 120) for i, t in enumerate(ranked)},
    )["cuts"][cut]
    built = weight_vector_cut_block(
        prefix, ATTRACTIVENESS_FEED_TOP_N, live["tickers"], DATE_POST_REPAIR
    )
    assert built["basis"] == live["basis"]
    assert built["source"] == live["source"]
    assert built["size"] == live["size"]


# ══ Contamination is computed from the arm's own weights (§7.5) ═════════════


def test_exposure_is_derived_from_the_weight_vector_not_labelled():
    """RED against any implementation that records a per-arm contamination
    verdict as a constant: retune the vector and the verdict must move with it.
    """
    assert vendor_fundamentals_exposure(MOMZERO_PILLAR_WEIGHTS, DATE_PRE_REPAIR)[
        "weight_fraction_on_placeholder_pillars"
    ] == pytest.approx(0.6)
    assert vendor_fundamentals_exposure(HARD3_PILLAR_WEIGHTS, DATE_PRE_REPAIR)[
        "weight_fraction_on_placeholder_pillars"
    ] == pytest.approx(0.0)
    equal_weight_champion = dict.fromkeys(PILLAR_ORDER_FOR_WEIGHTS, 1.0)
    assert vendor_fundamentals_exposure(equal_weight_champion, DATE_PRE_REPAIR)[
        "weight_fraction_on_placeholder_pillars"
    ] == pytest.approx(0.5)

    retuned = {**HARD3_PILLAR_WEIGHTS, "quality": 1.0}
    assert vendor_fundamentals_exposure(retuned, DATE_PRE_REPAIR)["contaminated"] is True


def test_the_two_arms_get_DIFFERENT_verdicts_on_the_same_date():
    """I8262 deliverable 4: do not backfill all three arms identically. The
    exposure is a property of the arm, not of the backfill run."""
    d = DATE_PRE_REPAIR
    assert vendor_fundamentals_exposure(MOMZERO_PILLAR_WEIGHTS, d)["contaminated"] is True
    assert vendor_fundamentals_exposure(HARD3_PILLAR_WEIGHTS, d)["contaminated"] is False


def test_post_repair_dates_are_uncontaminated_for_every_arm():
    for weights in (MOMZERO_PILLAR_WEIGHTS, HARD3_PILLAR_WEIGHTS):
        assert vendor_fundamentals_exposure(weights, DATE_POST_REPAIR)["contaminated"] is False
    assert vendor_fundamentals_exposure(MOMZERO_PILLAR_WEIGHTS, VENDOR_FUNDAMENTALS_REPAIR_DATE)[
        "run_date_is_pre_repair"
    ] is False


def test_the_placeholder_pillars_are_exactly_the_vendor_half():
    """A fourth pillar creeping into this tuple would relabel a clean arm as
    contaminated; a missing one would clear a contaminated arm silently."""
    assert set(VENDOR_FUNDAMENTAL_PILLARS) == {"quality", "growth", "stewardship"}
    assert not set(VENDOR_FUNDAMENTAL_PILLARS) & {"value", "momentum", "defensiveness"}


# ══ The reconstruction itself ═══════════════════════════════════════════════


def test_backfill_emits_both_weight_arms_stamped_and_exposed():
    profiles = _profiles()
    membership = _membership(profiles)
    cuts, refused = build_backfilled_arm_cuts(
        DATE_PRE_REPAIR,
        profiles,
        population=sorted(membership["ranks"]),
        champion_cuts=membership["cuts"],
        reconstructed_at="2026-08-24T00:00:00+00:00",
    )
    assert MOMZERO_CUT in cuts and HARD3_CUT in cuts, refused
    for name, block in cuts.items():
        assert block["size"] == ATTRACTIVENESS_FEED_TOP_N or block["size"] == 20
        assert "alpha-engine-config-I8262" in block["backfilled_from"], name
        assert block["vendor_fundamentals_exposure"]["repaired_from"] == VENDOR_FUNDAMENTALS_REPAIR_DATE
        assert 0.0 <= block["champion_overlap"] <= 1.0
    assert cuts[MOMZERO_CUT]["vendor_fundamentals_exposure"]["contaminated"] is True
    assert cuts[HARD3_CUT]["vendor_fundamentals_exposure"]["contaminated"] is False


def test_a_backfilled_cut_is_distinguishable_from_a_contemporaneous_one():
    """I8262 deliverable 3. Without this the cohort silently mixes two
    provenances and no reader can tell which weeks were reconstructed."""
    profiles = _profiles()
    membership = _membership(profiles, DATE_POST_REPAIR)
    live = build_universe_membership(
        DATE_POST_REPAIR,
        sorted(profiles)[:ATTRACTIVENESS_FEED_TOP_N],
        {t: float(120 - i) for i, t in enumerate(sorted(profiles))},
        hard3_attractiveness={t: float(i) for i, t in enumerate(sorted(profiles))},
    )["cuts"][HARD3_CUT]
    assert "backfilled_from" not in live

    cuts, _ = build_backfilled_arm_cuts(
        DATE_POST_REPAIR, profiles,
        population=sorted(membership["ranks"]), champion_cuts=membership["cuts"],
    )
    assert "backfilled_from" in cuts[HARD3_CUT]


def test_the_arm_ranks_the_champions_population_not_the_raw_profiles_file():
    """The profiles file legitimately carries Metron-supplemental and
    fundamental-only rows the scanner never evaluated (I7844). Attractiveness
    is a percentile over the population being scored, so ranking those extra
    rows would give the arm a different population from the champion it is
    compared against — and the board would attribute a population difference to
    the weight vector (§4)."""
    profiles = _profiles()
    membership = _membership(profiles)
    population = sorted(membership["ranks"])
    # Top-scoring on every pillar, so without the restriction they would take
    # the whole cut — an intruder that could never rank makes this test blind.
    intruders = {
        f"Z{i:03d}": {"sector": "Tech", **dict.fromkeys(_FACTOR_KEY.values(), 100.0)}
        for i in range(30)
    }
    cuts, _ = build_backfilled_arm_cuts(
        DATE_PRE_REPAIR, {**profiles, **intruders},
        population=population, champion_cuts=membership["cuts"],
    )
    for block in cuts.values():
        assert not set(block["tickers"]) & set(intruders)

    clean, _ = build_backfilled_arm_cuts(
        DATE_PRE_REPAIR, profiles, population=population, champion_cuts=membership["cuts"],
    )
    assert cuts[HARD3_CUT]["tickers"] == clean[HARD3_CUT]["tickers"], (
        "an out-of-population row shifted the percentiles, which means the "
        "restriction ran after the scoring chokepoint instead of before it"
    )


def test_an_arm_that_resolves_to_the_champion_is_refused_not_emitted():
    """The pre-repair case, and it is the EXPECTED one for ``hard3``, not a
    remote one: the champion was a three-pillar ranking by accident on those
    dates, so an arm that zeroes exactly the three placeholder pillars can
    reproduce it. Emitting it would enter the cohort as a clone and report the
    champion against itself as a tie (§4)."""
    names = [f"T{i:03d}" for i in range(80)]
    profiles = _profiles(80)
    champion_cuts = {
        CHAMPION_60: {
            "basis": "attractiveness_rank",
            "size": ATTRACTIVENESS_FEED_TOP_N,
            "tickers": names[:ATTRACTIVENESS_FEED_TOP_N],
        },
    }
    cuts, refused = build_backfilled_arm_cuts(
        DATE_PRE_REPAIR, profiles, population=names, champion_cuts=champion_cuts,
    )
    # Force the collision: the champion cut IS whatever hard3 picked.
    champion_cuts[CHAMPION_60]["tickers"] = sorted(cuts[HARD3_CUT]["tickers"])
    cuts2, refused2 = build_backfilled_arm_cuts(
        DATE_PRE_REPAIR, profiles, population=names, champion_cuts=champion_cuts,
    )
    assert HARD3_CUT not in cuts2
    assert "vacuous" in refused2[HARD3_CUT]


def test_an_unpaired_arm_is_refused_rather_than_scored_alone():
    """No champion cut at that width means nothing to count-match or test
    vacuity against — the row would go on the board unpaired."""
    profiles = _profiles(80)
    cuts, refused = build_backfilled_arm_cuts(
        DATE_PRE_REPAIR, profiles, population=sorted(profiles), champion_cuts={},
    )
    assert cuts == {}
    assert all("count-match" in r for r in refused.values()), refused


def test_a_short_population_is_refused_not_narrowed():
    """A short arm turns a weight-vector comparison into a breadth one (§4)."""
    names = [f"T{i:03d}" for i in range(30)]
    profiles = {t: p for t, p in _profiles(80).items() if t in names}
    champion_cuts = {CHAMPION_60: {"size": 60, "tickers": names}}
    cuts, refused = build_backfilled_arm_cuts(
        DATE_PRE_REPAIR, profiles, population=names, champion_cuts=champion_cuts,
    )
    assert HARD3_CUT not in cuts
    assert "30 of 60" in refused[HARD3_CUT]


def test_absent_profiles_fail_loud():
    with pytest.raises(UniverseMembershipError, match="no factor profiles"):
        build_backfilled_arm_cuts(DATE_PRE_REPAIR, {}, population=["T000"], champion_cuts={})


# ══ The champion's history is never touched ════════════════════════════════


def test_the_champion_is_not_a_backfillable_arm():
    """The single rule the whole script is built around. Recomputing the
    champion from today's factor store would restate the fundamentals it ranked
    on and inject look-ahead into the one arm whose history is real."""
    assert BACKFILLABLE_ARM_PREFIXES == (MOMZERO_CUT_PREFIX, HARD3_CUT_PREFIX)
    assert not any("attractiveness_top_".startswith(p) for p in BACKFILLABLE_ARM_PREFIXES)
    assert CHALLENGER_CUT_PREFIX not in BACKFILLABLE_ARM_PREFIXES, (
        "mom121's 12-1 profiles are a separate artifact whose first snapshot is "
        "2026-08-18 — there is nothing earlier to reconstruct FROM, and "
        "reconstructing it from the champion's profiles would emit the champion"
    )


def test_merge_refuses_to_write_any_cut_outside_the_backfillable_arms():
    """A backfill that COULD write the champion's key is one bad argument away
    from rewriting the history every arm is measured against — so the refusal
    is by arm name, checked before anything else, and it holds for a name that
    is not in the artifact at all (where the never-overwrite rule cannot help).
    """
    membership = _membership(_profiles())
    novel = "attractiveness_notanarm_top_60"
    assert novel not in membership["cuts"]
    merged, refused = merge_backfilled_cuts(
        membership,
        {
            CHAMPION_60: {"size": 60, "tickers": ["HACKED"]},
            novel: {"size": 60, "tickers": ["ALSO_HACKED"]},
        },
    )
    assert novel not in merged["cuts"]
    assert "not a backfillable arm" in refused[novel]
    assert CHAMPION_60 in refused
    assert merged["cuts"][CHAMPION_60]["tickers"] == membership["cuts"][CHAMPION_60]["tickers"]


def test_merge_never_overwrites_a_contemporaneous_cut():
    """A live write is the record of what the run actually served; a
    reconstruction that replaced it would make that unrecoverable, silently."""
    profiles = _profiles()
    membership = _membership(profiles, DATE_POST_REPAIR)
    membership["cuts"][HARD3_CUT] = {
        "basis": "attractiveness_rank_hard3", "size": 60,
        "tickers": sorted(profiles)[:60], "source": "live",
    }
    merged, refused = merge_backfilled_cuts(
        membership, {HARD3_CUT: {"size": 60, "tickers": ["OTHER"], "source": "backfill"}}
    )
    assert merged["cuts"][HARD3_CUT]["source"] == "live"
    assert "already written live" in refused[HARD3_CUT]
    assert "backfilled_arms" not in merged


def test_the_artifact_is_not_stamped_as_a_whole_reconstruction():
    """``backfilled_from`` at artifact level means "all of this is
    reconstructed", which is false here — the champion's picks in it are
    first-class. Provenance is per CUT, plus an index."""
    profiles = _profiles()
    membership = _membership(profiles)
    cuts, _ = build_backfilled_arm_cuts(
        DATE_PRE_REPAIR, profiles,
        population=sorted(membership["ranks"]), champion_cuts=membership["cuts"],
    )
    merged, _ = merge_backfilled_cuts(membership, cuts)
    assert "backfilled_from" not in merged
    assert set(merged["backfilled_arms"]) == set(cuts)


@pytest.mark.parametrize("damage", ["dropped", "modified"])
def test_preservation_guard_catches_a_damaged_live_record(damage):
    """RED by construction: the guard is run against an artifact carrying the
    exact damage it exists to catch. A backfill that mutated the archived cuts
    would not produce a wrong number — it would produce an unfalsifiable one,
    because those cuts are the denominator of every arm comparison."""
    membership = _membership(_profiles())
    after = {**membership, "cuts": dict(membership["cuts"])}
    if damage == "dropped":
        after["cuts"].pop(CHAMPION_60)
        match = "DROPPED"
    else:
        after["cuts"][CHAMPION_60] = {
            **after["cuts"][CHAMPION_60], "tickers": ["REWRITTEN"],
        }
        match = "MODIFIED"
    with pytest.raises(UniverseMembershipError, match=match):
        assert_backfill_preserved_live_cuts(membership, after, DATE_PRE_REPAIR)


def test_preservation_guard_passes_on_an_honest_merge():
    """A guard that cannot pass is as useless as one that cannot fail."""
    profiles = _profiles()
    membership = _membership(profiles)
    cuts, _ = build_backfilled_arm_cuts(
        DATE_PRE_REPAIR, profiles,
        population=sorted(membership["ranks"]), champion_cuts=membership["cuts"],
    )
    merged, _ = merge_backfilled_cuts(membership, cuts)
    assert_backfill_preserved_live_cuts(membership, merged, DATE_PRE_REPAIR)


# ══ Registration — a reconstructed arm must already be a scored arm ═════════


def test_every_backfillable_arm_is_already_on_the_measurement_surface():
    """Writing shadow output for an arm the board does not score is the
    registered-but-unscored rumour §3 warns about. The backfill adds history to
    arms that are ALREADY scored; it must never introduce a new arm name."""
    for prefix in BACKFILLABLE_ARM_PREFIXES:
        assert f"{prefix}{ATTRACTIVENESS_FEED_TOP_N}" in OBSERVE_ONLY_CUTS, prefix


# ══ The script's default is a dry run ══════════════════════════════════════


def test_the_backfill_script_writes_nothing_unless_told_to():
    """A production data write is deliberate, in-region and opt-in. The default
    invocation reads S3 and writes nothing."""
    import importlib.util

    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts", "backfill_cut_arms.py",
    )
    src = open(path).read()
    assert '"--write"' in src
    assert "--overwrite" not in src, (
        "no overwrite flag: a contemporaneous cut is never replaced, so there is "
        "nothing an overwrite mode could legitimately do"
    )
    assert "universe_membership/latest.json" not in src.split('"""', 2)[2], (
        "the backfill must never write the pointer the predictor resolves its "
        "live universe from"
    )
    spec = importlib.util.spec_from_file_location("backfill_cut_arms", path)
    assert spec is not None
