"""Research producer spec registry (config#1221 / ARCHITECTURE §37).

Champion = the live agentic LangGraph producer (authoritative; emitted by the
research graph, ``build`` is None here). Challengers carry a ``build`` callable
``(run_date, archive_manager, **ctx) -> signals_payload`` that produces a
conforming signals.json from the SAME scanner candidate set (scanner held
constant across producers — a clean selection-only comparison).

A challenger may carry ``build=None`` when its shadow signals are produced by
its OWN pipeline rather than built during the weekly producer run — see
``thinktank_coverage`` below. Such a spec is scored by the leaderboard exactly
like any other challenger, but ``producers.runner`` must NOT try to build it
(``runner`` filters on ``build is not None``; including it would raise
TypeError inside the per-spec except, then trip the completeness gate and turn
a healthy run red).

A spec may instead be ``kind="retired"`` once it is no longer wired into the
live pipeline — this makes liveness a queryable fact (``retired_date``)
instead of a stale ``description`` string a downstream reader (e.g. the
evaluator/backtester's e2e_lift aggregation) has to re-derive by inference.
See ``agentic_sector_teams`` below (config-I2993).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date as _date
from datetime import timedelta as _timedelta

from producers.filling_arms import (
    run_scanner_predictor_direct_producer,
    run_scanner_top20_predictor_producer,
)
from producers.no_agent import run_no_agent_producer
from producers.single_agent import run_single_agent_producer

# champion-challenger-policy.md §3: "A retired arm is scored for a trailing
# window (default: 8 cycles past retired_date), so 'we retired the wrong
# one' is detectable rather than a matter of opinion." Named constants
# (config-I6427) rather than a magic number, since the policy explicitly
# frames "8 cycles" as a default a slot could override, and a "cycle" here
# is one weekly producer-leaderboard run (see scoring/leaderboard_producers.py).
RETIRED_TRAILING_WINDOW_CYCLES = 8
RETIRED_TRAILING_WINDOW_CYCLE_DAYS = 7
RETIRED_TRAILING_WINDOW_DAYS = RETIRED_TRAILING_WINDOW_CYCLES * RETIRED_TRAILING_WINDOW_CYCLE_DAYS


@dataclass(frozen=True)
class ProducerSpec:
    name: str
    kind: str  # "champion" | "challenger" | "retired"
    version: str
    description: str
    build: Callable | None = None  # None for the champion (live agentic graph)
    # ISO date (str) a spec stopped being the live producer, or None while
    # still champion/challenger. Additive field (config-I2993) — defaults to
    # None so every pre-existing spec is unaffected.
    retired_date: str | None = None
    # ── Promotion eligibility (alpha-engine-config-I9277, Brian's ruling
    # 2026-08-29: "for the research arm, we should make all arms promote
    # eligible, including think tank") ──────────────────────────────────────
    #
    # Eligibility is an EXPLICIT RECORDED PROPERTY carrying a reason, never an
    # arm's absence from a hand-maintained list. The defect this closes: the
    # promotion engine's ``VALID_CHAMPIONS`` tuple in crucible-backtester was
    # a SECOND, hand-maintained register that silently omitted
    # ``no_agent_quant`` and ``single_agent_quant`` — the only two arms with
    # sufficient evidence to win. Nothing anywhere recorded that they were
    # excluded, or why: they were simply not typed into a tuple in another
    # repo. "Scored but ineligible, for no recorded reason" is exactly the
    # silent-omission class, and an absence cannot be reviewed, alerted on, or
    # rendered on an audit artifact.
    #
    # So: every arm in this dict is promotion-eligible BY DEFAULT. An
    # exclusion must be stated here, with a reason that appears on the
    # producer leaderboard's ``arms`` block and from there on the weekly
    # champion audit record. ``_assert_eligibility_coherent`` below enforces
    # that the flag and the reason cannot disagree.
    promotion_eligible: bool = True
    ineligible_reason: str | None = None
    # The first-party modules that IMPLEMENTED this arm, recorded when it is
    # retired (alpha-engine-config-I7827). champion-challenger-policy.md §6
    # requires the code to be DELETED rather than left dormant, and this field
    # is what makes that enforceable instead of a matter of memory:
    # `tests/test_retired_producer_not_reachable.py` asserts every name here is
    # absent from the tree AND unreachable from any Lambda handler's import
    # graph. Empty for a live arm.
    retired_modules: tuple[str, ...] = ()
    # WHERE this arm's score is read from (alpha-engine-config-I9307). Exactly
    # two values are legal:
    #
    #   "shadow"       — signals_shadow/{name}/{date}/signals.json, the artifact
    #                    this arm writes itself. Every arm should be here.
    #   "signals_live" — signals/{date}/signals.json, the LIVE artifact.
    #
    # It is a declared field rather than an inference because the second value
    # is a live hazard, and an inference cannot be refused. The champion arm was
    # scored from "signals_live" for seven weeks after the live signals producer
    # became ``signals_envelope`` — which is empty-by-contract and emits no
    # ENTER picks at all — so the arm contributed zero cohort dates while
    # rendering as merely *thin*. ``_assert_score_source_can_carry_output``
    # below now REFUSES that combination at import, so the defect cannot recur
    # silently: it becomes an ImportError with the reason in it.
    score_source: str = "shadow"


RESEARCH_PRODUCERS: dict[str, ProducerSpec] = {
    "agentic_sector_teams": ProducerSpec(
        name="agentic_sector_teams",
        kind="retired",
        version="v1",
        description="RETIRED six-team + macro economist + CIO LangGraph "
        "orchestration — the live ne-weekly-freshness-pipeline SF no longer "
        "invokes this graph (config#1580 ruling); SignalsEnvelope "
        "(config-I2515 Phase B) is the live signals.json producer now. Its "
        "CODE (graph/research_graph.py, graph/reducers.py, the handler's "
        "champion pass, local/run.py) was DELETED on 2026-08-20 under "
        "champion-challenger-policy.md §6 — alpha-engine-config-I7827. This "
        "row stays forever as the historical record; §6 deletes the code, not "
        "the registry entry.",
        build=None,
        # Derivation (config-I2993): research.db team_candidates/cio_evaluations
        # MAX date is trading_day 2026-07-10, i.e. the graph's last production
        # was the Saturday 2026-07-11 weekly cycle (calendar_date; weekly runs
        # tag trading_day = last closed trading day per the dual-track date
        # convention). retired_date is the first calendar date after that last
        # production — mirrors the backtester's day-after-last-use
        # ``cutover_date`` convention (e.g. neutralization_live_forward_ic,
        # cutover 2026-06-22).
        retired_date="2026-07-12",
        # Deleted 2026-08-20 under §6 — alpha-engine-config-I7827. The guard
        # in tests/ fails if any of these comes back or becomes reachable.
        retired_modules=(
            "graph.research_graph",
            "graph.reducers",
            "local.run",
            "local.offline_stubs",
        ),
        # champion-challenger-policy.md §3: retired arms are scored for a
        # trailing window as historical evidence, and §6 makes retirement an
        # operator decision — so a retired arm is never promotion-eligible.
        # Stated, not inferred from ``kind``: the promotion engine reads this
        # flag, and an engine that had to re-derive "retired implies
        # ineligible" is one refactor away from not doing so.
        promotion_eligible=False,
        ineligible_reason=(
            "retired 2026-07-12 (alpha-engine-config-I2993) — scored for the "
            "champion-challenger-policy.md §3 trailing window as historical "
            "evidence only; §6 makes reinstatement an operator decision"
        ),
    ),
    "no_agent_quant": ProducerSpec(
        name="no_agent_quant",
        kind="challenger",
        version="v1",
        description="pure-quant floor: scanner candidates scored by the technical "
        "composite, deterministic top-N ENTER gate, no LLM (config#1221)",
        build=run_no_agent_producer,
    ),
    "single_agent_quant": ProducerSpec(
        name="single_agent_quant",
        kind="challenger",
        version="v1",
        description="single-agent: ONE Sonnet call assesses qual for all scanner "
        "candidates; deterministic quant + composite; no multi-agent fan-out, no "
        "macro/CIO (config#1223 / M3 baseline)",
        build=run_single_agent_producer,
    ),
    "scanner_predictor_direct": ProducerSpec(
        name="scanner_predictor_direct",
        kind="challenger",
        version="v1",
        description="scanner top-~60 candidates passed straight to the "
        "predictor, its top-N by predicted_alpha taken as the arm's picks. "
        "LIVE since 2026-07-13 (config-I2364 operator bootstrap). Registered "
        "here 2026-08-29 (alpha-engine-config-I9277): it had been a promotion "
        "arm for six weeks while existing in NO producer register — only in "
        "crucible-backtester's VALID_CHAMPIONS literal.",
        # alpha-engine-config-I9307: this arm now BUILDS its own shadow, on
        # every weekly pass, regardless of whether it is currently serving.
        #
        # It used to be scored from signals/{date}/signals.json on the theory
        # that "when this arm is champion its picks ARE signals.json". That
        # stopped being true when SignalsEnvelope became the live producer
        # (epic config-I2515 Phase B): the envelope is empty-by-contract and
        # emits ZERO ENTER picks, so the arm contributed nothing to the cohort
        # from 2026-07-18 onward while its row still rendered as `thin`.
        #
        # Building it here — not capturing it in the executor where the picks
        # are actually synthesized — is what satisfies §3: the executor only
        # synthesizes for the arm that is CURRENTLY champion, so an
        # executor-side capture would go dark on the incumbent the moment the
        # pointer moved. See producers/filling_arms.py for the full rationale.
        build=run_scanner_predictor_direct_producer,
    ),
    "scanner_top20_predictor": ProducerSpec(
        name="scanner_top20_predictor",
        kind="challenger",
        version="v1",
        description="scanner top-20 (not top-60) passed directly to the "
        "predictor — the arm Brian's 2026-08-27 ruling names. Registered here "
        "2026-08-29 (alpha-engine-config-I9277); previously only in "
        "crucible-backtester's VALID_CHAMPIONS literal.",
        # alpha-engine-config-I9307: was scored ONLY as a crucible-backtester
        # end-to-end counterfactual — a different source and a different cohort
        # from every other arm on this board (-I9279), i.e. the asymmetry that
        # hid the champion's silence. It now builds its own shadow through the
        # same one writer, so it is on the shared basis like everything else.
        build=run_scanner_top20_predictor_producer,
    ),
    "thinktank_coverage": ProducerSpec(
        name="thinktank_coverage",
        kind="challenger",
        version="v1",
        description="Think Tank coverage arm: per-ticker qualitative theses "
        "(thesis + pillar/moat tiers) ranked by the Think Tank's own rating, "
        "top-CHALLENGER_TOP_N submitted as the arm's picks. Arm 3 of 3 in the "
        "count-matched predictor universe (config-I4983).",
        # build=None ON PURPOSE. Unlike the other challengers, this arm's
        # shadow (signals_shadow/thinktank_coverage/{date}/signals.json) is
        # written by the Think Tank's OWN daily run, not synthesised during
        # the weekly producer pass. `producers.runner` skips specs without a
        # build callable; `scoring.leaderboard_producers` scores them all.
        #
        # Registration was deliberately held back (alpha-engine-config-I5195
        # scope 4b, 2026-07-28): the Think Tank had been dead 11 days on the
        # 900s Lambda ceiling, so registering it then would have added a
        # permanently-missing arm and made every cohort incomplete. That
        # blocker cleared 2026-07-29 when config-I5208 moved the run to EC2
        # spot (ARCHITECTURE §47) and it resumed writing the shadow view.
        build=None,
    ),
}


def challenger_producers() -> list[ProducerSpec]:
    """EVERY challenger arm, including those produced by their own pipeline.

    This is the scoring set: the leaderboard reads each arm's shadow from S3,
    so it does not care who wrote it. Callers that need to BUILD shadows want
    :func:`buildable_challenger_producers` instead.
    """
    return [p for p in RESEARCH_PRODUCERS.values() if p.kind == "challenger"]


def buildable_challenger_producers() -> list[ProducerSpec]:
    """Challengers the weekly producer run is responsible for BUILDING.

    Excludes arms whose shadow is written by their own pipeline (``build is
    None``). Keeping this distinct from :func:`challenger_producers` is
    load-bearing: ``producers.runner`` both calls ``spec.build`` and asserts
    every expected name emitted, so an externally-produced arm in that list
    would raise TypeError and then fail the completeness gate on every run.
    """
    return [p for p in challenger_producers() if p.build is not None]


def retired_producers(as_of: str | _date | None = None) -> list[ProducerSpec]:
    """Retired arms still inside their trailing scoring window
    (champion-challenger-policy.md §3): every ``kind=="retired"`` row whose
    ``retired_date`` falls within ``RETIRED_TRAILING_WINDOW_DAYS`` of
    ``as_of``.

    This is the SCORING set for retired arms: ``scoring.leaderboard_producers``
    feeds every row this returns into the producer leaderboard alongside the
    champion and live challengers, tagged ``kind="retired"`` in the artifact
    so a downstream reader (and any promotion engine filtering on
    ``kind=="challenger"``) can tell historical-evidence-only rows apart from
    a live challenger — retired arms are scored but never promotion-eligible.

    A retired arm that has aged past the window is EXCLUDED (not returned
    here), even though it stays in ``RESEARCH_PRODUCERS`` forever as a
    historical record — retirement per §6 does not delete the registry row,
    only the code, and this selector is what makes "still inside the trailing
    window" a queryable fact rather than something a caller has to re-derive.

    ``as_of`` is the cohort/run date driving the scoring pass — an ISO date
    string (matches the ``date_str`` callers already thread through
    ``scoring/leaderboard_producers.py``) or a ``date``. Defaults to today
    (UTC) so callers that don't have an explicit run date still get a sane
    answer; tests should always pass an explicit ``as_of`` rather than
    relying on wall-clock time.
    """
    if as_of is None:
        cutoff = _date.today()
    elif isinstance(as_of, str):
        cutoff = _date.fromisoformat(as_of)
    else:
        cutoff = as_of

    out: list[ProducerSpec] = []
    for p in RESEARCH_PRODUCERS.values():
        if p.kind != "retired" or not p.retired_date:
            continue
        retired = _date.fromisoformat(p.retired_date)
        window_end = retired + _timedelta(days=RETIRED_TRAILING_WINDOW_DAYS)
        if retired <= cutoff <= window_end:
            out.append(p)
    return out


def promotion_eligible_producers() -> list[ProducerSpec]:
    """EVERY arm the promotion engine may move the live pointer onto.

    THE single register for promotion eligibility (alpha-engine-config-I9277,
    Brian's ruling 2026-08-29: "for the research arm, we should make all arms
    promote eligible, including think tank"). crucible-backtester's
    ``champion_promotion.py`` no longer carries its own ``VALID_CHAMPIONS``
    tuple; it resolves the arm set from the producer leaderboard's ``arms``
    block, which is this function projected onto the artifact.

    Note what is NOT filtered here: ``confidence``, ``n_dates_scored``, and
    the sign of an arm's alpha are all evidence questions the GATE weighs,
    never eligibility questions. An arm grading negative is still eligible —
    that is the whole point of a winner-take-all slot, and Brian ruled with
    both currently-scored arms grading negative.
    """
    return [p for p in RESEARCH_PRODUCERS.values() if p.promotion_eligible]


def ineligible_producers() -> dict[str, str]:
    """``{arm_name: reason}`` for every registered arm that may NOT be promoted.

    The reason is load-bearing: it is carried onto the producer leaderboard and
    from there onto the weekly champion audit record, so "this arm was scored
    but could not win" is always a stated fact with a justification attached,
    never an arm's silent absence from a tuple.
    """
    return {
        p.name: (p.ineligible_reason or "no reason recorded")
        for p in RESEARCH_PRODUCERS.values()
        if not p.promotion_eligible
    }


def _assert_eligibility_coherent() -> None:
    """Import-time guard: the flag and the reason can never disagree.

    An ineligible arm with no reason reintroduces exactly the silent-omission
    defect I9277 closes — the exclusion would render on the audit artifact as
    a blank, which reads as "no reason to state" rather than "nobody stated
    one". An ELIGIBLE arm carrying a reason is the same bug mirrored: a reader
    (or a future filter) would trust the prose over the flag.
    """
    for spec in RESEARCH_PRODUCERS.values():
        if not spec.promotion_eligible and not spec.ineligible_reason:
            raise ValueError(
                f"producer {spec.name!r} is promotion_eligible=False with no "
                "ineligible_reason — an exclusion MUST carry a recorded reason "
                "(alpha-engine-config-I9277)"
            )
        if spec.promotion_eligible and spec.ineligible_reason:
            raise ValueError(
                f"producer {spec.name!r} is promotion_eligible=True but carries "
                f"ineligible_reason={spec.ineligible_reason!r} — the flag and the "
                "reason contradict each other (alpha-engine-config-I9277)"
            )


_assert_eligibility_coherent()


def champion_producer() -> ProducerSpec | None:
    """The live ``kind=="champion"`` producer, or ``None`` when no spec is
    currently registered as champion (config-I2993: retiring a spec does not
    auto-promote a successor — registering the live producer as a new champion
    spec is tracked separately). Callers MUST treat ``None`` as a legitimate,
    non-error state, not an invariant violation."""
    return next((p for p in RESEARCH_PRODUCERS.values() if p.kind == "champion"), None)


# ── Producer/champion compatibility matrix (config#5713) ───────────────────
# The declared registry row behind the executor's read-path coherence
# assertion (alpha-engine executor/champion.py::
# assert_producer_champion_coherence). Champion-challenger-policy §2 has an
# arm name its slot; this matrix names which producers RELY on which arms:
#
# * ``EMPTY_BUY_CANDIDATES_BY_CONTRACT_PRODUCERS`` — producers whose
#   signals.json ``buy_candidates`` is ALWAYS ``[]`` by contract (they never
#   propose entries themselves; see scoring/signals_envelope.py's docstring
#   caveat). An empty list from such a producer is not a market condition —
#   it is the producer's honest "no opinion" — so it is only safe under a
#   champion arm that synthesizes candidates.
# * ``FILLING_CHAMPION_ARMS`` — arms that synthesize ``buy_candidates`` in
#   the executor (apply_champion_selection) and are therefore the only
#   legitimate partners for empty-by-contract producers.
# * ``NOOP_CHAMPION_ARMS`` — arms that pass ``buy_candidates`` through
#   untouched; pairing an empty-by-contract producer with one of these
#   guarantees no new entry is ever proposed, silently (the defect
#   config#5713 exists to detect).
#
# The executor reads the runtime values from its private risk.yaml
# (union-extended over its own fail-closed baselines); this row is the
# policy-level declaration and the drift-test anchor for the producer side.
EMPTY_BUY_CANDIDATES_BY_CONTRACT_PRODUCERS = ("signals_envelope",)

# DERIVED from the register, not typed a second time (alpha-engine-config
# -I9277). As a hand-maintained literal this tuple was
# ``("scanner_predictor_direct", "thinktank_coverage")`` — it had silently
# gone stale when ``scanner_top20_predictor`` became a promotion arm on
# 2026-08-27, so a promotion onto that arm would have hit the executor's
# coherence assertion with the arm in NEITHER the filling nor the noop set.
# Same defect class as VALID_CHAMPIONS, same fix: one register, projected.
FILLING_CHAMPION_ARMS = tuple(
    sorted(p.name for p in RESEARCH_PRODUCERS.values() if p.promotion_eligible)
)
NOOP_CHAMPION_ARMS = ("agentic",)

# The producer that actually writes ``signals/{date}/signals.json`` today.
# Declared here so the guard below can be evaluated at import instead of being
# a fact somebody has to remember (epic config-I2515 Phase B moved this from
# the agentic graph to the envelope, and nothing downstream noticed).
LIVE_SIGNALS_PRODUCER = "signals_envelope"

SCORE_SOURCE_SHADOW = "shadow"
SCORE_SOURCE_SIGNALS_LIVE = "signals_live"
VALID_SCORE_SOURCES = (SCORE_SOURCE_SHADOW, SCORE_SOURCE_SIGNALS_LIVE)


def _assert_score_source_can_carry_output() -> None:
    """Refuse, at import, any arm scored from a source that cannot carry its
    output (alpha-engine-config-I9307).

    THE DEFECT THIS CLOSES, in one sentence: the champion arm was scored by
    reading ENTER picks out of ``signals/{date}/signals.json`` while the live
    producer of that artifact was ``signals_envelope``, which is
    empty-by-contract and emits no ENTER picks at all — so the arm was
    structurally incapable of scoring, contributed 2 cohort dates in 7 weeks,
    and rendered as a merely *thin* row the whole time.

    Nothing in the fleet could see that, because it is a relationship between
    two facts held in different modules: which artifact an arm is scored FROM,
    and whether that artifact's producer ever emits picks. Both facts are now
    declared in this file, so the relationship is checkable — and it is checked
    HERE, at import, rather than alerted on per cycle, because a structural
    impossibility is not a measurement outcome (champion-challenger-policy.md
    §7.2). A registry that cannot produce a comparable measurement must not
    load at all.

    Raises ``ValueError`` (surfacing as an ImportError at the call site) rather
    than warning: §7.2's dominant bug class is a record asserting an action
    that never happened, and a warning here would produce exactly that.
    """
    for spec in RESEARCH_PRODUCERS.values():
        if spec.score_source not in VALID_SCORE_SOURCES:
            raise ValueError(
                f"producer {spec.name!r} declares score_source="
                f"{spec.score_source!r}, which is not one of {VALID_SCORE_SOURCES}"
            )
        if spec.kind == "retired":
            continue
        if (
            spec.score_source == SCORE_SOURCE_SIGNALS_LIVE
            and LIVE_SIGNALS_PRODUCER in EMPTY_BUY_CANDIDATES_BY_CONTRACT_PRODUCERS
        ):
            raise ValueError(
                f"producer {spec.name!r} declares score_source="
                f"{SCORE_SOURCE_SIGNALS_LIVE!r}, but the live producer of "
                f"signals/{{date}}/signals.json is {LIVE_SIGNALS_PRODUCER!r}, "
                "which is EMPTY-BY-CONTRACT (see "
                "EMPTY_BUY_CANDIDATES_BY_CONTRACT_PRODUCERS): it never emits an "
                "ENTER pick, so this arm can never score and would render as a "
                "thin row forever. Give the arm its own signals_shadow/ writer "
                "(producers/filling_arms.py) and score_source='shadow'. "
                "alpha-engine-config-I9307."
            )
        if spec.score_source == SCORE_SOURCE_SHADOW and spec.build is None:
            # Legitimate: the arm's own pipeline writes the shadow on its own
            # cadence (thinktank_coverage). Not an error — but it IS the
            # registration/writer split that must stay visible, so it is stated
            # rather than inferred.
            continue


_assert_score_source_can_carry_output()


def score_source_for(name: str) -> str:
    """Where ``name``'s score is read from. Unregistered arms default to the
    shadow prefix — the only source on which an unregistered arm could ever be
    compared like the rest."""
    spec = RESEARCH_PRODUCERS.get(name)
    return spec.score_source if spec else SCORE_SOURCE_SHADOW
