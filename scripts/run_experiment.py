#!/usr/bin/env python3
"""Drive ONE research (R-slot) experiment arm from definition to graded verdict.

WHY THIS EXISTS. The R-slot critical path — "define a variant" to "graded
verdict versus the champion" — is four stages, and every one of them is
reachable with nothing but boto3 credentials: no Step Function, no EC2, no
executor, no predictor, no trading box, no console.

    1. define   a ``ProducerSpec`` in ``producers/registry.py``
    2. champion ``signals/{date}/signals.json`` (already produced)
    3. produce  ``spec.build()`` -> ``signals_shadow/{arm}/{date}/signals.json``
    4. grade    ``scoring.leaderboard_producers.build_producer_leaderboard``

Stages 3 and 4 had NO command-line entry point. They were reachable only from
``lambda/handler.py`` and ``lambda/eval_rolling_mean_handler.py``, so grading a
single new variant meant three unscripted hand-calls from a Python REPL. That
is the whole gap this closes.

WHY IT DOES NOT GO THROUGH ``producers.runner.run_challengers``. That function
is the weekly cohort's completeness gate: it builds EVERY always-on buildable
challenger and raises ``ChallengerShadowGapError`` if any of them fails. That is
correct for the weekly run — an always-on producer emitting nothing is a defect
— and exactly wrong for hand-driven experimentation, where one unrelated broken
arm would starve the arm under test of its verdict. So this script calls
``spec.build`` for the ONE named arm directly and never touches the gate. The
gate is unchanged and still governs the weekly run.

GRADING IS ALREADY PER-ARM-TOLERANT — measured, not assumed.
``build_producer_leaderboard`` does not build anything. It enumerates cohort
dates from what is ALREADY under ``signals_shadow/``, loads one ``SpecHistory``
per registered spec, and scores each independently (``score_leaderboard``
isolates per-spec failures into that spec's own row). An arm with no shadow
artifact scores as an honest empty row; it does not take out its neighbours.
So a single-arm verdict needs no cohort faking and none is done here.

THE VERDICT IS RENDERED, NOT INVENTED. Every number printed is read off the
leaderboard artifact the weekly run writes. The evidence floor shown is the
slot's own ``min_dates_for_inference`` (``LEADERBOARD_SLOTS["producer"]``) and
the per-row ``confidence`` / ``measurability`` the scorer already computes —
this script declares no significance rule of its own.

WRITES. Nothing is written without ``--write``, matching
``scripts/backfill_cut_arms.py`` and ``scripts/backfill_filling_arm_shadows.py``.
``--write`` covers BOTH production prefixes this touches:
``signals_shadow/{arm}/{date}/signals.json`` and
``research/producer_leaderboard/{date}.json``.

Caveat, stated rather than hidden: grading an UNMEASURABLE board fires the same
best-effort observe alert the Lambda would, dry run or not — that publish lives
inside ``build_producer_leaderboard`` and this script deliberately does not
reach in and mute production alerting.

Usage::

    # Grade one arm against the champion, writing nothing (DEFAULT):
    AWS_PROFILE=ne-admin python3 scripts/run_experiment.py --arm no_agent_quant --date 2026-08-21

    # Build this arm's shadow for the date first, then grade it:
    AWS_PROFILE=ne-admin python3 scripts/run_experiment.py --arm no_agent_quant --date 2026-08-21 --produce

    # Same, persisting the shadow AND the leaderboard artifact:
    AWS_PROFILE=ne-admin python3 scripts/run_experiment.py --arm no_agent_quant --date 2026-08-21 --produce --write

    # A range of produce dates, one verdict as of the last:
    AWS_PROFILE=ne-admin python3 scripts/run_experiment.py --arm no_agent_quant --date 2026-08-21 --produce --produce-date 2026-08-07 --produce-date 2026-08-14 --produce-date 2026-08-21

``--write`` performs production data-repo writes and belongs on an in-region box
(the trading EC2 off-market-hours), not on a laptop — same rule as every other
manual write to this bucket.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boto3  # noqa: E402

from producers.registry import RESEARCH_PRODUCERS  # noqa: E402
from scoring.leaderboard_producers import (  # noqa: E402
    _cohort_dates,
    _load_producer_specs,
    _resolve_horizons,
    _resolve_realized_returns_by_horizon,
    build_producer_leaderboard,
    median_cohort_spacing_days,
)
from scoring.leaderboard_scoring import (  # noqa: E402
    COMPARISON_NO_COMMON_COHORT,
    overlap_lags_for,
    paired_alpha_vs_champion,
    slot_spec,
    strict_cohort_intersection,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("run_experiment")

DEFAULT_BUCKET = os.environ.get("S3_BUCKET", "alpha-engine-research")
SHADOW_KEY = "signals_shadow/{arm}/{date}/signals.json"
SLOT_ID = "producer"


class ExperimentError(RuntimeError):
    """The requested experiment cannot be run as asked.

    Always raised, never returned as a status: an unknown arm or an
    unmeasurable board must stop the human, not print a plausible blank.
    """


# ── Stage 1: resolve the arm ─────────────────────────────────────────────────


def resolve_arm(name: str):
    """The ``ProducerSpec`` registered under ``name``.

    Refuses loudly. A typo'd arm name must never fall through to "no rows for
    that arm", which is indistinguishable from a real arm that produced
    nothing — the exact conflation champion-challenger-policy.md §7.2 forbids.
    """
    spec = RESEARCH_PRODUCERS.get(name)
    if spec is None:
        raise ExperimentError(
            f"unknown arm {name!r} — not registered in "
            f"producers/registry.py::RESEARCH_PRODUCERS. Registered arms: "
            f"{sorted(RESEARCH_PRODUCERS)}"
        )
    return spec


# ── Stage 3: produce this ONE arm's shadow ───────────────────────────────────


def build_manager(bucket: str, s3: Any = None):
    """The ``ArchiveManager`` the arm builders take as ``archive_manager``.

    The REAL one, not a stand-in: ``no_agent_quant`` and ``single_agent_quant``
    call ``load_population()`` and ``load_latest_theses()``, which read the
    SQLite research DB — a minimal ``(s3, bucket)`` duck type (the shape
    ``scripts/backfill_filling_arm_shadows.py`` gets away with, because the
    filling arms touch S3 only) would return an EMPTY population here and the
    arm would build a shadow over nothing while looking healthy.

    ``s3`` is injectable so a test can hand in a moto client; production passes
    None and ArchiveManager builds its own.
    """
    from archive.manager import ArchiveManager

    manager = ArchiveManager(bucket=bucket)
    if s3 is not None:
        manager.s3 = s3
    return manager


def produce_arm(spec, manager, date_str: str, *, write: bool) -> dict:
    """Build ``spec``'s shadow signals for ``date_str``. Returns a record.

    Deliberately NOT routed through ``producers.runner.run_challengers`` — see
    the module docstring. A build failure RAISES: a hand-run experiment that
    silently produced nothing and then printed a blank verdict is worse than
    a traceback.
    """
    if spec.build is None:
        raise ExperimentError(
            f"arm {spec.name!r} carries build=None — its shadow is written by "
            f"its own pipeline, not by this run (see producers/registry.py). "
            f"Drop --produce and grade the shadow it already wrote."
        )
    payload = spec.build(date_str, manager, run_time=date_str, population=None)
    key = SHADOW_KEY.format(arm=spec.name, date=date_str)
    n_enter = sum(
        1
        for v in (payload.get("signals") or {}).values()
        if isinstance(v, dict) and v.get("signal") == "ENTER"
    )
    if write:
        manager.write_shadow_signals_json(spec.name, date_str, date_str, payload)
    return {"date": date_str, "key": key, "n_enter": n_enter, "written": bool(write)}


# ── Stage 4: grade, and render the verdict ───────────────────────────────────


def grade_arm(
    s3: Any,
    bucket: str,
    arm: str,
    date_str: str,
    *,
    top_n: int,
    write: bool,
    closes_panel_loader: Any = None,
) -> dict:
    """Build the producer leaderboard and project the ONE arm's rows out of it.

    Returns ``{"arm", "champion", "board_status", "min_dates_for_inference",
    "horizons": [...]}``. Raises when the board could not be measured or when
    the arm has no row on it — both are conditions a human must see, not a
    quiet empty table.
    """
    res = build_producer_leaderboard(
        s3, bucket, date_str, top_n=top_n, write=write, closes_panel_loader=closes_panel_loader
    )
    status = res.get("status")
    if status != "ok":
        raise ExperimentError(
            f"producer leaderboard is {status!r} on {date_str}: "
            f"{res.get('leaderboard', {}).get('unmeasurable_reason') or res.get('error') or res}"
        )
    board = res["leaderboard"]
    floor = board.get("min_dates_for_inference", slot_spec(SLOT_ID).min_dates_for_inference)

    horizons = []
    for block in board.get("horizons") or []:
        row = next((r for r in block.get("specs") or [] if r.get("name") == arm), None)
        if row is None:
            continue
        horizons.append(
            {
                "horizon_days": block["horizon_days"],
                "horizon_status": block.get("status"),
                "horizon_reason": block.get("reason"),
                "observations_overlap": block.get("observations_overlap"),
                # The arms with NO cohort AT THIS HORIZON. Named because they
                # are what empties the board's all-arm intersection and nulls
                # every row's topn_alpha_vs_champion — a reader seeing
                # `no_common_cohort` must be able to see WHICH arm caused it
                # without walking the artifact. Per horizon, never per board:
                # an arm can be inside the 21d intersection and outside the
                # 252d one purely because the longer horizon has not matured.
                "arms_with_no_cohort": sorted(
                    r["name"]
                    for r in (block.get("specs") or [])
                    if not (r.get("n_dates_scored") or 0)
                ),
                "row": row,
            }
        )
    if not horizons:
        raise ExperimentError(
            f"arm {arm!r} has no row on the producer leaderboard for {date_str}. "
            f"Registered arms are scored on every horizon, so an absent row means "
            f"the arm is not in the scoring set — a retired arm past its trailing "
            f"window (producers.registry.retired_producers), or a registry/board "
            f"disagreement. Arms on this board: "
            f"{sorted({r.get('name') for b in (board.get('horizons') or []) for r in (b.get('specs') or [])})}"
        )
    return {
        "arm": arm,
        "date": date_str,
        "champion": board.get("champion"),
        "board_status": status,
        "board_key": res.get("key"),
        "min_dates_for_inference": floor,
        "primary_metric": slot_spec(SLOT_ID).primary_metric,
        "horizons": horizons,
        "pairwise_vs_champion": pairwise_vs_champion(
            s3, bucket, arm, date_str, top_n=top_n, closes_panel_loader=closes_panel_loader
        ),
    }


def pairwise_vs_champion(
    s3: Any,
    bucket: str,
    arm: str,
    date_str: str,
    *,
    top_n: int,
    closes_panel_loader: Any = None,
) -> dict[int, dict]:
    """The arm-versus-champion paired difference over the dates THOSE TWO arms
    share — per horizon.

    WHY THIS SECOND PASS EXISTS (measured 2026-09-01, not assumed). The board's
    own ``topn_alpha_vs_champion`` is narrowed by
    ``leaderboard_scoring.apply_cohort_intersection`` to
    ``strict_cohort_intersection(ALL arms)`` — the dates on which EVERY
    registered arm scored. That is correct for a leaderboard, where §4 requires
    one common cohort behind a ranked table. It also means a single registered
    arm that has written no shadow at all empties the intersection, and every
    row on the board — including the arm under test — comes back
    ``topn_alpha_vs_champion: null`` with ``comparison_status:
    no_common_cohort``. Reproduced in
    ``tests/test_run_experiment_cli.py::TestBoardIntersectionIsAllArms``.

    So the exact condition this CLI exists to survive — one broken or absent
    sibling arm — takes out the most powerful statistic the design offers, at
    the GRADING stage, well after ``ChallengerShadowGapError`` has been
    side-stepped at the produce stage.

    The answer is neither to fake a cohort for the absent arms nor to widen the
    board's rule. It is to ask the narrower question the CLI was actually
    asked — *this arm versus the champion* — over the intersection of exactly
    those two, using the SAME public scorer the boards use
    (``paired_alpha_vs_champion``, public precisely so a per-arm caller can
    reach it) at the same width and the same HAC lag. Nothing is recomputed by
    hand and no new significance rule is introduced: the figure is reported
    beside its own ``n_dates`` and its own intersection span, labelled
    pairwise, and never substituted for the board's cohort-wide figure.

    Reuses the same private loaders ``build_producer_leaderboard`` uses. That
    coupling is deliberate and read-only — resolving the champion pointer or
    the realized-return join a second way is how two surfaces come to disagree
    about what an arm scored. The closes panel is memoised per invocation
    (``leaderboard_producers._PANEL_CACHE``), so this pass adds no second
    ArcticDB read.
    """
    slot = slot_spec(SLOT_ID)
    horizons = _resolve_horizons(None, slot.horizons_days[0], slot.horizons_days)
    dates = _cohort_dates(s3, bucket, "signals_shadow/", depth=1)
    champion, challengers = _load_producer_specs(s3, bucket, dates, as_of=date_str)
    if champion is None:
        return {}
    if arm == champion.name:
        return {}
    arm_hist = next((c for c in challengers if c.name == arm), None)
    if arm_hist is None:
        raise ExperimentError(
            f"arm {arm!r} is not in the producer scoring set for {date_str} "
            f"(challengers: {sorted(c.name for c in challengers)})"
        )
    realized_by_horizon, _notes, _pop = _resolve_realized_returns_by_horizon(
        bucket, dates, horizons, symbols=None, closes_panel_loader=closes_panel_loader
    )
    spacing = median_cohort_spacing_days(dates)
    out: dict[int, dict] = {}
    for h in horizons:
        realized = realized_by_horizon.get(h) or {}
        inter = strict_cohort_intersection([arm_hist, champion], realized)
        restricted = {d: realized[d] for d in inter if d in realized}
        out[h] = {
            "n_dates": len(inter),
            "first": inter[0] if inter else None,
            "last": inter[-1] if inter else None,
            "metric": (
                paired_alpha_vs_champion(
                    arm_hist, champion, restricted, top_n,
                    overlap_lags=overlap_lags_for(h, spacing),
                )
                if inter
                else None
            ),
        }
    return out


def _fmt_metric(m: Any) -> str:
    """One metric block (``{mean, se, t_stat, n_dates}``) on one line."""
    if not isinstance(m, dict):
        return "        —  (not scored)"
    mean, se, t = m.get("mean"), m.get("se"), m.get("t_stat")
    return (
        f"mean={mean if mean is not None else '—':>10}  "
        f"se={se if se is not None else '—':>10}  "
        f"t={t if t is not None else '—':>8}  "
        f"n={m.get('n_dates', '—')}"
    )


def render_verdict(verdict: dict) -> str:
    """The one-screen human rendering. Every number here is read off the
    leaderboard artifact; nothing is recomputed and no significance rule is
    invented (the floor shown is the slot's own)."""
    floor = verdict["min_dates_for_inference"]
    lines = [
        "",
        "=" * 78,
        f"  ARM        {verdict['arm']}",
        f"  AS OF      {verdict['date']}",
        f"  CHAMPION   {verdict['champion'] or '(none registered — champion-free metrics only)'}",
        f"  SLOT       producer   primary metric: {verdict['primary_metric']}",
        f"  EVIDENCE   floor = {floor} scored cohort dates "
        f"(LEADERBOARD_SLOTS['producer'].min_dates_for_inference)",
        f"  BOARD      {verdict['board_status']}"
        + (f"  -> {verdict['board_key']}" if verdict.get("board_key") else "  (not written — no --write)"),
        "=" * 78,
    ]
    for h in verdict["horizons"]:
        row = h["row"]
        n = row.get("n_dates_scored") or 0
        clears = n >= floor
        lines += [
            "",
            f"  ── {h['horizon_days']} sessions "
            f"[horizon: {h['horizon_status']}{'  — ' + h['horizon_reason'] if h.get('horizon_reason') else ''}]",
            f"     kind={row.get('kind')}  confidence={row.get('confidence')}  "
            f"measurability={row.get('measurability')}",
            f"     n_dates_scored={n}  vs floor {floor}  ->  "
            f"{'CLEARS the evidence floor' if clears else 'BELOW the evidence floor — not a result'}",
        ]
        if row.get("unmeasurable_reason"):
            lines.append(f"     unmeasurable: {row['unmeasurable_reason']}")
        pw = (verdict.get("pairwise_vs_champion") or {}).get(h["horizon_days"])
        lines += [
            f"     alpha vs champion    {_fmt_metric(row.get('topn_alpha_vs_champion'))}"
            f"   [board cohort: {row.get('comparison_status')}]",
        ]
        if pw is not None:
            lines.append(
                f"       pairwise vs champ  {_fmt_metric(pw.get('metric'))}"
                f"   [{pw['n_dates']} shared date(s)"
                + (f": {pw['first']}..{pw['last']}]" if pw["n_dates"] else "]")
            )
        lines += [
            f"     alpha vs benchmark   {_fmt_metric(row.get('topn_alpha_vs_benchmark'))}",
            f"     alpha vs population  {_fmt_metric(row.get('topn_alpha_vs_population'))}",
            f"     realized rank IC     {_fmt_metric(row.get('realized_rank_ic'))}",
        ]
        if h.get("observations_overlap"):
            lines.append(
                "     NOTE observations OVERLAP at this horizon — the t-stat is "
                "HAC-corrected, not iid."
            )
        if not row.get("promotion_eligible", True):
            lines.append(f"     NOT promotion-eligible: {row.get('ineligible_reason')}")
        starved = [a for a in h.get("arms_with_no_cohort") or [] if a != row.get("name")]
        if starved and row.get("comparison_status") == COMPARISON_NO_COMMON_COHORT:
            lines += [
                "     NOTE the board's cross-arm figure is narrowed to the dates "
                "EVERY registered arm scored.",
                f"          These scored none at this horizon, so that "
                f"intersection is empty: {', '.join(starved)}.",
                "          `pairwise vs champ` above is this arm against the "
                "champion over the dates those TWO",
                "          share — the narrower question, honestly labelled. It "
                "is NOT the board figure.",
            ]
    lines += ["", "=" * 78, ""]
    return "\n".join(lines)


# ── Entry point ──────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arm", required=True, help="ProducerSpec name from producers/registry.py")
    ap.add_argument("--date", required=True, help="trading day the verdict is AS OF (YYYY-MM-DD)")
    ap.add_argument(
        "--produce",
        action="store_true",
        help="build this arm's shadow signals first (bypasses the weekly cohort gate)",
    )
    ap.add_argument(
        "--produce-date",
        action="append",
        default=None,
        help="a produce date; repeatable for a range. Defaults to --date.",
    )
    ap.add_argument(
        "--write",
        action="store_true",
        help="actually PUT the shadow signals and the leaderboard artifact. Default is a DRY RUN.",
    )
    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument("--top-n", type=int, default=50, help="count-matching width (slot default 50)")
    ap.add_argument("--json", dest="json_out", help="also write the verdict to this LOCAL path")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.write:
        logger.info("DRY RUN — pass --write to persist. Nothing will be written to S3.\n")

    spec = resolve_arm(args.arm)
    s3 = boto3.client("s3")

    if args.produce:
        manager = build_manager(args.bucket, s3)
        # The arm builders read the population/theses tables off the research
        # DB; without this they see an empty population and build a shadow over
        # nothing (ArchiveManager.load_population returns [] with no db_conn).
        manager.download_db()
        for d in args.produce_date or [args.date]:
            rec = produce_arm(spec, manager, d, write=args.write)
            logger.info(
                "produce  %s %s -> %s  (%d ENTER)%s",
                rec["date"],
                spec.name,
                rec["key"],
                rec["n_enter"],
                "" if rec["written"] else "  [DRY RUN — not written]",
            )
        if not args.write:
            logger.info(
                "\nNOTE: --produce without --write means this arm's shadow for the "
                "date(s) above is NOT on S3, so the verdict below reflects only the "
                "cohort already persisted.\n"
            )

    verdict = grade_arm(
        s3, args.bucket, spec.name, args.date, top_n=args.top_n, write=args.write
    )
    print(render_verdict(verdict))
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(verdict, fh, indent=2, default=str)
        logger.info("verdict json -> %s", args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
