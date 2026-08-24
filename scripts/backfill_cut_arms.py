#!/usr/bin/env python3
"""Reconstruct the weight-vector challenger arms across the archived factor-profile
history (alpha-engine-config-I8262).

Why. Three of the universe-cut slot's five arms were first emitted in the week
they were written, so their first read at a matured 126-session horizon lands in
2027-02 — later than the champion's own, for arms that are pure deterministic
functions of an input S3 has retained for every cycle since 2026-05-21.
``momzero`` and ``hard3`` both read ``factors/profiles/{date}/by_ticker.json``
and a weight vector; nothing else. Their historical picks are computable today.

What it does NOT do, and will refuse to:

* **It never reconstructs the champion.** ``attractiveness_top_N`` is archived as
  written and is used as written. Recomputing it from today's factor store would
  restate the fundamentals it ranked on and inject look-ahead into the one arm
  whose history is real. ``merge_backfilled_cuts`` refuses any cut name outside
  ``BACKFILLABLE_ARM_PREFIXES``, and ``assert_backfill_preserved_live_cuts``
  re-checks the assembled artifact rather than trusting that refusal.
* **It never overwrites a cut that already exists.** A contemporaneous write is
  the record of what the run served.
* **It never writes ``universe_membership/latest.json``** — that is the pointer
  the predictor resolves its live universe from — and never the immutable
  ``runs/{stamp}.json`` copies, which record live invocations only.
* **It never reconstructs ``mom121``.** Its 12-1 profiles are a separate
  artifact whose first snapshot is 2026-08-18; there is nothing earlier to
  reconstruct from, and reconstructing it from the champion's profiles would
  silently emit the champion. (Contrary to I8262's second finding, those shadow
  profiles ARE written — to ``factors/profiles_shadow/mom121/`` — and only the
  membership ``source`` string was wrong. That is fixed in the same PR, derived
  from ``factor_scoring.CHALLENGER_PROFILE_PREFIX`` so the two cannot disagree
  again.)

Where the cuts land. Into each date's existing
``universe_membership/{date}/membership.json::cuts``, added alongside the cuts
already there, each stamped ``backfilled_from`` plus a
``vendor_fundamentals_exposure`` block. Not a parallel prefix: the only reader
of the slot's cuts is ``leaderboard_producers._load_cut_specs``, which resolves
``universe_membership/{date}/membership.json`` and nothing else — a parallel
prefix would need a change in that consumer, which this work is scoped out of,
and an arm written somewhere no board reads is the registered-but-unscored
rumour champion-challenger-policy.md §3 warns about.

Contamination. Recorded per arm and per date, never assumed
(``scoring.universe_membership.vendor_fundamentals_exposure``): the fraction of
the arm's OWN weight vector sitting on the three vendor-fundamental pillars that
were placeholder-saturated before 2026-08-20 (alpha-engine-config-I8255).
``momzero`` 0.6, ``hard3`` 0.0. So a pre-repair ``momzero`` cut is exactly
reproducible and is a measurement of the defect; a pre-repair ``hard3`` cut rests
on pillars that carried real spread throughout. What ``hard3`` cannot escape is
the other half of the same fact — the champion was a three-pillar ranking BY
ACCIDENT on those dates, so a ``hard3`` cut may resolve to the champion's exact
membership. That is measured per date as ``champion_overlap`` and an exact
collision is REFUSED, not emitted.

Usage::

    python scripts/backfill_cut_arms.py                      # DRY RUN (default)
    python scripts/backfill_cut_arms.py --report out.json    # dry run + local report
    python scripts/backfill_cut_arms.py --write              # write, IN-REGION only

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
from datetime import UTC, datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boto3  # noqa: E402

from scoring.universe_membership import (  # noqa: E402
    BACKFILLABLE_ARM_PREFIXES,
    UniverseMembershipError,
    assert_backfill_preserved_live_cuts,
    assert_cut_invariants,
    build_backfilled_arm_cuts,
    merge_backfilled_cuts,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backfill_cut_arms")

_BUCKET = os.environ.get("S3_BUCKET", "alpha-engine-research")
_PROFILES = "factors/profiles/{date}/by_ticker.json"
_MEMBERSHIP = "universe_membership/{date}/membership.json"
_REPORT_KEY = "research/cut_backfill/{stamp}.json"


def _profile_dates(s3) -> list[str]:
    """Dated subdirs under ``factors/profiles/`` (YYYY-MM-DD), chronological."""
    dates: set[str] = set()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=_BUCKET, Prefix="factors/profiles/", Delimiter="/"):
        for cp in page.get("CommonPrefixes") or []:
            token = cp["Prefix"].rstrip("/").rsplit("/", 1)[-1]
            if len(token) == 10 and token[4] == "-" and token[7] == "-":
                dates.add(token)
    return sorted(dates)


def _get_json(s3, key: str) -> dict | None:
    try:
        return json.loads(s3.get_object(Bucket=_BUCKET, Key=key)["Body"].read())
    except Exception as exc:  # noqa: BLE001 — absence is a reported outcome, not a crash
        logger.debug("get_object(%s): %s", key, exc)
        return None


def backfill_date(s3, date: str, *, reconstructed_at: str) -> dict:
    """One date's outcome — always a record, never a silent skip.

    ``status`` is one of ``reconstructed`` / ``nothing_to_add`` / ``skipped``,
    and a skip always carries ``reason``. champion-challenger-policy.md §3: a
    cycle an arm produced nothing for is a recorded miss.
    """
    out: dict = {"date": date, "status": "skipped", "added": {}, "refused": {}}
    membership = _get_json(s3, _MEMBERSHIP.format(date=date))
    if not membership:
        out["reason"] = "no archived membership artifact — nothing to add the arms to"
        return out
    population = sorted(membership.get("ranks") or {})
    if not population:
        out["reason"] = (
            "the archived artifact carries no `ranks` table, so the champion's own "
            "ranked population is unknown — ranking the raw profiles instead would "
            "score the arm over a different population than the champion (I7844, §4)"
        )
        return out
    profiles = _get_json(s3, _PROFILES.format(date=date))
    if not profiles:
        out["reason"] = "no factors/profiles/{date}/by_ticker.json — the arm's only input is absent"
        return out

    try:
        cuts, refused = build_backfilled_arm_cuts(
            date,
            profiles,
            population=population,
            champion_cuts=membership.get("cuts") or {},
            reconstructed_at=reconstructed_at,
        )
    except UniverseMembershipError as exc:
        out["reason"] = f"not reconstructable: {exc}"
        return out

    merged, merge_refused = merge_backfilled_cuts(membership, cuts)
    refused = {**refused, **merge_refused}
    added = sorted(set(merged.get("cuts", {})) - set(membership.get("cuts") or {}))

    # The guards run on the ASSEMBLED artifact, in the order that matters: the
    # preservation guard first, because a reconstruction that damaged the live
    # record is a different and worse failure than one that emits a bad arm.
    assert_backfill_preserved_live_cuts(membership, merged, date)
    assert_cut_invariants(merged, date)

    out["population"] = len(population)
    out["added"] = {
        name: {
            "size": merged["cuts"][name]["size"],
            "champion_overlap": merged["cuts"][name].get("champion_overlap"),
            "contaminated": merged["cuts"][name]["vendor_fundamentals_exposure"]["contaminated"],
            "placeholder_weight": merged["cuts"][name]["vendor_fundamentals_exposure"][
                "weight_fraction_on_placeholder_pillars"
            ],
        }
        for name in added
    }
    out["refused"] = refused
    out["status"] = "reconstructed" if added else "nothing_to_add"
    out["_merged"] = merged
    return out


def _summarize(results: list[dict]) -> dict:
    """The per-arm contamination record I8262's Closes-when asks for.

    Built from what was actually emitted, not from the arms' definitions — an
    arm refused on every pre-repair date has no contaminated history however
    exposed its weight vector is, and stating the exposure alone would claim
    otherwise.
    """
    per_arm: dict[str, dict] = {}
    for r in results:
        for name, rec in (r.get("added") or {}).items():
            a = per_arm.setdefault(
                name,
                {
                    "dates": [],
                    "contaminated_dates": [],
                    "weight_fraction_on_placeholder_pillars": rec["placeholder_weight"],
                    "max_champion_overlap": 0.0,
                    "high_overlap_dates": [],
                },
            )
            a["dates"].append(r["date"])
            if rec["contaminated"]:
                a["contaminated_dates"].append(r["date"])
            ov = rec.get("champion_overlap") or 0.0
            a["max_champion_overlap"] = max(a["max_champion_overlap"], ov)
            if ov >= 0.9:
                a["high_overlap_dates"].append([r["date"], ov])
    for a in per_arm.values():
        a["n_dates"] = len(a["dates"])
        a["n_contaminated"] = len(a["contaminated_dates"])
        a["contaminated"] = bool(a["contaminated_dates"])
    refusals: dict[str, int] = {}
    for r in results:
        for name in (r.get("refused") or {}):
            refusals[name] = refusals.get(name, 0) + 1
    return {
        "per_arm": per_arm,
        "refusals_by_arm": refusals,
        "dates_considered": len(results),
        "dates_reconstructed": sum(1 for r in results if r["status"] == "reconstructed"),
        "backfillable_arms": list(BACKFILLABLE_ARM_PREFIXES),
        "champion_reconstructed": False,
        "champion_note": (
            "the champion's archived picks are used as written and are never "
            "recomputed — see BACKFILLABLE_ARM_PREFIXES"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--write",
        action="store_true",
        help="actually write the merged membership artifacts. Default is a DRY RUN.",
    )
    ap.add_argument("--report", help="write the run report to this LOCAL path (works in dry run too)")
    ap.add_argument("--since", help="only consider profile dates >= this ISO date")
    args = ap.parse_args()

    s3 = boto3.client("s3")
    reconstructed_at = datetime.now(UTC).isoformat(timespec="seconds")
    dates = [d for d in _profile_dates(s3) if not args.since or d >= args.since]
    logger.info("factor-profile dates: %d (%s … %s)", len(dates), dates[0] if dates else "-", dates[-1] if dates else "-")

    results: list[dict] = []
    written = 0
    for date in dates:
        r = backfill_date(s3, date, reconstructed_at=reconstructed_at)
        merged = r.pop("_merged", None)
        results.append(r)
        if r["status"] == "reconstructed":
            logger.info("  %s: %s | refused %s", date, r["added"], list(r["refused"]) or "none")
            if args.write:
                s3.put_object(
                    Bucket=_BUCKET,
                    Key=_MEMBERSHIP.format(date=date),
                    Body=json.dumps(merged, separators=(",", ":"), default=str).encode("utf-8"),
                    ContentType="application/json",
                )
                written += 1
        else:
            logger.info("  %s: %s — %s", date, r["status"], r.get("reason") or list(r["refused"]) or "")

    summary = _summarize(results)
    report = {
        "tracker": "alpha-engine-config-I8262",
        "reconstructed_at": reconstructed_at,
        "dry_run": not args.write,
        "written": written,
        "summary": summary,
        "per_date": results,
    }
    logger.info("summary: %s", json.dumps(summary["per_arm"], indent=2, default=str))
    logger.info(
        "done — dates=%d reconstructed=%d written=%d%s",
        len(dates),
        summary["dates_reconstructed"],
        written,
        " (DRY RUN — nothing written)" if not args.write else "",
    )
    if args.report:
        with open(args.report, "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        logger.info("report → %s", args.report)
    if args.write:
        s3.put_object(
            Bucket=_BUCKET,
            Key=_REPORT_KEY.format(stamp=reconstructed_at.replace(":", "").replace("-", "")),
            Body=json.dumps(report, separators=(",", ":"), default=str).encode("utf-8"),
            ContentType="application/json",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
