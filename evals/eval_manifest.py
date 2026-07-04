"""
Capture-date manifest aggregator (Option B PR 2).

Eval artifacts live under ``decision_artifacts/_eval/`` in one of two
layouts (config#793 dual-layout tolerance):

* **Canonical flat** (current writes) —
  ``_eval/{YYMMDDHHMM}_{judged_agent_id}.{run_id}.{judge_model}.json``
  per the ``nousergon_lib.eval_artifacts`` convention. The
  timestamp-encoded ``judge_run_id`` groups one batch's files by shared
  prefix; no date sub-partition.
* **Legacy nested** (pre-config#793, NOT backfilled) —
  ``_eval/{judge_run_date}/{judge_run_id}/
  {judged_agent_id}.{run_id}.{judge_model}.json`` from the 2026-05-08
  Option B partition arc. Months of historical forensic artifacts live
  here; the scanner reads them too so the swap strands nothing.

In both layouts the judged artifact's capture date lives in
``judged_artifact_s3_key`` as a foreign-key field, NOT in the eval path.

Operator queries by capture date ("show me all evals for the 5/9
captures") therefore need an index layer. This module builds that
layer: per-capture-date manifest files at
``decision_artifacts/_eval_by_capture/{capture_date}/manifest.json``
listing every eval that scored an artifact captured that day.

The manifest is eventually-consistent — a daily aggregator scans the
``_eval/`` prefix over a rolling lookback window and rewrites the
manifests it touches. Operators querying within ~24h of a judge run
may miss the most recent evals; that's an acceptable trade for the
institutional separation of write path from index path.

Entry points:

* ``build_manifests(date_window)`` — pure function: scan, group,
  emit. Returns the list of (capture_date, manifest) tuples it
  wrote. Idempotent (same input ⇒ same output; safe to re-run).

* ``main()`` — CLI for operator backfills + scheduled invocation.

Composes with:

- PR 1 schema (judge_run_id required on every RubricEvalArtifact)
- PR 3 corpus backfill (this aggregator runs after the backfill to
  build manifests for migrated evals)
- The rolling-mean Lambda (separate consumer; reads CW metrics,
  not these manifests)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from nousergon_lib.eval_artifacts import (
    EVAL_LATEST_FILENAME as DEFAULT_LATEST_FILENAME,
)

# Make ``evals``, ``graph`` etc. importable when invoked as
# `python evals/eval_manifest.py` from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)


# ── Constants ───────────────────────────────────────────────────────────


DEFAULT_BUCKET = "alpha-engine-research"
DEFAULT_EVAL_PREFIX = "decision_artifacts/_eval/"
DEFAULT_MANIFEST_PREFIX = "decision_artifacts/_eval_by_capture/"
MANIFEST_SCHEMA_VERSION = 1
DEFAULT_LOOKBACK_DAYS = 14
"""Default rolling window the aggregator scans. 14 days covers ~2
Saturday SF firings + every weekday EOD between them — wide enough
to catch any cross-week re-judging."""


# ── Capture-date extraction ─────────────────────────────────────────────


_CAPTURE_DATE_RE = __import__("re").compile(
    r"^decision_artifacts/(\d{4})/(\d{2})/(\d{2})/",
)


def _capture_date_from_s3_key(judged_artifact_s3_key: str | None) -> str | None:
    """Extract ``YYYY-MM-DD`` from a DecisionArtifact S3 key.

    Mirrors the helper in ``evals/judge.py`` (kept local for module
    isolation — manifest builder doesn't import the judge runtime).
    Returns None when the key is missing or doesn't match the
    canonical capture-date prefix.
    """
    if not judged_artifact_s3_key:
        return None
    match = _CAPTURE_DATE_RE.match(judged_artifact_s3_key)
    if match is None:
        return None
    y, m, d = match.groups()
    try:
        date(int(y), int(m), int(d))
    except (ValueError, TypeError):
        return None
    return f"{y}-{m}-{d}"


# ── Eval scan ───────────────────────────────────────────────────────────


_CANONICAL_FLAT_RE = __import__("re").compile(
    r"^(\d{10})_.+\.json$",
)
"""Match the canonical flat eval_artifacts basename (config#793),
i.e. the key tail AFTER the ``_eval/`` prefix:
``{YYMMDDHHMM}_{basename}.json``. The 10-digit timestamp run_id is the
lib's ``new_eval_run_id`` shape; the ``_`` separator is the lib's
multi-file-per-run grouping prefix. Matched against the prefix-stripped
relative key (which must contain no further ``/``)."""


def _list_eval_keys(
    s3_client: Any, *, bucket: str, prefix: str,
    judge_run_dates: list[str],
) -> list[str]:
    """List every eval-artifact S3 key, tolerant of BOTH layouts.

    config#793 swapped new writes to the canonical flat
    ``nousergon_lib.eval_artifacts`` layout
    (``{prefix}{YYMMDDHHMM}_{basename}.json``) from the legacy nested
    Option B layout (``{prefix}{judge_run_date}/{judge_run_id}/
    {basename}.json``). Months of historical artifacts live at the
    legacy layout and are NOT backfilled, so the scanner must read both:

    * **Legacy nested** — scoped to the requested ``judge_run_dates``
      (bounded cost; the date IS the path partition).
    * **Canonical flat** — a single top-level LIST of ``{prefix}`` for
      the flat keys. The flat keys carry no date directory, so we can't
      date-scope the LIST; instead we filter by the timestamp-encoded
      ``judge_run_id`` (``YYMMDDHHMM``) falling within the requested
      window. The flat layout accumulates one entry per (agent, run,
      model) per batch — trivial for S3 LIST even over multi-year
      history (the lib's flat-layout rationale).

    The ``latest.json`` sidecar is excluded — it's an operator-UX
    pointer, not an eval artifact.
    """
    paginator = s3_client.get_paginator("list_objects_v2")
    keys: list[str] = []
    seen: set[str] = set()

    # ── Legacy nested layout — scoped per judge_run_date ──────────────
    for d in judge_run_dates:
        date_prefix = f"{prefix}{d}/"
        for page in paginator.paginate(Bucket=bucket, Prefix=date_prefix):
            for obj in page.get("Contents", []) or []:
                key = obj["Key"]
                if key.endswith(".json") and key not in seen:
                    keys.append(key)
                    seen.add(key)

    # ── Canonical flat layout — single top-level scan, filtered by
    #    the timestamp-encoded judge_run_id within the date window.
    # Build the set of YYMMDD prefixes for the requested window so a
    # flat key is kept iff its run_id's date falls inside it.
    wanted_yymmdd = {
        d.replace("-", "")[2:]  # "2026-05-09" -> "260509"
        for d in judge_run_dates
        if len(d) == 10
    }
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            if key in seen or not key.endswith(".json"):
                continue
            if not key.startswith(prefix):
                continue
            rel = key[len(prefix):]
            # Flat keys live directly under the prefix (no further "/").
            # Anything with a "/" is the nested legacy layout (already
            # collected date-scoped above) or an unrelated sub-tree.
            if "/" in rel:
                continue
            # latest.json sidecar — pointer, not an artifact.
            if rel == DEFAULT_LATEST_FILENAME:
                continue
            m = _CANONICAL_FLAT_RE.match(rel)
            if m is None:
                continue
            run_id = m.group(1)
            if wanted_yymmdd and run_id[:6] not in wanted_yymmdd:
                continue
            keys.append(key)
            seen.add(key)

    return keys


def _load_eval_artifact(
    s3_client: Any, *, bucket: str, key: str,
) -> dict | None:
    """Load + parse one eval artifact from S3.

    Returns None on parse / fetch errors so the aggregator continues
    past corrupt or transient-fail items rather than halting the
    whole run.
    """
    try:
        body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
        return json.loads(body)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[eval_manifest] failed to load eval s3://%s/%s — skipping: %s",
            bucket, key, exc,
        )
        return None


# ── Manifest build ──────────────────────────────────────────────────────


def _make_manifest_entry(eval_artifact: dict, eval_s3_key: str) -> dict:
    """Project the ``RubricEvalArtifact`` JSON down to the manifest's
    per-eval fields. Stays small + stable — operators query manifest,
    drill into the full artifact via eval_s3_key when needed."""
    return {
        "judge_run_id": eval_artifact.get("judge_run_id"),
        "judge_run_date": eval_artifact.get("timestamp", "")[:10] or None,
        "judged_agent_id": eval_artifact.get("judged_agent_id"),
        "judged_run_id": eval_artifact.get("run_id"),
        "judge_model": eval_artifact.get("judge_model"),
        "rubric_id": eval_artifact.get("rubric_id"),
        "rubric_version": eval_artifact.get("rubric_version"),
        "eval_s3_key": eval_s3_key,
        "judged_artifact_s3_key": eval_artifact.get("judged_artifact_s3_key"),
        "judge_skip_reason": eval_artifact.get("judge_skip_reason"),
    }


def build_manifests(
    *,
    s3_client: Any,
    bucket: str = DEFAULT_BUCKET,
    eval_prefix: str = DEFAULT_EVAL_PREFIX,
    manifest_prefix: str = DEFAULT_MANIFEST_PREFIX,
    judge_run_dates: list[str] | None = None,
    today: date | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    write: bool = True,
) -> dict[str, dict]:
    """Scan the eval corpus, group by capture_date, write manifests.

    Returns a dict keyed by capture_date with the full manifest body
    that was written (or would be written, when ``write=False``).

    ``judge_run_dates`` overrides the default rolling-window scan —
    pass an explicit list of dates to bound the scan during backfills.
    Falls back to ``today - lookback_days .. today`` when None.

    Idempotent: re-running with the same input produces byte-identical
    output (sort order pinned). Safe to schedule daily.
    """
    today = today or datetime.now(timezone.utc).date()
    if judge_run_dates is None:
        judge_run_dates = [
            (today - timedelta(days=offset)).isoformat()
            for offset in range(lookback_days + 1)
        ]

    eval_keys = _list_eval_keys(
        s3_client, bucket=bucket, prefix=eval_prefix,
        judge_run_dates=judge_run_dates,
    )
    logger.info(
        "[eval_manifest] scanning %d judge_run_dates, found %d eval files",
        len(judge_run_dates), len(eval_keys),
    )

    by_capture_date: dict[str, list[dict]] = defaultdict(list)
    skipped_unkeyed = 0
    for key in eval_keys:
        artifact = _load_eval_artifact(s3_client, bucket=bucket, key=key)
        if artifact is None:
            continue
        capture_date = _capture_date_from_s3_key(
            artifact.get("judged_artifact_s3_key"),
        )
        if capture_date is None:
            # In-memory / synthetic / replay artifacts without an S3
            # backref can't be indexed by capture_date. They remain
            # discoverable at their judge_run_id directory; just
            # absent from this manifest layer.
            skipped_unkeyed += 1
            continue
        by_capture_date[capture_date].append(_make_manifest_entry(artifact, key))

    if skipped_unkeyed:
        logger.info(
            "[eval_manifest] skipped %d evals without judged_artifact_s3_key",
            skipped_unkeyed,
        )

    manifests: dict[str, dict] = {}
    for capture_date, entries in by_capture_date.items():
        # Sort entries deterministically — pin the canonical order so
        # repeated runs produce byte-identical manifest bodies.
        entries.sort(key=lambda e: (
            e.get("judged_agent_id") or "",
            e.get("judged_run_id") or "",
            e.get("judge_model") or "",
            e.get("eval_s3_key") or "",
        ))
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "capture_date": capture_date,
            "generated_at": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z",
            ),
            "eval_count": len(entries),
            "evals": entries,
        }
        manifests[capture_date] = manifest
        if write:
            manifest_key = f"{manifest_prefix}{capture_date}/manifest.json"
            body = json.dumps(manifest, indent=2, sort_keys=False).encode("utf-8")
            s3_client.put_object(Bucket=bucket, Key=manifest_key, Body=body)
            logger.info(
                "[eval_manifest] wrote s3://%s/%s (%d evals)",
                bucket, manifest_key, len(entries),
            )

    return manifests


# ── CLI entry ───────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bucket", default=DEFAULT_BUCKET)
    p.add_argument("--eval-prefix", default=DEFAULT_EVAL_PREFIX)
    p.add_argument("--manifest-prefix", default=DEFAULT_MANIFEST_PREFIX)
    p.add_argument(
        "--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS,
        help="Rolling scan window in days (default 14).",
    )
    p.add_argument(
        "--judge-run-dates", default=None,
        help="Comma-separated list of YYYY-MM-DD dates to scan. "
             "Overrides --lookback-days when set (operator backfill).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Compute manifests but don't write to S3.",
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    import boto3

    s3 = boto3.client("s3")
    judge_run_dates = (
        [d.strip() for d in args.judge_run_dates.split(",") if d.strip()]
        if args.judge_run_dates
        else None
    )
    manifests = build_manifests(
        s3_client=s3,
        bucket=args.bucket,
        eval_prefix=args.eval_prefix,
        manifest_prefix=args.manifest_prefix,
        judge_run_dates=judge_run_dates,
        lookback_days=args.lookback_days,
        write=not args.dry_run,
    )
    summary = {
        "manifests_built": len(manifests),
        "evals_indexed": sum(m["eval_count"] for m in manifests.values()),
        "capture_dates": sorted(manifests.keys()),
        "dry_run": args.dry_run,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
