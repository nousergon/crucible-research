"""
backfill_eval_option_b.py — Migrate the historical eval corpus into
the Option B partition layout.

Companion to ROADMAP P1 line 83 followup. The PR 1 schema bump (2026-
05-08) made ``judge_run_id`` required on every RubricEvalArtifact and
moved the path shape from
``_eval/{date}/{agent_id}/{run_id}.{judge_model}.json`` to
``_eval/{judge_run_date}/{judge_run_id}/
{agent_id}.{run_id}.{judge_model}.json``.

Existing eval files predate the bump — they live at the old path with
schema_version=1 and no ``judge_run_id``. This script:

1. Lists every eval file under the old shape (``_eval/{date}/{agent_id}/...``)
2. Parses each file
3. Adds a synthetic ``judge_run_id`` (one UUID per source date, since
   pre-fix all evals on a date came from a single batch — manual or
   cron — and we don't have batch boundary info to do finer grouping)
4. Bumps ``schema_version`` 1 → 2
5. Writes the migrated artifact at the new path
6. Removes the old object

Pre-write S3 backup of the entire ``_eval/`` corpus is taken before
any writes happen so the migration is reversible.

Usage:
    # local dry-run — shows the migration plan without S3 writes
    python scripts/backfill_eval_option_b.py --dry-run

    # apply
    python scripts/backfill_eval_option_b.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Make ``evals``, ``graph`` etc. importable when invoked from repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import boto3  # noqa: E402
from botocore.exceptions import ClientError  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── Constants ───────────────────────────────────────────────────────────


DEFAULT_BUCKET = "alpha-engine-research"
OLD_EVAL_PREFIX = "decision_artifacts/_eval/"
"""Same prefix for old + new layouts; the difference is the structure
under it: old has agent_id as a directory segment, new has it in
the filename and a judge_run_id directory between date and file."""


# ── Old-shape detection ─────────────────────────────────────────────────


_OLD_PATH_RE = __import__("re").compile(
    r"^decision_artifacts/_eval/(\d{4}-\d{2}-\d{2})/"
    r"([^/]+)/"
    r"([^/]+)\.([^/.]+)\.json$"
)


def _parse_old_key(key: str) -> dict | None:
    """Decompose an old-shape key into its parts, or return None for
    new-shape / non-eval keys.

    Old shape:
      decision_artifacts/_eval/{date}/{agent_id}/{run_id}.{judge_model}.json

    New shape (NOT matched here):
      decision_artifacts/_eval/{date}/{judge_run_id_uuid}/
        {agent_id}.{run_id}.{judge_model}.json
    """
    match = _OLD_PATH_RE.match(key)
    if match is None:
        return None
    date_str, agent_id, run_id, judge_model = match.groups()
    # Disambiguate from new-shape: new-shape's middle segment is a UUID,
    # never contains a colon (agent_ids commonly do — sector_quant:tech),
    # never contains a dot. UUIDs are 36 chars with 4 dashes.
    if _looks_like_uuid(agent_id):
        return None
    return {
        "date": date_str,
        "agent_id": agent_id,
        "run_id": run_id,
        "judge_model": judge_model,
    }


def _looks_like_uuid(s: str) -> bool:
    """Best-effort UUID detection — 8-4-4-4-12 hex, total 36 chars
    with 4 dashes."""
    if len(s) != 36:
        return False
    parts = s.split("-")
    if len(parts) != 5 or [len(p) for p in parts] != [8, 4, 4, 4, 12]:
        return False
    try:
        for p in parts:
            int(p, 16)
    except ValueError:
        return False
    return True


# ── Migration ──────────────────────────────────────────────────────────


def list_old_shape_keys(s3_client: Any, *, bucket: str) -> list[str]:
    """List every eval-artifact S3 key under the old shape."""
    keys: list[str] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=OLD_EVAL_PREFIX):
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            if _parse_old_key(key) is not None:
                keys.append(key)
    return keys


def group_by_date(keys: list[str]) -> dict[str, list[str]]:
    """Group old-shape keys by their date partition. We mint one
    synthetic ``judge_run_id`` per source date — pre-fix all evals on
    a date typically came from one batch and we have no finer grouping
    info available."""
    by_date: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        parsed = _parse_old_key(key)
        if parsed is None:
            continue
        by_date[parsed["date"]].append(key)
    return dict(by_date)


def migrate_one(
    s3_client: Any, *, bucket: str,
    old_key: str, judge_run_id: str, dry_run: bool,
) -> tuple[str, dict] | None:
    """Read old-shape eval, augment with judge_run_id, write to new
    shape, delete old. Returns (new_key, migrated_artifact) on success,
    None on parse / fetch / write error.
    """
    parsed = _parse_old_key(old_key)
    if parsed is None:
        return None
    try:
        body = s3_client.get_object(Bucket=bucket, Key=old_key)["Body"].read()
        artifact = json.loads(body)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[backfill] load failed for s3://%s/%s — skipping: %s",
            bucket, old_key, exc,
        )
        return None

    # Augment + bump schema version. RubricEvalArtifact has
    # extra="forbid" so we can't add new fields without a schema bump
    # — audit info (this is a backfilled artifact, not a natively-
    # emitted one) lives in the pre-write S3 backup snapshot, not on
    # the artifact JSON itself.
    artifact["judge_run_id"] = judge_run_id
    artifact["schema_version"] = 2

    new_key = (
        f"{OLD_EVAL_PREFIX}{parsed['date']}/{judge_run_id}/"
        f"{parsed['agent_id']}.{parsed['run_id']}.{parsed['judge_model']}.json"
    )

    if dry_run:
        return new_key, artifact

    try:
        s3_client.put_object(
            Bucket=bucket, Key=new_key,
            Body=json.dumps(artifact, indent=2).encode("utf-8"),
        )
        s3_client.delete_object(Bucket=bucket, Key=old_key)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "[backfill] write/delete failed for old=%s new=%s: %s",
            old_key, new_key, exc,
        )
        return None
    return new_key, artifact


def backfill_corpus(
    s3_client: Any, *, bucket: str, dry_run: bool,
) -> dict:
    """Scan, group, migrate. Returns summary dict."""
    old_keys = list_old_shape_keys(s3_client, bucket=bucket)
    by_date = group_by_date(old_keys)

    # Mint one synthetic UUID per source date. Plain UUIDs (no
    # "backfill-" prefix) so the new-shape path detection
    # (_looks_like_uuid in this module + the path-shape check anywhere
    # else that distinguishes old vs new) recognizes them
    # unambiguously. Audit info (this is from a migration, not a real
    # batch) lives on the artifact JSON via the migration_source
    # field set at write time.
    judge_run_ids_by_date = {d: str(uuid.uuid4()) for d in by_date}

    summary = {
        "old_shape_keys_total": len(old_keys),
        "dates_processed": len(by_date),
        "judge_run_ids_minted": judge_run_ids_by_date,
        "migrated": 0,
        "failed": 0,
        "dry_run": dry_run,
    }

    for d, keys in sorted(by_date.items()):
        jrid = judge_run_ids_by_date[d]
        logger.info(
            "[backfill] migrating %d evals from date=%s under judge_run_id=%s",
            len(keys), d, jrid,
        )
        for key in keys:
            result = migrate_one(
                s3_client, bucket=bucket, old_key=key,
                judge_run_id=jrid, dry_run=dry_run,
            )
            if result is None:
                summary["failed"] += 1
            else:
                summary["migrated"] += 1

    return summary


# ── CLI entry ───────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bucket", default=DEFAULT_BUCKET)
    p.add_argument(
        "--dry-run", action="store_true",
        help="Show migration plan without writing or deleting.",
    )
    args = p.parse_args(argv)

    s3 = boto3.client("s3")
    run_time = datetime.now(UTC).isoformat()

    if not args.dry_run:
        # Pre-write backup snapshot — copies the entire _eval/ corpus
        # under a backups/_eval-pre-option-b-{ts}/ prefix so the
        # migration is reversible.
        backup_prefix = (
            f"backups/_eval-pre-option-b-{run_time[:19].replace(':', '')}/"
        )
        logger.info(
            "[backfill] pre-write backup: copying _eval/ corpus to "
            "s3://%s/%s",
            args.bucket, backup_prefix,
        )
        paginator = s3.get_paginator("list_objects_v2")
        copied = 0
        for page in paginator.paginate(
            Bucket=args.bucket, Prefix=OLD_EVAL_PREFIX,
        ):
            for obj in page.get("Contents", []) or []:
                src_key = obj["Key"]
                dst_key = backup_prefix + src_key
                try:
                    s3.copy_object(
                        Bucket=args.bucket,
                        CopySource={"Bucket": args.bucket, "Key": src_key},
                        Key=dst_key,
                    )
                    copied += 1
                except ClientError as exc:
                    logger.warning(
                        "[backfill] backup copy failed for %s: %s",
                        src_key, exc,
                    )
        logger.info("[backfill] pre-write backup copied %d files", copied)

    summary = backfill_corpus(s3, bucket=args.bucket, dry_run=args.dry_run)
    print(json.dumps(summary, indent=2))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
