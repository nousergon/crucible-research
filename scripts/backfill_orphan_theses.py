"""
backfill_orphan_theses.py — One-shot backfill for the population/investment_thesis sync bug.

Until the 2026-04-25 atomic-thesis-write fix, archive_writer in research_graph.py
saved population without writing the corresponding investment_thesis row.
Result: every population member added since 2026-03-16 (last clean batch) is
an "orphan" — held in the executor, no thesis in research.db.

The 2026-04-22 hard-fail in score_aggregator (PR #42) catches this loudly when
a held ticker has a material trigger and the held-stock update path produces
an unscoreable record. Symptom: Saturday SF Research Lambda fails with
"thesis_update for X missing final_score AND both sub-scores".

This script seeds an investment_thesis row for every population member that
lacks one, using the long_term_score from population as a placeholder for
final_score. Sub-scores are intentionally left NULL — the placeholder records
exist solely so that prior_thesis lookup succeeds; final_score carries the
held-stock thesis update path.

Usage:
    # local — pulls research.db from S3, backfills, uploads back
    python scripts/backfill_orphan_theses.py --bucket alpha-engine-research

    # dry-run — show what would be inserted without writing
    python scripts/backfill_orphan_theses.py --bucket alpha-engine-research --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import tempfile
from datetime import UTC, datetime

import boto3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def find_orphans(conn: sqlite3.Connection) -> list[dict]:
    """Return population rows that lack a corresponding investment_thesis."""
    rows = conn.execute(
        """
        SELECT p.symbol, p.sector, p.long_term_score, p.long_term_rating,
               p.conviction, p.price_target_upside, p.thesis_summary,
               p.entry_date
        FROM population p
        LEFT JOIN (
            SELECT MAX(id) AS id, symbol FROM investment_thesis GROUP BY symbol
        ) it ON p.symbol = it.symbol
        WHERE it.id IS NULL
        ORDER BY p.entry_date, p.symbol
        """
    ).fetchall()
    return [
        {
            "symbol": r[0], "sector": r[1], "long_term_score": r[2],
            "long_term_rating": r[3], "conviction": r[4],
            "price_target_upside": r[5], "thesis_summary": r[6],
            "entry_date": r[7],
        }
        for r in rows
    ]


def insert_placeholder(conn: sqlite3.Connection, orphan: dict, run_time: str) -> None:
    """Write a placeholder investment_thesis from population data.

    `final_score` (= column `score`) is set to long_term_score so the
    held-stock update path's prior_thesis carries a usable final_score.
    Sub-scores stay NULL — they're not consulted unless final_score is
    missing (score_aggregator's recompute path), which won't happen here.
    """
    date = orphan["entry_date"] or run_time[:10]
    conn.execute(
        """
        INSERT INTO investment_thesis
            (symbol, date, run_time, rating, score,
             quant_score, qual_score,
             technical_score, news_score, research_score, macro_modifier,
             thesis_summary, conviction, signal,
             price_target_upside, stale_days, consistency_flag)
        VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, 1.0,
                ?, ?, ?, ?, 0, 0)
        """,
        (
            orphan["symbol"], date, run_time,
            orphan["long_term_rating"] or "HOLD",
            orphan["long_term_score"] or 50.0,
            orphan["thesis_summary"] or "",
            orphan["conviction"] or "stable",
            orphan["long_term_rating"] or "HOLD",
            orphan["price_target_upside"],
        ),
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", default="alpha-engine-research")
    p.add_argument("--db-key", default="research.db")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be inserted; do not write or upload.")
    args = p.parse_args()

    s3 = boto3.client("s3")
    run_time = datetime.now(UTC).isoformat()

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        local_db = tf.name
    logger.info("Downloading s3://%s/%s -> %s", args.bucket, args.db_key, local_db)
    s3.download_file(args.bucket, args.db_key, local_db)

    conn = sqlite3.connect(local_db)
    try:
        orphans = find_orphans(conn)
        if not orphans:
            logger.info("No orphans found — population and investment_thesis are in sync.")
            return 0

        logger.info("Found %d orphan population members:", len(orphans))
        for o in orphans:
            logger.info(
                "  %-6s entry=%s rating=%s score=%.2f sector=%s",
                o["symbol"], o["entry_date"], o["long_term_rating"],
                o["long_term_score"] or 0.0, o["sector"],
            )

        if args.dry_run:
            logger.info("DRY RUN — not writing.")
            return 0

        # Pre-backfill snapshot for rollback
        backup_key = f"backups/research.db.pre-orphan-backfill.{run_time[:19].replace(':','')}.db"
        logger.info("Uploading pre-backfill snapshot to s3://%s/%s",
                    args.bucket, backup_key)
        s3.upload_file(local_db, args.bucket, backup_key)

        for o in orphans:
            insert_placeholder(conn, o, run_time)
        conn.commit()
        logger.info("Inserted %d placeholder thesis rows.", len(orphans))

        # Verify
        post_orphans = find_orphans(conn)
        if post_orphans:
            logger.error("Post-backfill verification failed: %d orphans remain",
                         len(post_orphans))
            return 1
        logger.info("Verification passed: 0 orphans remain.")
    finally:
        conn.close()

    logger.info("Uploading patched research.db to s3://%s/%s",
                args.bucket, args.db_key)
    s3.upload_file(local_db, args.bucket, args.db_key)
    logger.info("Backfill complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
