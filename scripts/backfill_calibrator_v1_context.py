"""
backfill_calibrator_v1_context.py — Populate the v12 score_performance
context columns from archived signals.json.

Companion to ROADMAP P0 line ~103. The schema-extension PR (v12) adds
five nullable columns (quant_score, qual_score, conviction,
sector_modifier, market_regime) to score_performance. New rows
written by record_new_buy_scores after the producer wire-up land
populated; existing rows (213 today, 2026-03-04 → 2026-04-24)
have NULLs.

This script reads signals/{date}/signals.json from S3 for every
distinct score_date in score_performance, joins on (symbol,
signal_date), and populates the new columns via idempotent
UPDATE-WHERE-NULL. One-shot operator script — not invoked by SF.

signals.json shape (verified on 2026-04-24 sample):
  top.market_regime          → str        ("bull"/"bear"/"neutral"/...)
  top.sector_modifiers        → dict[str, float]
  top.signals[ticker].quant_score   → int|float
  top.signals[ticker].qual_score    → int|float
  top.signals[ticker].conviction    → "rising"|"stable"|"declining"
  top.signals[ticker].sector        → str

Defensive on missing keys / older formats: rows that can't be matched
stay NULL; the v1 calibrator trainer filters them out as documented in
the line ~103 ROADMAP entry.

Usage:
    # local — pulls research.db from S3, backfills, uploads back
    python scripts/backfill_calibrator_v1_context.py --bucket alpha-engine-research

    # dry-run — show what would be updated without writing
    python scripts/backfill_calibrator_v1_context.py --bucket alpha-engine-research --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

# Make ``archive`` importable when invoked as `python scripts/...` from
# the repo root (matches the convention used by other one-shot scripts
# that need access to the schema-migration helpers).
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


_SIGNAL_KEY_TEMPLATE = "signals/{date}/signals.json"


def fetch_signals_payload(
    s3_client, bucket: str, score_date: str,
) -> dict | None:
    """Return the signals.json dict for ``score_date`` or None.

    None means the object doesn't exist — older score_performance rows
    may pre-date the signals/ archive entirely. Caller treats as
    "leave NULL"; not an error.
    """
    key = _SIGNAL_KEY_TEMPLATE.format(date=score_date)
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "NoSuchKey":
            return None
        raise
    body = obj["Body"].read()
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        logger.warning(
            "[%s] signals.json parse error: %s — treating as missing",
            score_date, exc,
        )
        return None


def extract_context_for_ticker(
    payload: dict, ticker: str,
) -> dict | None:
    """Pull (quant_score, qual_score, conviction, sector_modifier,
    market_regime) for ``ticker`` from a signals.json payload.

    Returns None if the ticker isn't in the signals dict (e.g. tickers
    that scored ENTER on a date before the signals.json archive
    started, or rows from a different ticker source). At least one of
    the five fields must be non-None for the return to be useful;
    otherwise return None so the UPDATE is skipped (preserves the
    SELECT … WHERE NULL semantics — repeated runs don't churn).
    """
    signals = payload.get("signals") or {}
    sig = signals.get(ticker)
    if sig is None:
        return None

    sector = sig.get("sector")
    sector_modifiers = payload.get("sector_modifiers") or {}
    sector_modifier = sector_modifiers.get(sector) if sector else None
    market_regime = payload.get("market_regime")

    out = {
        "quant_score": sig.get("quant_score"),
        "qual_score": sig.get("qual_score"),
        "conviction": sig.get("conviction"),
        "sector_modifier": sector_modifier,
        "market_regime": market_regime,
    }
    if all(v is None for v in out.values()):
        return None
    return out


def backfill_one_row(
    conn: sqlite3.Connection,
    symbol: str,
    score_date: str,
    context: dict,
    dry_run: bool,
) -> bool:
    """Apply UPDATE-WHERE-NULL for the 5 columns. Returns True iff at
    least one column was actually written (skipping rows whose every
    target column is already non-NULL avoids a noisy churn count)."""
    cur = conn.execute(
        "SELECT quant_score, qual_score, conviction, sector_modifier, market_regime "
        "FROM score_performance WHERE symbol = ? AND score_date = ?",
        (symbol, score_date),
    )
    row = cur.fetchone()
    if row is None:
        return False
    cur_quant, cur_qual, cur_conv, cur_sec, cur_regime = row

    updates: list[tuple[str, object]] = []
    if cur_quant is None and context["quant_score"] is not None:
        updates.append(("quant_score", context["quant_score"]))
    if cur_qual is None and context["qual_score"] is not None:
        updates.append(("qual_score", context["qual_score"]))
    if cur_conv is None and context["conviction"] is not None:
        updates.append(("conviction", context["conviction"]))
    if cur_sec is None and context["sector_modifier"] is not None:
        updates.append(("sector_modifier", context["sector_modifier"]))
    if cur_regime is None and context["market_regime"] is not None:
        updates.append(("market_regime", context["market_regime"]))

    if not updates:
        return False

    if dry_run:
        return True

    # `col` is always one of 5 hardcoded literal column names from the `updates`
    # list built above (quant_score/qual_score/conviction/sector_modifier/
    # market_regime) — never context/row-derived; bound values travel via `values`.
    set_clause = ", ".join(f"{col} = ?" for col, _ in updates)
    values = [v for _, v in updates] + [symbol, score_date]
    conn.execute(
        f"UPDATE score_performance SET {set_clause} "  # noqa: S608
        f"WHERE symbol = ? AND score_date = ?",
        values,
    )
    return True


def backfill_all(
    conn: sqlite3.Connection, s3_client, bucket: str, dry_run: bool,
) -> dict:
    """Iterate every (symbol, score_date) pair with at least one NULL
    new-column. Group by score_date so signals.json fetches are 1-per-date."""
    rows = conn.execute(
        """
        SELECT symbol, score_date FROM score_performance
        WHERE quant_score IS NULL
           OR qual_score IS NULL
           OR conviction IS NULL
           OR sector_modifier IS NULL
           OR market_regime IS NULL
        ORDER BY score_date, symbol
        """,
    ).fetchall()
    by_date: dict[str, list[str]] = {}
    for symbol, score_date in rows:
        by_date.setdefault(score_date, []).append(symbol)

    summary = {
        "rows_eligible": len(rows),
        "dates_processed": 0,
        "dates_missing_signals": 0,
        "rows_updated": 0,
        "rows_skipped_no_match": 0,
    }
    for score_date, symbols in by_date.items():
        payload = fetch_signals_payload(s3_client, bucket, score_date)
        if payload is None:
            summary["dates_missing_signals"] += 1
            summary["rows_skipped_no_match"] += len(symbols)
            logger.info(
                "[%s] signals.json absent — %d row(s) stay NULL",
                score_date, len(symbols),
            )
            continue
        summary["dates_processed"] += 1
        date_updated = 0
        for symbol in symbols:
            ctx = extract_context_for_ticker(payload, symbol)
            if ctx is None:
                summary["rows_skipped_no_match"] += 1
                continue
            if backfill_one_row(conn, symbol, score_date, ctx, dry_run):
                summary["rows_updated"] += 1
                date_updated += 1
        logger.info(
            "[%s] processed %d/%d row(s)",
            score_date, date_updated, len(symbols),
        )

    if not dry_run:
        conn.commit()
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bucket", default="alpha-engine-research")
    p.add_argument("--db-key", default="research.db")
    p.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be updated; do not write or upload.",
    )
    args = p.parse_args()

    s3 = boto3.client("s3")
    run_time = datetime.now(UTC).isoformat()

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        local_db = tf.name
    logger.info(
        "Downloading s3://%s/%s -> %s", args.bucket, args.db_key, local_db,
    )
    s3.download_file(args.bucket, args.db_key, local_db)

    conn = sqlite3.connect(local_db)
    try:
        # Apply the v12 migration on the downloaded DB before iterating
        # so this script is order-independent w/r/t the producer PR. If
        # the schema is already v12, the migration is a no-op.
        try:
            from archive.schema import ensure_schema

            ensure_schema(conn)
        except Exception as exc:
            logger.warning(
                "ensure_schema() failed (continuing — assume v12 already "
                "applied): %s", exc,
            )
        summary = backfill_all(conn, s3, args.bucket, args.dry_run)
    finally:
        conn.close()

    logger.info("Summary: %s", json.dumps(summary, indent=2))

    if args.dry_run:
        logger.info("DRY RUN — not uploading.")
        return 0

    if summary["rows_updated"] == 0:
        logger.info("No rows updated — skipping upload.")
        return 0

    backup_key = (
        f"backups/research.db.pre-calibrator-v1-backfill."
        f"{run_time[:19].replace(':','')}.db"
    )
    logger.info(
        "Uploading pre-backfill snapshot to s3://%s/%s",
        args.bucket, backup_key,
    )
    s3.upload_file(local_db, args.bucket, backup_key)

    logger.info(
        "Uploading updated DB to s3://%s/%s", args.bucket, args.db_key,
    )
    s3.upload_file(local_db, args.bucket, args.db_key)

    logger.info("Backfill complete: %d rows updated.", summary["rows_updated"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
