"""Operator script: render the L83 judge cross-validation agreement report.

Usage::

    python scripts/run_judge_cross_validation.py \\
        --bundle ~/Development/alpha-engine-docs/private/judge-crossval-260513 \\
        --out ~/Development/alpha-engine-docs/private/judge-crossval-260513/REPORT.md

The bundle directory is the one produced by ``sample_and_bundle.py`` and
filled in by the operator. The report can also be written to S3 with
``--s3-key`` (path under ``alpha-engine-research`` bucket) for the
durable record that the ROADMAP entry refers to.

This script is operator-driven (quarterly cadence) — not on any cron.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Repo root on sys.path so ``from evals.cross_validation import ...`` resolves
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.cross_validation import run_cross_validation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle", type=Path, required=True,
        help="Path to the rating bundle directory (contains index.json + worksheets/).",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Path to write the markdown report locally (default: BUNDLE/REPORT.md).",
    )
    parser.add_argument(
        "--s3-key", type=str, default=None,
        help="Optional S3 key under alpha-engine-research bucket "
             "(eg 'eval/cross_validation/2026-Q2.md').",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable INFO logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    bundle_dir: Path = args.bundle.expanduser().resolve()
    if not bundle_dir.is_dir():
        parser.error(f"bundle directory not found: {bundle_dir}")
    if not (bundle_dir / "index.json").exists():
        parser.error(f"index.json missing in bundle: {bundle_dir}")

    out_path: Path = (args.out or bundle_dir / "REPORT.md").expanduser().resolve()

    report, agreements = run_cross_validation(bundle_dir)
    out_path.write_text(report)
    print(f"Wrote report → {out_path}")
    print(f"  {len(agreements)} dimension cells")

    if args.s3_key:
        try:
            import boto3
        except ImportError:
            print("WARNING: boto3 not installed — skipping S3 upload", file=sys.stderr)
            return 0
        s3 = boto3.client("s3")
        s3.put_object(
            Bucket="alpha-engine-research",
            Key=args.s3_key,
            Body=report.encode("utf-8"),
            ContentType="text/markdown",
        )
        print(f"Uploaded → s3://alpha-engine-research/{args.s3_key}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
