"""
Sync research.db between local filesystem and S3.

Usage:
  python local/sync_db.py pull   # download from S3
  python local/sync_db.py push   # upload to S3
  python local/sync_db.py status # show local vs S3 metadata
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

# `config` reads AWS_REGION/S3_BUCKET from os.environ at import time, so
# load_dotenv() above must run first for a local .env override to take effect.
from config import AWS_REGION, S3_BUCKET  # noqa: E402

_DB_KEY = "research.db"
_LOCAL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "research.db")


def pull():
    s3 = boto3.client("s3", region_name=AWS_REGION)
    try:
        s3.download_file(S3_BUCKET, _DB_KEY, _LOCAL_PATH)
        size = os.path.getsize(_LOCAL_PATH)
        print(f"Pulled research.db → {_LOCAL_PATH} ({size:,} bytes)")
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print("No research.db found in S3 yet.")
        else:
            raise


def push():
    if not os.path.exists(_LOCAL_PATH):
        print(f"Local research.db not found at {_LOCAL_PATH}")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d")
    s3 = boto3.client("s3", region_name=AWS_REGION)

    # Backup existing
    backup_key = f"backups/research_{timestamp}_manual.db"
    try:
        s3.copy_object(
            Bucket=S3_BUCKET,
            CopySource={"Bucket": S3_BUCKET, "Key": _DB_KEY},
            Key=backup_key,
        )
        print(f"Backed up existing S3 DB to {backup_key}")
    except ClientError:
        pass

    s3.upload_file(_LOCAL_PATH, S3_BUCKET, _DB_KEY)
    size = os.path.getsize(_LOCAL_PATH)
    print(f"Pushed {_LOCAL_PATH} → s3://{S3_BUCKET}/{_DB_KEY} ({size:,} bytes)")


def status():
    s3 = boto3.client("s3", region_name=AWS_REGION)

    local_info = "Not found"
    if os.path.exists(_LOCAL_PATH):
        size = os.path.getsize(_LOCAL_PATH)
        mtime = datetime.fromtimestamp(os.path.getmtime(_LOCAL_PATH)).isoformat()
        local_info = f"{size:,} bytes, modified {mtime}"

    s3_info = "Not found"
    try:
        meta = s3.head_object(Bucket=S3_BUCKET, Key=_DB_KEY)
        s3_size = meta["ContentLength"]
        s3_mtime = meta["LastModified"].isoformat()
        s3_info = f"{s3_size:,} bytes, modified {s3_mtime}"
    except ClientError:
        pass

    print(f"Local:  {_LOCAL_PATH} — {local_info}")
    print(f"S3:     s3://{S3_BUCKET}/{_DB_KEY} — {s3_info}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["pull", "push", "status"])
    args = parser.parse_args()
    {"pull": pull, "push": push, "status": status}[args.action]()


if __name__ == "__main__":
    main()
