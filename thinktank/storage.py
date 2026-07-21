"""Thin S3 store for the ``thinktank/`` namespace.

Follows the repo convention (``archive/manager.py``): plain boto3 client,
retry on puts, ``None`` on missing keys for gets. All thinktank artifacts go
through this module so the namespace boundary (never write outside
``thinktank/`` + the shared SFT prefix) is auditable in one place.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import boto3

logger = logging.getLogger(__name__)


class ThinktankStore:
    def __init__(self, bucket: str, s3_client: Any | None = None) -> None:
        self.bucket = bucket
        self.s3 = s3_client or boto3.client("s3")

    def get_json(self, key: str) -> Any | None:
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        except self.s3.exceptions.NoSuchKey:
            return None
        except Exception as exc:  # noqa: BLE001
            code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey"):
                return None
            raise
        return json.loads(obj["Body"].read().decode("utf-8"))

    def get_text(self, key: str) -> str | None:
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        except self.s3.exceptions.NoSuchKey:
            return None
        except Exception as exc:  # noqa: BLE001
            code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey"):
                return None
            raise
        return obj["Body"].read().decode("utf-8")

    def list_keys(self, prefix: str) -> list[str]:
        keys: list[str] = []
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            keys.extend(obj["Key"] for obj in page.get("Contents", []))
        return keys

    def put_json(self, key: str, payload: Any) -> None:
        body = json.dumps(payload, indent=2, default=str).encode("utf-8")
        self._put(key, body, "application/json")

    def put_jsonl(self, key: str, rows: list[dict]) -> None:
        body = ("\n".join(json.dumps(r, default=str) for r in rows) + "\n").encode(
            "utf-8"
        )
        self._put(key, body, "application/x-ndjson")

    def _put(self, key: str, body: bytes, content_type: str) -> None:
        last: Exception | None = None
        for attempt in range(3):
            try:
                self.s3.put_object(
                    Bucket=self.bucket, Key=key, Body=body, ContentType=content_type
                )
                return
            except Exception as exc:  # noqa: BLE001 — retried, then re-raised loud
                last = exc
                logger.warning("s3 put %s attempt %d failed: %s", key, attempt + 1, exc)
        raise RuntimeError(f"s3 put failed after retries: s3://{self.bucket}/{key}") from last
