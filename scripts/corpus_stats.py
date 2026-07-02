#!/usr/bin/env python3
"""Distillation SFT-corpus stats reader.

Reads the SFT-lossless capture sinks (research `decision_artifacts/_sft_raw/` +
metron `metron/_sft_raw/`), normalizes the two schema generations, dedups on a
content hash of the model INPUT, segregates by (producer, task, teacher, source),
and emits a compact stats artifact for the console panel + the #1542 kill-gate
trigger.

Pure boto3 + stdlib (no nousergon_lib) so it runs on any surface that can read
the bucket — the records are plain JSON; only the WRITE side needs the lib.

Trigger metric = deduped, single-(dominant)-teacher **quant-calibrator**
(`sector_quant:*`) pairs vs the ~1000 target (EPIC config#1542).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import date, datetime, timedelta, timezone

RESEARCH_PREFIX = "decision_artifacts/_sft_raw/"
METRON_PREFIX = "metron/_sft_raw/"
OUT_PREFIX = "decision_artifacts/distillation/corpus_stats/"
DEFAULT_BUCKET = "alpha-engine-research"

TRIGGER_TASK = "sector_quant"          # the quant sub-scorer calibrator (config#1135 first target)
TRIGGER_TARGET = 1000                  # ~1000 deduped single-teacher pairs (config#1542)


# ---------------------------------------------------------------------------
# Normalization — collapse v1 (top-level fields, no producer) + v2 (meta/producer)
# ---------------------------------------------------------------------------
def normalize(rec: dict, *, source_hint: str) -> dict | None:
    """Return a normalized view or None if the record is not a usable SFT row."""
    meta = rec.get("meta") or {}
    sv = rec.get("schema_version")
    agent_id = rec.get("agent_id") or meta.get("agent_id")
    if agent_id is None and "input_messages" not in rec:
        return None  # not an SFT record
    producer = rec.get("producer")
    if producer is None:
        # v1 legacy or metron-by-path
        producer = "metron_advisor" if source_hint == "metron" else "crucible_research"
    model = rec.get("model") or rec.get("model_name")
    source = meta.get("source") or rec.get("source") or "live"
    ts = rec.get("captured_at") or rec.get("timestamp") or ""
    run_id = rec.get("run_id") or meta.get("run_id") or ""
    task = (agent_id.split(":")[0] if agent_id else "advisor")
    # content hash over the model INPUT (dedup identical teacher inputs)
    inp = rec.get("input_messages")
    basis = json.dumps(inp, sort_keys=True, ensure_ascii=False) if inp is not None \
        else json.dumps([producer, agent_id, run_id, rec.get("call_seq")], ensure_ascii=False)
    chash = hashlib.sha256(basis.encode("utf-8")).hexdigest()
    return {
        "producer": producer, "agent_id": agent_id or "", "task": task,
        "model": model or "unknown", "source": source,
        "schema_version": sv, "run_id": run_id, "ts": ts, "chash": chash,
    }


# ---------------------------------------------------------------------------
# Aggregation (pure)
# ---------------------------------------------------------------------------
def compute_stats(records_by_key, *, generated_date: str) -> dict:
    """records_by_key: iterable of (s3_key_or_path, source_hint, [raw_line, ...])."""
    raw = unparseable = 0
    seen: set[str] = set()
    dupes = 0
    deduped: list[dict] = []
    by_date_dedup: Counter = Counter()
    by_date_raw: Counter = Counter()

    for key, source_hint, lines in records_by_key:
        rec_date = _date_from_key(key)
        for line in lines:
            line = line.strip()
            if not line:
                continue
            raw += 1
            by_date_raw[rec_date] += 1
            try:
                rec = json.loads(line)
            except Exception:
                unparseable += 1
                continue
            n = normalize(rec, source_hint=source_hint)
            if n is None:
                unparseable += 1
                continue
            if n["chash"] in seen:
                dupes += 1
                continue
            seen.add(n["chash"])
            n["date"] = rec_date
            deduped.append(n)
            by_date_dedup[rec_date] += 1

    by_producer = Counter(n["producer"] for n in deduped)
    by_source = Counter(n["source"] for n in deduped)
    by_task = Counter(n["task"] for n in deduped)
    by_teacher = Counter(n["model"] for n in deduped)
    by_schema = Counter(str(n["schema_version"]) for n in deduped)

    # trigger metric: deduped quant-calibrator, single (dominant) teacher
    quant = [n for n in deduped if n["task"] == TRIGGER_TASK]
    quant_by_teacher = Counter(n["model"] for n in quant)
    dominant_teacher, dominant_n = (quant_by_teacher.most_common(1)[0]
                                    if quant_by_teacher else ("none", 0))

    dates_sorted = sorted(d for d in by_date_dedup if d)
    cum = 0
    growth = []
    for d in dates_sorted:
        cum += by_date_dedup[d]
        growth.append({"date": d, "added": by_date_dedup[d], "cumulative": cum})

    return {
        "schema_version": 1,
        "generated_at": _now_iso(),
        "generated_date": generated_date,
        "trigger": {
            "task": TRIGGER_TASK,
            "target_pairs": TRIGGER_TARGET,
            "deduped_single_teacher": dominant_n,
            "dominant_teacher": dominant_teacher,
            "quant_total_all_teachers": len(quant),
            "pct": round(100.0 * dominant_n / TRIGGER_TARGET, 1),
            "crossed": dominant_n >= TRIGGER_TARGET,
            "clock_started": False,
        },
        "totals": {
            "raw_records": raw,
            "unparseable": unparseable,
            "duplicates_dropped": dupes,
            "deduped_pairs": len(deduped),
        },
        "by_producer": dict(by_producer),
        "by_source": dict(by_source),
        "by_task": dict(by_task.most_common()),
        "by_teacher": dict(by_teacher),
        "by_schema_version": dict(by_schema),
        "quant_calibrator_by_teacher": dict(quant_by_teacher),
        "capture": _capture_freshness(dates_sorted),
        "growth": growth,
    }


def _capture_freshness(dates_sorted: list[str]) -> dict:
    if not dates_sorted:
        return {"last_captured_date": None, "captured_dates": [], "missing_saturdays": []}
    first = datetime.strptime(dates_sorted[0], "%Y-%m-%d").date()
    today = _today()
    captured = set(dates_sorted)
    missing = []
    d = first
    while d <= today:
        if d.weekday() == 5 and d.strftime("%Y-%m-%d") not in captured:  # Saturday
            missing.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return {
        "last_captured_date": dates_sorted[-1],
        "captured_dates": dates_sorted,
        "missing_saturdays": missing,
    }


def _date_from_key(key: str) -> str:
    # .../_sft_raw/{YYYY-MM-DD}/{run_id}/{agent}.jsonl  → first date-looking segment
    for part in key.replace("\\", "/").split("/"):
        if len(part) == 10 and part[4] == "-" and part[7] == "-":
            return part
    return "unknown"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _today() -> date:
    return datetime.now(timezone.utc).date()


# ---------------------------------------------------------------------------
# Source adapters
# ---------------------------------------------------------------------------
def iter_local(root: str):
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".jsonl"):
                continue
            p = os.path.join(dirpath, fn)
            hint = "metron" if "metron" in p else "research"
            with open(p, encoding="utf-8", errors="replace") as fh:
                yield p, hint, fh.read().splitlines()


def iter_s3(client, bucket: str):
    paginator = client.get_paginator("list_objects_v2")
    for prefix, hint in ((RESEARCH_PREFIX, "research"), (METRON_PREFIX, "metron")):
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith(".jsonl"):
                    continue
                body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
                yield key, hint, body.decode("utf-8", errors="replace").splitlines()


# ---------------------------------------------------------------------------
# Public entrypoint (importable by the research post-step) + CLI
# ---------------------------------------------------------------------------
def compute_corpus_stats(s3_client, bucket: str = DEFAULT_BUCKET,
                         target_date: str | None = None, write: bool = True) -> dict:
    gen_date = target_date or _today().strftime("%Y-%m-%d")
    stats = compute_stats(iter_s3(s3_client, bucket), generated_date=gen_date)
    if write:
        body = json.dumps(stats, indent=2).encode("utf-8")
        for key in (f"{OUT_PREFIX}{gen_date}.json", f"{OUT_PREFIX}latest.json"):
            s3_client.put_object(Bucket=bucket, Key=key, Body=body,
                                 ContentType="application/json")
        stats["output_key"] = f"{OUT_PREFIX}latest.json"
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute distillation SFT-corpus stats.")
    ap.add_argument("--local", help="Read from a local dir tree instead of S3 (test).")
    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument("--no-write", action="store_true", help="Print, do not write to S3.")
    ap.add_argument("--date", help="generated_date override (YYYY-MM-DD).")
    args = ap.parse_args()

    if args.local:
        stats = compute_stats(iter_local(args.local),
                              generated_date=args.date or _today().strftime("%Y-%m-%d"))
        print(json.dumps(stats, indent=2))
        return
    import boto3
    client = boto3.client("s3")
    stats = compute_corpus_stats(client, args.bucket, args.date, write=not args.no_write)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
