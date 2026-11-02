"""Declared per-ticker holds: names the scanner must not evaluate this cycle.

A hold removes a ticker from the scanner universe in
``scanner_orchestrator.build_candidates_artifact``, before ``run_quant_filter``
runs. Every downstream set is derived from that universe: the eval log, the
universe board's rows, every attractiveness and tech_score rank table
(``universe_membership._restrict_to_scanned_universe``), and so every cut the
champion pointer can serve. A held name therefore cannot reach a pick through
any arm. The name stays in the S&P constituents and in every data store; only
its candidacy is suspended.

A hold is for a name whose INPUTS are known to be wrong, not for a view on the
name. The first entry is Corteva (CTVA), whose price history carries an
unadjusted 2026-10-01 spin-off break (77.65 -> 12.57) until the declared
adjustment in nousergon-data lands (alpha-engine-config-I11806). Until then its
return, momentum and ATR features read a false -84% day, and its P/E reads 1.75
because the post-spin price is divided by the pre-spin company's earnings.

Each hold carries an absolute ``valid_through`` date, so a forgotten entry
expires instead of excluding the name for ever, plus a ``why`` and the issue
that owns removing it. Entries change by PR only.

MACHINE-WRITTEN holds (alpha-engine-config-I13044) are the automatic form of
the CTVA entry. When nousergon-data's daily append writes a bar whose ratio to
the prior stored close is split-like and no registered corporate action
explains it, it pages at ERROR and opens
``s3://{bucket}/data_quality/ticker_holds/{TICKER}.json`` with
``status: active``. That producer also clears the hold, by evidence once the
stored series no longer prints the break, or by an operator with a reason.
``machine_holds`` reads the active ones; they carry no ``valid_through``
(``None``: held until cleared). A declared hold for the same ticker wins,
since it is a person's dated decision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any

MACHINE_HOLDS_PREFIX = "data_quality/ticker_holds/"


@dataclass(frozen=True)
class TickerHold:
    valid_from: str  # YYYY-MM-DD, inclusive
    valid_through: str | None  # YYYY-MM-DD, inclusive; None = until cleared (machine holds only)
    why: str
    issue: str
    source: str = "declared"


DECLARED_HOLDS: dict[str, TickerHold] = {
    "CTVA": TickerHold(
        valid_from="2026-10-01",
        # Extended 2026-10-14 (Brian's ruling #10): held until Corteva's first
        # post-spin earnings. Corteva announced on 2026-10-06 that its Q3 2026
        # results are released after the market close on Wednesday
        # 2026-11-11, so the hold covers run dates through 11-11 and the first
        # run that can see those results (11-12) scans the name again.
        valid_through="2026-11-11",
        why=(
            "2026-10-01 spin-off of VYLR (1:1) left CTVA's stored price history "
            "unadjusted (77.65 -> 12.57, a false -84% day) and its P/E at 1.75 "
            "(post-spin price over pre-spin earnings). The price restatement "
            "(CRSP factor 0.1555) is nousergon-data#2032; the P/E stays wrong "
            "until the first post-spin earnings (Q3 2026, released after the "
            "close on 2026-11-11), so the hold runs through that date"
        ),
        issue="alpha-engine-config-I11806",
    ),
}


def active_holds(run_date: str, holds: dict[str, TickerHold] | None = None) -> dict[str, TickerHold]:
    """The holds in force on ``run_date`` (both bounds inclusive)."""
    d = date.fromisoformat(str(run_date)[:10])
    table = DECLARED_HOLDS if holds is None else holds
    return {
        ticker: hold
        for ticker, hold in table.items()
        if date.fromisoformat(hold.valid_from) <= d <= date.fromisoformat(hold.valid_through)
    }


def machine_holds(s3_client: Any, bucket: str, run_date: str) -> dict[str, TickerHold]:
    """Active machine-written holds whose break is on or before ``run_date``.

    Raises on any read or parse failure: a hold is a correctness guard on
    picks, so a scanner that cannot read them must not scan as if there were
    none.
    """
    d = date.fromisoformat(str(run_date)[:10])
    out: dict[str, TickerHold] = {}
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=MACHINE_HOLDS_PREFIX):
        for item in page.get("Contents", []) or []:
            key = item["Key"]
            if not key.endswith(".json"):
                continue
            body = json.loads(s3_client.get_object(Bucket=bucket, Key=key)["Body"].read())
            if body.get("status") != "active":
                continue
            if date.fromisoformat(body["break_date"]) > d:
                continue
            out[body["ticker"]] = TickerHold(
                valid_from=body["break_date"],
                valid_through=None,
                why=body["why"],
                issue=body.get("issue", "alpha-engine-config-I13044"),
                source=f"s3://{bucket}/{key}",
            )
    return out
