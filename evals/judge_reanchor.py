"""Judge re-anchor marker (config#2575 item 7 — MECHANISM ONLY).

ARCH §19 / L4578(a) doctrine: a judge-model swap is a REGIME BREAK, not
a quality regression. Scores before and after are not comparable, and
``EvalRollingMean``'s rolling-4-week-mean + ``evals.control_bands``'
Shewhart/CUSUM baselines must not straddle the change — a baseline that
does trips false alarms on the discontinuity itself, not a real
regression.

This module provides the LOGGING mechanism for that event: a
structured, dated marker written to the SAME system-wide changelog
corpus ``evals.rolling_mean``'s regression auto-emit already uses
(``changelog/entries/{date}/{event_id}.json``, schema_version 1.0.0) —
reused rather than inventing a second changelog surface, so
``root_cause_category``/``event_type``-keyed corpus readers (retro
tooling, the dashboard's changelog tile) already know how to render it.

**config#2575 status: mechanism built and unit-tested, NOT wired to any
promotion code path.** The OpenRouter shadow judge has not been promoted
to authoritative — item 6's perturbation-suite validation gates that,
and even a PASS this pass does not by itself authorize promotion (see
the config#2575 PR description). ``log_judge_reanchor_marker`` is ready
for whoever performs the actual promotion to call at that time; nothing
in this codebase currently calls it in a live path.

**What promotion should do when it happens** (not implemented here —
documented so the mechanism's contract is unambiguous for that future
change):

1. Call :func:`log_judge_reanchor_marker` with the old/new resolved
   model identity and the promotion date, BEFORE flipping any
   consumption switch (escalation routing / RationaleClustering /
   ReplayConcordance / Director).
2. Pass the returned changelog ``event_id``/S3 key to
   ``evals.control_bands.compute_and_emit_control_bands(reset_before=...)``
   on the next control-band run for the promoted judge's combos, so the
   Shewhart/CUSUM baseline restarts clean at the promotion date rather
   than straddling the model swap (mirrors the existing manual
   ``reset_before`` operator workflow that module documents — this is
   the "automatic reset driven off the artifact corpus' judge_resolved_model"
   follow-up its own docstring already flags as not-yet-built).
3. Update ``evals/judge_models.py``'s registry note for the promoted
   ``JudgeModelSpec`` and remove its logical key from
   ``SHADOW_LOGICAL_KEYS``.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from hashlib import sha1
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Same changelog corpus ``evals.rolling_mean``'s regression auto-emit
# writes to (see that module's ``_CHANGELOG_BUCKET`` / ``_CHANGELOG_PREFIX``
# / ``_CHANGELOG_SCHEMA_VERSION``) — duplicated here rather than imported
# since those names are module-private there (leading underscore) and
# re-exporting them would blur which module owns the corpus location;
# both modules writing the identical values is the same low-risk
# duplication convention this codebase already uses for
# ``_CUSTOM_ID_PATTERN`` (``evals/judge.py`` vs ``krepis/judge.py``) — a
# values-drift here is caught immediately (both write to the same S3
# location; a typo'd bucket/prefix just means entries land somewhere
# unexpected, not silently corrupt existing ones).
_CHANGELOG_BUCKET = os.environ.get("CHANGELOG_BUCKET", "alpha-engine-research")
_CHANGELOG_PREFIX = "changelog/entries"
_CHANGELOG_SCHEMA_VERSION = "1.0.0"

EVENT_TYPE_JUDGE_REANCHOR = "judge_reanchor"


def log_judge_reanchor_marker(
    *,
    logical_key: str,
    old_resolved_model: Optional[str],
    new_resolved_model: str,
    reason: str,
    s3_client: Any = None,
    now: Optional[datetime] = None,
) -> Optional[str]:
    """Write one ``judge_reanchor`` changelog entry marking a judge-model
    regime break for ``logical_key``.

    Call this at the moment ``request_model`` is deliberately repinned
    for a ``JudgeModelSpec`` (a registry edit in
    ``evals/judge_models.py``) OR at promotion time when a shadow judge
    tier becomes authoritative — either event invalidates rolling-mean /
    control-band history for the affected combos per ARCH §19.

    ``old_resolved_model`` is ``None`` for a judge tier's FIRST
    authoritative anchor (nothing to break from — e.g. the OpenRouter
    shadow tier's eventual promotion, where "old" is "shadow-only, no
    prior authoritative series exists").

    Best-effort like the sibling regression auto-emit
    (``evals.rolling_mean._emit_regression_entry``): any write failure
    logs WARNING and returns ``None`` rather than raising — a changelog
    write failing must never block the actual re-pin/promotion it is
    documenting. Returns the S3 key on success.
    """
    try:
        ts = now or datetime.now(timezone.utc)
        ts_utc = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        entry_date = ts.strftime("%Y-%m-%d")
        ts_id = ts_utc.replace(":", "-").rstrip("Z")
        actor = "alpha-engine-research-judge-reanchor"

        digest_input = (
            f"{logical_key}|{old_resolved_model}|{new_resolved_model}|{ts_utc}"
        ).encode()
        event_hash = sha1(digest_input).hexdigest()[:7]
        event_id = f"{ts_id}_{actor}_{event_hash}"

        summary = (
            f"Judge re-anchor: {logical_key} "
            f"{old_resolved_model or '<none — first anchor>'} → {new_resolved_model}"
        )[:240]
        description = (
            f"Judge logical_key: {logical_key}\n"
            f"Old resolved model: {old_resolved_model or '<none — first anchor>'}\n"
            f"New resolved model: {new_resolved_model}\n"
            f"Reason: {reason}\n"
            f"Regime-break doctrine: ARCH §19 / L4578(a) — rolling-mean "
            f"(evals/rolling_mean.py) and control-band "
            f"(evals/control_bands.py) baselines for every "
            f"(judged_agent_id, criterion, judge_model={logical_key}) combo "
            f"must be reset via compute_and_emit_control_bands(reset_before="
            f"{ts_utc!r}) on the next control-band run, or they will alarm "
            f"on this discontinuity rather than a real regression.\n"
            f"Emitted by: alpha-engine-research evals/judge_reanchor.py"
        )

        entry = {
            "schema_version": _CHANGELOG_SCHEMA_VERSION,
            "event_id": event_id,
            "ts_utc": ts_utc,
            "event_type": EVENT_TYPE_JUDGE_REANCHOR,
            "severity": "info",
            "subsystem": "eval",
            "root_cause_category": "judge_model_upgrade",
            "resolution_type": None,
            "started_at": ts_utc,
            "detected_at": ts_utc,
            "resolved_at": ts_utc,
            "verified_at": None,
            "summary": summary,
            "description": description,
            "resolution_notes": None,
            "actor": actor,
            "machine": "research:evals/judge_reanchor.py",
            "source": "judge-reanchor-marker",
            "auto_emitted": False,
            "git_refs": [],
            "prompt_version": None,
            "run_id": entry_date,
            "eval_run_ref": None,
            "judge_reanchor": {
                "logical_key": logical_key,
                "old_resolved_model": old_resolved_model,
                "new_resolved_model": new_resolved_model,
                "reason": reason,
            },
        }
        key = f"{_CHANGELOG_PREFIX}/{entry_date}/{event_id}.json"

        if s3_client is None:
            import boto3

            s3_client = boto3.client("s3")

        s3_client.put_object(
            Bucket=_CHANGELOG_BUCKET,
            Key=key,
            Body=json.dumps(entry).encode("utf-8"),
            ContentType="application/json",
        )
        logger.info(
            "[judge_reanchor] marker written: s3://%s/%s logical_key=%s "
            "%s -> %s",
            _CHANGELOG_BUCKET, key, logical_key,
            old_resolved_model, new_resolved_model,
        )
        return key
    except Exception as e:  # noqa: BLE001 — best-effort, never block the re-pin
        logger.warning(
            "[judge_reanchor] marker write failed (best-effort, swallowed): %s",
            e,
        )
        return None
