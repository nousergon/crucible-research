"""experiment_record.py — per-challenger-arm ``experiment_record.v1`` (Phase C,
alpha-engine-config#3077).

Each challenger producer (``producers.registry.challenger_producers()``) is a
manifest-shaped experiment arm: a single, deterministic Slot-R (research)
implementation (``ProducerSpec.build``), run against the SAME scanner
candidate set as the champion. This module builds one ``experiment_record.v1``
per challenger PER RUN and writes it to
``experiments/{spec.name}/records/{run_date}.json`` (+ ``.../latest.json``,
S3 key uses ``spec.name`` verbatim — matches the existing
``signals_shadow/{producer_name}/`` prefix convention). The payload's own
``experiment_id`` field is ``spec.name`` with underscores replaced by hyphens
(``no_agent_quant`` -> ``no-agent-quant``) since the schema's
``experiment_id`` pattern forbids underscores — see :func:`experiment_id_for`.

FAIL-SOFT BY DESIGN — this is deliberately isolated from
``producers.runner.run_challengers``'s existing FAIL-HARD shadow-signal-write
doctrine (config#1683 / ``ChallengerShadowGapError``). The shadow signals.json
write is the load-bearing experiment deliverable; a record-emission bug must
NEVER touch that gap logic or turn a healthy shadow write into a run failure.
Errors here log + fire the same ``publish_observe_alert`` LOUD-but-non-fatal
path ``run_challengers`` already uses for shadow-write gaps — visible, but
never raised.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

EXPERIMENT_RECORD_PREFIX = "experiments"
LATEST_EXPERIMENT_RECORD_FILENAME = "latest.json"


def _local_git_sha() -> str | None:
    """Best-effort local ``git rev-parse HEAD`` (dev/test fallback when
    ``ALPHA_ENGINE_CODE_SHA`` is unset). Never raises."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        sha = out.stdout.strip()
        return sha or None
    except Exception:  # noqa: BLE001 — best-effort provenance, never fatal
        return None


def _resolve_code_sha() -> str | None:
    """This repo's own resolved SHA: the baked ``ALPHA_ENGINE_CODE_SHA`` env
    var first (Lambda image, GHA ``--build-arg GIT_SHA`` / manual deploy.sh
    stamp — same convention ``graph/research_graph.py``'s decision-capture
    provenance already reads), local ``git rev-parse HEAD`` fallback
    (dev/test)."""
    return os.environ.get("ALPHA_ENGINE_CODE_SHA") or _local_git_sha()


def _manifest_hash(slots: list[dict]) -> str:
    canonical = json.dumps(slots, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def experiment_id_for(producer_name: str) -> str:
    """Map a ``ProducerSpec.name`` (e.g. ``no_agent_quant``) to a conformant
    ``experiment_record.v1`` ``experiment_id``.

    The schema's ``experiment_id`` pattern is
    ``^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$`` — lowercase alphanumerics and
    hyphens ONLY, no underscores. Every registered producer name in
    ``producers.registry.RESEARCH_PRODUCERS`` uses underscores
    (``no_agent_quant``, ``single_agent_quant``, ``agentic_sector_teams``),
    so this is a REQUIRED, deterministic, reversible (`-` <-> `_` is a 1:1
    substitution here since no producer name mixes both) transform, not a
    cosmetic choice — an unconverted name fails schema validation outright.
    """
    return producer_name.replace("_", "-")


def _artifact_link(
    name: str,
    *,
    key: str | None = None,
    reason: str | None = None,
    contract: str | None = None,
) -> dict:
    if key is not None:
        entry = {"name": name, "status": "emitted", "key": key}
    else:
        entry = {"name": name, "status": "absent", "reason": reason or "not produced this cycle"}
    if contract is not None:
        entry["contract"] = contract
    return entry


def build_challenger_experiment_record(
    spec,
    run_date: str,
    *,
    shadow_signals_key: str | None,
    error: str | None = None,
) -> dict:
    """Build the ``experiment_record.v1`` payload for one challenger arm's
    weekly attempt.

    ``spec`` is the ``producers.registry.ProducerSpec`` for this challenger.
    ``shadow_signals_key`` is the S3 key ``write_shadow_signals_json`` already
    returned for a successful attempt (``None`` when the producer raised or
    was otherwise skipped this cycle — ``error`` then carries why, becoming
    the artifact's honest ``absent`` reason).

    A challenger is a single Slot-R (research) implementation identified by
    its own ``entry_point`` — the code that ran is this repo's own resolved
    git SHA (there is no separate installable distribution per producer);
    ``spec.name``/``spec.version`` are folded into the fingerprint ``detail``
    so a manifest hash changes if either the code OR the registered producer
    version changes.

    ``experiment_id`` in the returned payload is :func:`experiment_id_for`
    applied to ``spec.name`` (hyphenated — the schema forbids underscores);
    the S3 KEY this record is written under (:func:`experiment_record_key`)
    instead uses ``spec.name`` verbatim, matching the existing
    ``signals_shadow/{producer_name}/`` prefix convention every other
    challenger artifact already uses.
    """
    code_sha = _resolve_code_sha()
    fingerprint = f"entry_point@{code_sha or 'unknown'}"
    slots = [{
        "slot": "research",
        "impl": "entry_point",
        "fingerprint": fingerprint,
        "detail": f"{spec.name}@{spec.version}",
    }]
    manifest_hash = _manifest_hash(slots)

    if shadow_signals_key:
        artifacts = [_artifact_link("shadow_signals", key=shadow_signals_key)]
        status = "complete"
    else:
        artifacts = [_artifact_link(
            "shadow_signals",
            reason=error or "challenger producer did not emit a shadow cohort this cycle",
        )]
        status = "failed"

    record = {
        "schema_version": 1,
        "experiment_id": experiment_id_for(spec.name),
        "run_date": run_date,
        "status": status,
        "manifest": {"hash": manifest_hash},
        "slots": slots,
        "artifacts": artifacts,
    }
    if code_sha:
        record["git"] = {"crucible-research": code_sha}
    return record


def experiment_record_key(producer_name: str, run_date: str) -> str:
    """S3 key for a challenger's dated experiment record. Uses
    ``producer_name`` (``spec.name``, underscores) verbatim — matching the
    existing ``signals_shadow/{producer_name}/`` prefix convention — NOT the
    hyphenated ``experiment_id`` the payload's ``experiment_id`` field carries
    (see :func:`experiment_id_for`)."""
    return f"{EXPERIMENT_RECORD_PREFIX}/{producer_name}/records/{run_date}.json"


def latest_experiment_record_key(producer_name: str) -> str:
    return f"{EXPERIMENT_RECORD_PREFIX}/{producer_name}/records/{LATEST_EXPERIMENT_RECORD_FILENAME}"


def write_challenger_experiment_record(archive_manager, producer_name: str, run_date: str, record: dict) -> dict:
    """Persist a challenger's experiment record to both the dated key and the
    standing ``latest.json`` pointer, via the SAME ``ArchiveManager._s3_put``
    chokepoint every other archive write uses. Returns
    ``{"dated_key": str, "latest_key": str}``."""
    body = json.dumps(record, indent=2, default=str)
    dated_key = experiment_record_key(producer_name, run_date)
    latest_key = latest_experiment_record_key(producer_name)
    archive_manager._s3_put(dated_key, body)
    archive_manager._s3_put(latest_key, body)
    return {"dated_key": dated_key, "latest_key": latest_key}
