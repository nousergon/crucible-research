"""Producer-side contract test for the challenger-arm ``experiment_record.v1``
boundary (alpha-engine-config#3077 Phase C).

crucible-research is the PRODUCER of one ``experiment_record.v1`` per
challenger arm per run — ``experiments/{spec.name}/records/{run_date}.json``
(+ ``.../latest.json``), built by
``producers.experiment_record.build_challenger_experiment_record`` and
written by ``producers.runner.run_challengers`` right after each challenger's
shadow-signals write. This test builds real payloads via the local builder
and asserts they validate against the shared ``nousergon_lib.contracts``
schema — the single cross-repo source of truth (mirrors
``tests/test_research_intel_producer_contract.py``'s
``contracts.conformance_errors`` producer-contract pattern).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from producers.experiment_record import (  # noqa: E402
    build_challenger_experiment_record,
    experiment_id_for,
    experiment_record_key,
    latest_experiment_record_key,
    write_challenger_experiment_record,
)
from producers.registry import RESEARCH_PRODUCERS  # noqa: E402

_SPEC = RESEARCH_PRODUCERS["no_agent_quant"]
_RUN_DATE = "2026-07-18"


def test_complete_run_validates_against_lib_contract():
    from nousergon_lib import contracts

    record = build_challenger_experiment_record(
        _SPEC, _RUN_DATE,
        shadow_signals_key=f"signals_shadow/{_SPEC.name}/{_RUN_DATE}/signals.json",
    )
    errors = contracts.conformance_errors("experiment_record", record)
    assert errors == [], (
        "experiment_record payload violates the nousergon_lib contract:\n  "
        + "\n  ".join(errors)
    )
    assert record["schema_version"] == 1
    assert record["run_date"] == _RUN_DATE
    assert record["status"] == "complete"


def test_experiment_id_is_hyphenated_not_underscored():
    # The schema's experiment_id pattern (^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$)
    # forbids underscores; every registered producer name uses them
    # (no_agent_quant, single_agent_quant) — this is a REQUIRED transform,
    # not cosmetic, or the payload fails schema validation outright.
    record = build_challenger_experiment_record(
        _SPEC, _RUN_DATE, shadow_signals_key="k",
    )
    assert record["experiment_id"] == "no-agent-quant"
    assert "_" not in record["experiment_id"]


def test_required_top_level_fields_present():
    record = build_challenger_experiment_record(
        _SPEC, _RUN_DATE, shadow_signals_key="k",
    )
    required = {"schema_version", "experiment_id", "run_date", "status", "manifest", "slots", "artifacts"}
    missing = required - record.keys()
    assert not missing, f"experiment_record producer dropped required field(s): {sorted(missing)}"


def test_failed_producer_grades_status_failed_with_absent_artifact():
    # Partial/failed run honesty: a challenger that raised must carry an
    # explicit status="absent" + reason row for shadow_signals — never omit
    # it from the artifacts table.
    from nousergon_lib import contracts

    record = build_challenger_experiment_record(
        _SPEC, _RUN_DATE, shadow_signals_key=None, error="boom: producer raised",
    )
    assert record["status"] == "failed"
    row = next(a for a in record["artifacts"] if a["name"] == "shadow_signals")
    assert row["status"] == "absent"
    assert row["reason"] == "boom: producer raised"
    errors = contracts.conformance_errors("experiment_record", record)
    assert errors == []


def test_manifest_hash_is_deterministic_for_same_producer_and_code_sha():
    r1 = build_challenger_experiment_record(_SPEC, _RUN_DATE, shadow_signals_key="key-a")
    r2 = build_challenger_experiment_record(_SPEC, _RUN_DATE, shadow_signals_key="key-b")
    # The manifest hash covers the SLOTS (which code/producer ran), not the
    # shadow-signals S3 key (where the output landed).
    assert r1["manifest"]["hash"] == r2["manifest"]["hash"]


def test_each_registered_challenger_id_conforms_to_schema():
    from nousergon_lib import contracts
    from producers.registry import challenger_producers

    for spec in challenger_producers():
        record = build_challenger_experiment_record(spec, _RUN_DATE, shadow_signals_key="k")
        assert record["experiment_id"] == experiment_id_for(spec.name)
        errors = contracts.conformance_errors("experiment_record", record)
        assert errors == [], f"{spec.name}: {errors}"


class TestWriteChallengerExperimentRecord:
    def test_writes_dated_and_latest_keys_via_archive_manager(self):
        from unittest.mock import MagicMock

        am = MagicMock()
        record = build_challenger_experiment_record(
            _SPEC, _RUN_DATE, shadow_signals_key="k",
        )
        written = write_challenger_experiment_record(am, _SPEC.name, _RUN_DATE, record)
        assert written["dated_key"] == experiment_record_key(_SPEC.name, _RUN_DATE)
        assert written["latest_key"] == latest_experiment_record_key(_SPEC.name)
        assert written["dated_key"] == f"experiments/{_SPEC.name}/records/{_RUN_DATE}.json"
        assert written["latest_key"] == f"experiments/{_SPEC.name}/records/latest.json"
        # underscore preserved in the S3 KEY (matches signals_shadow/{name}/
        # convention) even though the payload's experiment_id is hyphenated.
        assert "_" in written["dated_key"]
        assert am._s3_put.call_count == 2
