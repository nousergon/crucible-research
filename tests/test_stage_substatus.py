"""A stage's status is no better than the worst of its own sub-results.

`alpha-engine-config-I10198`, enforcing `sf-pipeline-policy.md` §2.3b, clause
`SFP-2.3b-stage-status-is-the-worst-substatus` (`nous-ergon-ops-PR1119`).

Asserted against the REAL execution, not a fixture invented here:
`tests/fixtures/sf_substatus/weekly_2026-08-15_scheduled.json` is a verbatim
copy of the 76 stage results `nous-ergon-ops-PR1119` froze from execution
`54acfc69-…_f1036888-…`. Both repos assert against ONE artifact — the ops
copy is the after-the-fact conformance probe, this is the producer-side
prevention, and a reproduction that drifted between the two would let one of
them go green on a shape the other still fails.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import re
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FIXTURE = _REPO_ROOT / "fixtures" if False else (
    _REPO_ROOT / "tests" / "fixtures" / "sf_substatus" / "weekly_2026-08-15_scheduled.json"
)


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture(scope="module")
def sub():
    return _load("stage_substatus.py", "_i10198_stage_substatus")


@pytest.fixture(scope="module")
def frozen_history() -> list[dict]:
    return json.loads(_FIXTURE.read_text())


def _payload(history: list[dict], state: str) -> dict:
    for row in history:
        if row["state"] != state:
            continue
        for value in row["result"].values():
            if isinstance(value, dict) and isinstance(value.get("Payload"), dict):
                return value["Payload"]
    raise AssertionError(f"{state} not in the frozen history")


class TestTheRealFailure:
    def test_the_fixture_carries_no_infra_identifier(self):
        """This repo is PUBLIC. The frozen history came from a live execution
        and carried the AWS account id and an IAM role name inside an
        `AccessDenied` message; both are redacted here and nothing else is.
        `repository-tiering-policy` applies a PURPOSE test, not a secrecy
        test — an identifier has no public job in a test fixture, whether or
        not it is exploitable on its own. Asserted so a future refresh from
        the ops copy cannot quietly restore them.
        """
        raw = _FIXTURE.read_text()
        assert "REDACTED-ACCOUNT-ID" in raw
        assert "REDACTED-ROLE" in raw
        assert not re.search(r"\b\d{12}\b", raw), (
            "a 12-digit AWS account id is in the fixture"
        )

    def test_the_fixture_is_the_shape_that_cost_two_weeks(self, frozen_history):
        """Guards the reproduction itself: if this stops matching, every test
        below is asserting against something that never happened."""
        payload = _payload(frozen_history, "EvalRollingMean")
        assert payload["status"] == "OK"
        assert payload["agent_quality"]["status"] == "ERROR"
        assert "isoformat" in payload["agent_quality"]["error"]

    def test_the_2026_08_15_payload_is_now_a_stage_failure(self, sub, frozen_history):
        """The whole point. This exact payload returned `OK`, so the SF
        `Catch` never fired, `MarkEvalRollingMeanDegraded` never ran,
        `$.research_degraded_local` was never set, and
        `AlphaEngine/Eval/agent_quality_score` published nothing for the weeks
        of 2026-08-12 and 2026-08-19 while every surface read green."""
        payload = dict(_payload(frozen_history, "EvalRollingMean"))
        with pytest.raises(sub.StageSubResultError) as excinfo:
            sub.enforce_worst_substatus(payload, stage="EvalRollingMean")
        assert payload["status"] == "ERROR"
        assert "agent_quality" in str(excinfo.value)
        assert "isoformat" in str(excinfo.value)
        assert [f["path"] for f in excinfo.value.findings] == ["agent_quality"]

    def test_only_the_errored_substatus_fails_the_stage(self, sub, frozen_history):
        """Precision over the same 76 real results, which also carry
        `insufficient`, `PARTIAL`, `MEASURED` and `STALE`. Exactly one state
        may fail — a check that fired on its healthy neighbours would be
        suppressed within a week."""
        failed = []
        for row in frozen_history:
            for value in row["result"].values():
                payload = value.get("Payload") if isinstance(value, dict) else None
                if not isinstance(payload, dict) or payload.get("status") not in sub.PASS_STATUSES:
                    continue
                try:
                    sub.enforce_worst_substatus(dict(payload), stage=row["state"])
                except sub.StageSubResultError:
                    failed.append(row["state"])
        assert failed == ["EvalRollingMean"], failed


class TestTheDerivationIsStructural:
    def test_a_nested_sub_result_at_any_depth_is_found(self, sub):
        findings = sub.find_failed_substatuses(
            {"status": "OK", "a": {"b": {"status": "ERROR", "error": "deep"}}}
        )
        assert [f["path"] for f in findings] == ["a.b"]

    def test_no_sub_result_key_is_named_anywhere_in_the_module(self):
        """A hand-listed set of sub-keys is the shape that let this survive:
        `agent_quality` would have joined such a list only after the week it
        cost. The only named keys are the EXCLUSIONS, each with its reason."""
        source = (_REPO_ROOT / "stage_substatus.py").read_text()
        code = "\n".join(
            line for line in source.splitlines()
            if not line.lstrip().startswith("#")
        )
        code = code.split('"""', 2)[-1]  # drop the module docstring
        for key in ("agent_quality", "control_bands", "producer_leaderboard", "calibration"):
            assert key not in code, (
                f"{key!r} is named in the derivation's CODE — the check must "
                "be structural, not a list of sub-results someone remembered"
            )

    def test_every_exclusion_carries_its_reason(self, sub):
        for key, reason in sub.EXCLUDED_SUBRESULT_KEYS.items():
            assert reason.strip(), f"{key} is excluded with no reason recorded"

    def test_every_degraded_status_carries_its_reason(self, sub):
        for key, reason in sub.DEGRADED_STATUSES.items():
            assert reason.strip(), f"{key} is classified degraded with no reason"

    def test_the_vocabularies_do_not_overlap(self, sub):
        assert not (sub.PASS_STATUSES & sub.ERROR_STATUSES)
        assert not (set(sub.DEGRADED_STATUSES) & sub.ERROR_STATUSES)
        assert not (set(sub.DEGRADED_STATUSES) & sub.PASS_STATUSES)

    def test_stage_coverage_is_excluded_by_name_with_its_reason(self, sub):
        """It is an OBSERVER of the stage, not a sub-result of it, and it
        already pages on its own artifact — a second reader would page twice
        for one fact (alpha-engine-config-I10130)."""
        assert "stage_coverage" in sub.EXCLUDED_SUBRESULT_KEYS
        result = {"status": "OK", "stage_coverage": {"status": "UNMEASURED"}}
        assert sub.find_failed_substatuses(result) == []

    def test_a_word_no_vocabulary_knows_is_reported_never_defaulted_to_pass(
        self, sub, caplog,
    ):
        result = {"status": "OK", "thing": {"status": "wobbly"}}
        with caplog.at_level(logging.ERROR):
            out = sub.enforce_worst_substatus(result, stage="X")
        assert out["substatus_unclassified"] is True
        assert out["substatus_findings"][0]["kind"] == "unclassified"
        assert any(r.levelno >= logging.ERROR for r in caplog.records)

    def test_a_named_partial_outcome_is_recorded_but_does_not_raise(self, sub):
        result = {"status": "OK", "thing": {"status": "PARTIAL"}}
        out = sub.enforce_worst_substatus(result, stage="X")
        assert out["status"] == "OK"
        assert out["substatus_degraded"] is True

    def test_a_clean_payload_is_untouched(self, sub):
        result = {"status": "OK", "thing": {"status": "OK"}}
        assert sub.enforce_worst_substatus(result, stage="X") == {
            "status": "OK", "thing": {"status": "OK"},
        }

    def test_a_stage_with_no_sub_results_is_a_no_op(self, sub):
        result = {"status": "OK", "count": 3}
        assert sub.enforce_worst_substatus(result, stage="X") == result

    def test_an_errored_sub_result_is_not_descended_into(self, sub):
        """Its own children are that failure's detail, not new findings —
        one cause must not be reported N times."""
        findings = sub.find_failed_substatuses(
            {"status": "OK", "a": {"status": "ERROR", "b": {"status": "ERROR"}}}
        )
        assert [f["path"] for f in findings] == ["a"]


class TestTheClassNotTheInstance:
    """Every handler in this repo whose payload can carry a sub-result must
    route its terminal status through the derivation. Population is DERIVED
    from the repo, not listed here — a sixth handler added tomorrow is held to
    the rule without a test edit, which is the whole difference between fixing
    a defect and fixing its class.
    """

    #: Files the rule does NOT yet reach, each with its reason and its
    #: tracked follow-up. An exclusion list with reasons beside it is the only
    #: kind that cannot widen silently; removing an entry shows up in a diff
    #: as a widening of scope.
    OUT_OF_SCOPE: dict[str, str] = {
        "lambda/scanner_handler.py": (
            "A REAL second instance of the class, NOT a false positive: "
            "`summary.weekly_ledger`, `summary.cut_promotion`, "
            "`summary.universe_board` and `summary.spec_promotion` all carry "
            "`status: \"error\"` under a `Scanner` / `ScannerLeaderboard` "
            "payload that reports OK. It is excluded from THIS change because "
            "reversing it also reverses a deliberately-designed fail-soft "
            "contract stated in `tests/test_weekly_ledger_wiring.py::"
            "test_a_ledger_failure_never_reds_the_scanner_run` — a ruling "
            "made elsewhere, which this PR is not the place to overturn. "
            "Carried by alpha-engine-config-I10198's class sweep."
        ),
    }

    @staticmethod
    def _handlers_with_substatus_producing_sub_results() -> list[str]:
        found = []
        for path in sorted((_REPO_ROOT / "lambda").glob("*.py")):
            source = path.read_text()
            # A handler that builds a `result` dict AND asserts stage coverage
            # is one the weekly SF reads a status from.
            if "assert_stage_coverage" in source and 'result = {' in source:
                found.append(str(path.relative_to(_REPO_ROOT)))
        return found

    def test_the_scan_is_not_vacuous(self):
        assert len(self._handlers_with_substatus_producing_sub_results()) >= 3

    def test_every_exclusion_carries_its_reason_and_its_follow_up(self):
        for filename, reason in self.OUT_OF_SCOPE.items():
            assert (_REPO_ROOT / filename).exists(), filename
            assert "alpha-engine-config-I" in reason, (
                f"{filename} is excluded with no tracked follow-up named"
            )

    @pytest.mark.parametrize("filename", _handlers_with_substatus_producing_sub_results.__func__())
    def test_handler_enforces_the_worst_substatus(self, filename):
        if filename in TestTheClassNotTheInstance.OUT_OF_SCOPE:
            pytest.skip(TestTheClassNotTheInstance.OUT_OF_SCOPE[filename])
        source = (_REPO_ROOT / filename).read_text()
        assert "enforce_worst_substatus" in source, (
            f"{filename} returns a status over sub-results it never checks — "
            "the alpha-engine-config-I10198 swallow (sf-pipeline-policy §2.3b)"
        )
