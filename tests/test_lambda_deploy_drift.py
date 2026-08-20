"""Contract tests for the Lambda deploy-drift detector
(alpha-engine-config-I7840).

The behaviours pinned here are the ones whose absence produced a real
incident, not the ones that are easy to assert:

- A merge that changed no deploy-triggering path is NOT drift. This is the
  I7799 false-halt shape, ported: on 2026-08-20 three merges that touched
  nothing in the preopen definition halted preopen trading anyway, unmanaged
  session, open positions carried. Here the equivalent mistake is bare
  stamped-SHA != HEAD, which would be red after almost every docs or tests
  merge because deploy.yml is path-filtered.
- An unmeasured verdict is never reported as in_sync.
- Severity follows the declared criticality, and only ERROR fails the job.
- The registry does not silently fall behind deploy.sh.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import time
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODULE_PATH = _REPO_ROOT / "infrastructure" / "lambda_deploy_drift.py"

_spec = importlib.util.spec_from_file_location("lambda_deploy_drift", _MODULE_PATH)
ldd = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
# Register before exec: dataclasses resolves annotations through
# sys.modules[cls.__module__], which is absent for a module loaded by
# spec alone (AttributeError on 3.14).
sys.modules[_spec.name] = ldd
_spec.loader.exec_module(ldd)


# ── path-filter semantics ────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "path,pattern,expected",
    [
        ("scoring/cut_promotion.py", "scoring/**", True),
        ("scoring/a/b/c.py", "scoring/**", True),
        ("config.py", "config.py", True),
        # `*` must NOT cross a slash, or `config/**`-style patterns and
        # fnmatch-based matching would classify unrelated files as deployable
        # and page for merges that never deploy.
        ("agents/sub/deep.py", "agents/*", False),
        ("requirements-alerts.txt", "requirements*.txt", True),
        ("Dockerfile.alerts", "Dockerfile*", True),
        ("tests/test_scanner.py", "scoring/**", False),
        (".github/workflows/ci.yml", ".github/workflows/deploy.yml", False),
    ],
)
def test_path_matches_github_glob_semantics(path, pattern, expected):
    assert ldd.path_matches(path, pattern) is expected


def test_deploy_paths_are_read_from_the_workflow_not_relisted():
    """The filter is the workflow's own `on.push.paths`.

    A second copy in the detector would go stale in the direction that pages
    for merges which never deploy.
    """
    patterns = ldd.load_deploy_paths()
    text = (_REPO_ROOT / ".github" / "workflows" / "deploy.yml").read_text()
    assert patterns, "deploy.yml declares no push path filter"
    for pattern in patterns:
        assert f"'{pattern}'" in text or f'"{pattern}"' in text


def test_empty_path_filter_raises_rather_than_reporting_healthy(tmp_path):
    """With no filter every verdict would be `no deployable change`, i.e. a
    detector that reports healthy because it cannot see. Refuse instead."""
    bad = tmp_path / "deploy.yml"
    bad.write_text("name: x\non:\n  push:\n    branches: [main]\n")
    with pytest.raises(ValueError, match="on.push.paths"):
        ldd.load_deploy_paths(bad)


# ── verdicts ─────────────────────────────────────────────────────────────────

_HEAD = "b" * 40
_STAMP = "a" * 40
_PATTERNS = ["scoring/**", "lambda/**"]


def _reader(sha, dirty=False):
    def read(name, alias):
        return {"git_sha": sha, "dirty": dirty, "last_modified": "x"}
    return read


def _check(spec, sha, monkeypatch, files=None, commit_age_min=600, dirty=False):
    """Run one verdict with git interaction stubbed."""
    monkeypatch.setattr(ldd, "_commit_exists", lambda s: True)
    if files is not None:
        change = ldd.RelevantChange(
            commits=["c" * 40] if files else [],
            files=set(files),
            oldest_commit_ts=int(time.time()) - commit_age_min * 60,
        )
        monkeypatch.setattr(ldd, "relevant_changes", lambda *a, **k: change)
    return ldd.check_function(
        "alpha-engine-research-scanner", spec, _HEAD, _PATTERNS,
        grace_minutes=45, reader=_reader(sha, dirty),
    )


_CRIT = ldd.FunctionSpec(ldd.CRITICAL, "trading critical path")
_OBS = ldd.FunctionSpec(ldd.OBSERVE, "observe only")


def test_no_deployable_change_is_not_drift(monkeypatch):
    """THE I7799 LESSON. A merge that touched only tests/docs does not even
    trigger deploy.yml, so the stamp legitimately lags main. Reporting that as
    drift would page on nearly every merge — and an alert that is right once
    in twenty is an alert nobody reads, which is how the 2026-08-18 scanner
    stayed stale for two days.
    """
    row = _check(_CRIT, _STAMP, monkeypatch, files=[])
    assert row["verdict"] == "in_sync_no_deployable_change"
    assert row["severity"] == ldd.OK
    assert row["measured"] is True


def test_deployable_change_past_grace_is_error_on_a_critical_function(monkeypatch):
    row = _check(_CRIT, _STAMP, monkeypatch, files=["scoring/cut_promotion.py"])
    assert row["verdict"] == "drifted"
    assert row["severity"] == ldd.ERROR


def test_same_drift_on_an_observe_only_function_is_a_warning(monkeypatch):
    row = _check(_OBS, _STAMP, monkeypatch, files=["scoring/cut_promotion.py"])
    assert row["verdict"] == "drifted"
    assert row["severity"] == ldd.WARNING


def test_a_fresh_deployable_commit_is_pending_not_drift(monkeypatch):
    """A deploy takes ~12-18 minutes. A run landing inside that window sees
    real, correct, in-flight drift; paging for it trains the reader to ignore
    the alert."""
    row = _check(_CRIT, _STAMP, monkeypatch,
                 files=["scoring/x.py"], commit_age_min=5)
    assert row["verdict"] == "pending_deploy"
    assert row["severity"] == ldd.OK


def test_grace_is_measured_from_the_oldest_unshipped_commit(monkeypatch):
    """A fresh merge stacked on a two-day-old undeployed one must not reset
    the clock — otherwise a busy repo can never accumulate a drift verdict."""
    monkeypatch.setattr(ldd, "_commit_exists", lambda s: True)
    now = int(time.time())
    change = ldd.RelevantChange(
        commits=["c" * 40, "d" * 40],
        files={"scoring/x.py"},
        oldest_commit_ts=now - 2 * 24 * 3600,
    )
    monkeypatch.setattr(ldd, "relevant_changes", lambda *a, **k: change)
    row = ldd.check_function(
        "alpha-engine-research-scanner", _CRIT, _HEAD, _PATTERNS,
        grace_minutes=45, now_ts=now, reader=_reader(_STAMP),
    )
    assert row["verdict"] == "drifted"
    assert row["oldest_unshipped_age_minutes"] == 2 * 24 * 60


def test_absent_stamp_is_unmeasured_and_never_in_sync(monkeypatch):
    row = _check(_CRIT, None, monkeypatch)
    assert row["verdict"] == "stamp_absent"
    assert row["measured"] is False
    assert row["severity"] == ldd.WARNING  # a blind spot is not proven drift


def test_unresolvable_stamp_is_unmeasured(monkeypatch):
    monkeypatch.setattr(ldd, "_commit_exists", lambda s: False)
    row = ldd.check_function(
        "alpha-engine-research-scanner", _CRIT, _HEAD, _PATTERNS,
        grace_minutes=45, reader=_reader("f" * 40),
    )
    assert row["verdict"] == "stamp_unknown_commit"
    assert row["measured"] is False


def test_aws_read_failure_is_reported_not_swallowed(monkeypatch):
    def boom(name, alias):
        raise RuntimeError("AccessDenied")
    row = ldd.check_function(
        "alpha-engine-research-scanner", _CRIT, _HEAD, _PATTERNS,
        grace_minutes=45, reader=boom,
    )
    assert row["verdict"] == "config_read_failed"
    assert row["measured"] is False
    assert "AccessDenied" in row["detail"]


def test_a_dirty_deploy_is_drift_even_when_the_sha_matches_head(monkeypatch):
    """A stamp equal to HEAD published from a dirty tree describes code that
    exists nowhere in git. Reporting it as in_sync is the one outcome this
    detector must never produce."""
    row = _check(_CRIT, _HEAD, monkeypatch, dirty=True)
    assert row["verdict"] == "deployed_from_dirty_tree"
    assert row["severity"] == ldd.ERROR


def test_retired_functions_are_declared_not_deleted(monkeypatch):
    spec = ldd.FunctionSpec(ldd.RETIRED, "retired 2026-08-04")
    row = _check(spec, None, monkeypatch)
    assert row["verdict"] == "retired"
    assert row["severity"] == ldd.OK


# ── job outcome ──────────────────────────────────────────────────────────────

def _fake_run(rows):
    watched = [r for r in rows if r["verdict"] != "retired"]
    return {
        "head_sha": _HEAD, "grace_minutes": 45, "deploy_paths": _PATTERNS,
        "functions": rows, "total": len(watched),
        "measured": sum(1 for r in watched if r["measured"]),
        "drifted": sum(1 for r in watched if r["verdict"] == "drifted"),
        "errors": sum(1 for r in watched if r["severity"] == ldd.ERROR),
        "warnings": sum(1 for r in watched if r["severity"] == ldd.WARNING),
    }


def _row(sev, verdict="drifted"):
    return {"function": "f", "criticality": "x", "alias": None,
            "head_sha": _HEAD, "verdict": verdict, "severity": sev,
            "measured": True, "detail": "d"}


def test_only_error_fails_the_job(monkeypatch, capsys):
    monkeypatch.setattr(ldd, "run", lambda **k: _fake_run([_row(ldd.WARNING)]))
    assert ldd.main([]) == 0
    monkeypatch.setattr(ldd, "run", lambda **k: _fake_run([_row(ldd.ERROR)]))
    assert ldd.main([]) == 1


def test_a_warning_is_still_annotated_and_summarised(monkeypatch, capsys):
    monkeypatch.setattr(ldd, "run", lambda **k: _fake_run([_row(ldd.WARNING)]))
    ldd.main([])
    out = capsys.readouterr().out
    assert "::warning" in out           # visible on the console, not silent
    assert "SUMMARY: measured=" in out  # the §2.7 number, every run


def test_summary_line_carries_the_measurability_numbers():
    report = _fake_run([_row(ldd.ERROR), _row(ldd.WARNING, "stamp_absent")])
    text = ldd.render(report)
    assert re.search(r"SUMMARY: measured=\d+/\d+ drifted=\d+ errors=\d+ warnings=\d+", text)
    # The same three numbers reach the alert body, so no surface can disagree
    # with another about how much was actually measured.
    assert "measured=" in ldd.alert_body(report)


def test_alert_is_one_rollup_not_one_per_function(monkeypatch):
    """config#2855: one latent defect, 11 red deploys, 11 pages. Alert fatigue
    is what let the 2026-08-18 drift sit unread."""
    published = []
    import sys
    import types
    fake = types.ModuleType("krepis.alerts")
    fake.publish = lambda **kw: published.append(kw)
    monkeypatch.setitem(sys.modules, "krepis.alerts", fake)
    ldd.emit_alert(_fake_run([_row(ldd.ERROR), _row(ldd.ERROR), _row(ldd.WARNING)]))
    assert len(published) == 1
    assert published[0]["severity"] == ldd.ERROR
    assert published[0]["dedup_key"].startswith("lambda-deploy-drift-")


def test_json_mode_emits_only_json(monkeypatch, capsys):
    """`--json` is what a machine reads. Interleaving ::warning:: workflow
    commands into it makes the machine-readable mode unparseable — and the
    parser's failure would look like the detector finding nothing."""
    import json as _json
    monkeypatch.setattr(ldd, "run", lambda **k: _fake_run([_row(ldd.WARNING)]))
    ldd.main(["--json"])
    _json.loads(capsys.readouterr().out)
