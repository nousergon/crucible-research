"""All-agents-strict for CIO + macro: hard-fail past the 429 deadline,
and persist+resume so an SF redrive doesn't re-pay them.

Per the directive (Brian, 2026-05-16): "We don't get anything from this
process if the sectors, or any other agent for that matter, fail/don't
run." CIO and the macro economist are agents in scope —

  * Their LLM calls are wrapped in the deadline-bounded
    ``invoke_with_rate_limit_retry``; a 429 past the ~75-min deadline
    propagates and (strict-mode default) the run hard-fails. No
    synthetic/empty CIO or macro substitute is promoted.
  * They are persisted on success (``save_agent_run``) and resumed
    (``load_agent_run``) so an SF redrive triggered by a *different*
    agent's failure reuses them with ZERO LLM calls — this is what
    keeps the long retry window affordable across a redrive (extends
    #194's sector-team persist+resume to CIO/macro).

Uses the ``monkeypatch`` fixture (NOT ``unittest.mock.patch``).
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile

import pytest

from agents import langchain_utils
from agents.langchain_utils import invoke_with_rate_limit_retry


class _FakeResp:
    headers: dict = {}


def _make_429():
    import anthropic

    try:
        return anthropic.RateLimitError(
            message="org rate limit",
            response=_FakeResp(),  # type: ignore[arg-type]
            body=None,
        )
    except Exception:
        class _Duck(Exception):
            status_code = 429

            def __init__(self):
                super().__init__("rate limit 429")
                self.response = _FakeResp()

        return _Duck()


# ── 429 past deadline → propagate (caller hard-fails) ─────────────────────


def test_cio_macro_429_past_deadline_propagates(monkeypatch):
    """The shared wrapper used by CIO + macro re-raises the 429 once
    the deadline is exceeded — the strict-mode caller turns that into
    status:ERROR with no signals.json/email. (CIO/macro use the same
    helper; this pins the propagation contract they rely on.)"""
    monkeypatch.setattr(langchain_utils.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        langchain_utils, "RATE_LIMIT_RETRY_DEADLINE_SECONDS", 0.01
    )

    def always_429():
        raise _make_429()

    with pytest.raises(Exception) as exc:
        invoke_with_rate_limit_retry(always_429, label="cio")
    from agents.langchain_utils import _is_rate_limit_error

    assert _is_rate_limit_error(exc.value)


# ── persist + resume (extends #194 pattern to CIO/macro) ──────────────────


class _FakeS3:
    def __init__(self):
        self.store: dict[str, bytes] = {}

    def put_object(self, *, Bucket, Key, Body):  # noqa: N803
        self.store[Key] = Body if isinstance(Body, bytes) else Body.encode()

    def get_object(self, *, Bucket, Key):  # noqa: N803
        if Key not in self.store:
            from botocore.exceptions import ClientError

            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "missing"}},
                "GetObject",
            )

        class _Body:
            def __init__(self, d):
                self._d = d

            def read(self):
                return self._d

        return {"Body": _Body(self.store[Key])}


@pytest.fixture
def archive():
    mod = pytest.importorskip(
        "archive.manager", reason="archive.manager requires gitignored config"
    )
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    am = mod.ArchiveManager(bucket="test-bucket", local_db_path=db_path)
    am.s3 = _FakeS3()
    am.db_conn = sqlite3.connect(db_path)
    am.db_conn.row_factory = sqlite3.Row
    am._ensure_schema()
    yield am
    am.close()
    os.unlink(db_path)


class TestAgentRunPersistence:
    def test_save_writes_expected_key(self, archive):
        archive.save_agent_run("2026-05-16", "cio", {"advanced_tickers": ["A"]})
        key = "archive/agent_runs/2026-05-16/cio.json"
        assert key in archive.s3.store
        env = json.loads(archive.s3.store[key].decode())
        assert env["run_date"] == "2026-05-16"
        assert env["agent_id"] == "cio"
        assert env["output"]["advanced_tickers"] == ["A"]

    def test_round_trip(self, archive):
        archive.save_agent_run("2026-05-16", "macro",
                               {"market_regime": "bull"})
        loaded = archive.load_agent_run("2026-05-16", "macro")
        assert loaded == {"market_regime": "bull"}

    def test_missing_returns_none(self, archive):
        assert archive.load_agent_run("2026-05-16", "cio") is None

    def test_different_run_date_not_reused(self, archive):
        archive.save_agent_run("2026-05-16", "cio", {"x": 1})
        assert archive.load_agent_run("2026-05-23", "cio") is None

    def test_identity_mismatch_falls_back_to_none(self, archive):
        key = "archive/agent_runs/2026-05-16/cio.json"
        archive.s3.store[key] = json.dumps({
            "run_date": "2026-05-09", "agent_id": "cio",
            "output": {"x": 1},
        }).encode()
        assert archive.load_agent_run("2026-05-16", "cio") is None

    def test_corrupt_json_falls_back_to_none(self, archive):
        key = "archive/agent_runs/2026-05-16/cio.json"
        archive.s3.store[key] = b"{ not json"
        assert archive.load_agent_run("2026-05-16", "cio") is None


class TestCioNodeResume:
    def test_cio_node_resumes_with_zero_llm_calls(self, archive, monkeypatch):
        """A persisted CIO output short-circuits cio_node — run_cio is
        NEVER called (zero Sonnet work on redrive)."""
        from graph import research_graph

        archive.save_agent_run(
            "2026-05-16", "cio",
            {"ic_decisions": [], "advanced_tickers": ["NVDA"],
             "entry_theses": {}},
        )

        def _boom(*a, **k):
            raise AssertionError(
                "run_cio called on resume — re-paid the CIO Sonnet call"
            )

        monkeypatch.setattr(research_graph, "run_cio", _boom)

        state = {
            "run_date": "2026-05-16",
            "archive_manager": archive,
            "sector_team_outputs": {},
        }
        out = research_graph.cio_node(state)
        assert out["advanced_tickers"] == ["NVDA"]

    def test_macro_node_resumes_with_zero_llm_calls(
        self, archive, monkeypatch
    ):
        from graph import research_graph

        archive.save_agent_run(
            "2026-05-16", "macro",
            {"macro_report": "real report", "market_regime": "bull",
             "sector_modifiers": {}, "sector_ratings": {}},
        )

        def _boom(*a, **k):
            raise AssertionError(
                "run_macro_agent_with_reflection called on resume"
            )

        monkeypatch.setattr(
            research_graph, "run_macro_agent_with_reflection", _boom
        )

        state = {
            "run_date": "2026-05-16",
            "archive_manager": archive,
            "macro_data": {},
        }
        out = research_graph.macro_economist_node(state)
        assert out["market_regime"] == "bull"
        assert out["macro_report"] == "real report"
