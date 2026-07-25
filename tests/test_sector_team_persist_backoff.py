"""Resilience + resumability pins for the 2026-05-16 multi-team-429 failure.

The 2026-05-16 Saturday SF recovery run failed: the Research Lambda
returned ``status:ERROR`` with::

    "sector team(s) failed: defensives/financials/technology:
     RateLimitError 429 — org rate limit of 450,000 input tokens/min,
     claude-haiku-4-5"

Mechanism (confirmed in code):

  * ``graph/research_graph.py`` ``build_graph()`` compiles the LangGraph
    with NO checkpointer → the Lambda runs it stateless → an SF re-run
    re-dispatches all 6 sector teams via ``Send()`` and re-pays every
    Haiku call.
  * The 6-team parallel fan-out bursts over the org's 450K Haiku
    input-TPM ceiling → ``RateLimitError 429``.
  * ``score_aggregator`` hard-failed the WHOLE run if ANY team carried
    an ``error``, discarding the successful teams (which lived only in
    in-memory graph state).

The 3-part fix, pinned here:

  A. 429-aware exponential backoff (``invoke_with_rate_limit_retry``)
     honoring the ``retry-after`` header around every sector-team Haiku
     ``.invoke()``, plus a higher constructor ``max_retries``.
  B. Per-sector-team S3 persistence on success
     (``ArchiveManager.save_sector_team_run`` →
     ``archive/sector_team_runs/{run_date}/{team_id}.json``).
  C. Resume short-circuit in ``sector_team_node`` (load persisted
     output → zero Haiku calls) + per-team isolation in
     ``score_aggregator`` (a failed team no longer nukes the run when
     other teams produced usable picks).

Load-bearing invariant: a re-invocation must NEVER re-pay a sector team
that already succeeded for that run_date.

These tests use the ``monkeypatch`` fixture (NOT ``unittest.mock.patch``)
to match the convention in ``test_macro_sector_coherence_gate.py`` /
``test_held_thesis_isolation.py`` — this repo has a known full-suite
test-bleed with ``unittest.mock.patch``.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile

import pytest

from agents import langchain_utils
from agents.langchain_utils import (
    SECTOR_TEAM_LLM_MAX_RETRIES,
    invoke_with_rate_limit_retry,
)

# ── Fakes ─────────────────────────────────────────────────────────────────


class _FakeResponse:
    """Minimal stand-in for an anthropic APIStatusError ``.response``."""

    def __init__(self, headers: dict | None = None):
        self.headers = headers or {}


def _make_rate_limit_error(retry_after: str | None = None):
    """Build a real ``anthropic.RateLimitError`` instance.

    The SDK constructor signature varies across versions, so fall back
    to a duck-typed object that ``_is_rate_limit_error`` still matches
    (status_code 429) if the real constructor rejects our kwargs.
    """
    import anthropic

    headers = {"retry-after": retry_after} if retry_after is not None else {}
    resp = _FakeResponse(headers)
    try:
        return anthropic.RateLimitError(
            message="org rate limit of 450,000 input tokens/min",
            response=resp,  # type: ignore[arg-type]
            body=None,
        )
    except Exception:
        class _DuckRateLimit(Exception):
            status_code = 429

            def __init__(self):
                super().__init__("rate limit 429")
                self.response = resp

        return _DuckRateLimit()


class _FakeS3:
    """In-memory S3 backing a dict so put/get round-trips.

    Mirrors just enough of the boto3 client surface that
    ``ArchiveManager._s3_put`` / ``_s3_get`` need.
    """

    def __init__(self):
        self.store: dict[str, bytes] = {}
        self.put_calls = 0

    def put_object(self, *, Bucket, Key, Body):  # noqa: N803
        self.put_calls += 1
        self.store[Key] = Body if isinstance(Body, bytes) else Body.encode()

    def get_object(self, *, Bucket, Key):  # noqa: N803
        if Key not in self.store:
            from botocore.exceptions import ClientError

            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "missing"}},
                "GetObject",
            )

        class _Body:
            def __init__(self, data):
                self._data = data

            def read(self):
                return self._data

        return {"Body": _Body(self.store[Key])}


@pytest.fixture
def archive():
    """An ArchiveManager with in-memory SQLite + the _FakeS3 client."""
    mod = pytest.importorskip(
        "archive.manager",
        reason="archive.manager requires gitignored config",
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


def _team_output(team_id: str, recs: list[dict] | None = None) -> dict:
    return {
        "team_id": team_id,
        "recommendations": recs if recs is not None else [{"ticker": "AAPL"}],
        "thesis_updates": {},
        "quant_output": {"ranked_picks": [{"ticker": "AAPL"}]},
        "qual_output": {},
        "peer_review_output": {},
        "tool_calls": [],
        "error": None,
        "partial": False,
        "partial_reasons": [],
    }


# ── Part A: 429-aware backoff ─────────────────────────────────────────────


class TestRateLimitBackoff:
    def test_constructor_max_retries_constant_is_raised(self):
        # langchain-anthropic defaults to 2; we need a higher floor for
        # sustained org-level 429.
        assert SECTOR_TEAM_LLM_MAX_RETRIES >= 5

    def test_team_429s_then_succeeds_on_retry(self, monkeypatch):
        """(1) A team that 429s then succeeds on backoff-retry: the
        wrapper retries and ultimately returns the success value."""
        sleeps: list[float] = []
        monkeypatch.setattr(
            "agents.langchain_utils.time.sleep",
            lambda s: sleeps.append(s),
        )

        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise _make_rate_limit_error()
            return "OK"

        result = invoke_with_rate_limit_retry(flaky, label="t:react")
        assert result == "OK"
        assert calls["n"] == 3
        # Slept twice (before attempts 2 and 3).
        assert len(sleeps) == 2

    def test_retry_after_header_is_honored(self, monkeypatch):
        sleeps: list[float] = []
        monkeypatch.setattr(
            "agents.langchain_utils.time.sleep",
            lambda s: sleeps.append(s),
        )
        # Zero jitter so we can assert the exact retry-after value.
        monkeypatch.setattr(
            "agents.langchain_utils.random.uniform", lambda a, b: 0.0
        )

        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] == 1:
                raise _make_rate_limit_error(retry_after="17")
            return "OK"

        invoke_with_rate_limit_retry(flaky, label="t:extract")
        assert sleeps == [17.0]

    def test_non_429_propagates_immediately(self, monkeypatch):
        """Non-rate-limit errors are NOT retried — they propagate
        unchanged so strict-mode / partial / isolation paths see them."""
        monkeypatch.setattr(
            "agents.langchain_utils.time.sleep",
            lambda s: (_ for _ in ()).throw(AssertionError("slept")),
        )

        calls = {"n": 0}

        def boom():
            calls["n"] += 1
            raise ValueError("schema validation failed")

        with pytest.raises(ValueError, match="schema validation failed"):
            invoke_with_rate_limit_retry(boom, label="t:x")
        assert calls["n"] == 1

    def test_persistent_429_raises_at_deadline_not_attempt_count(
        self, monkeypatch
    ):
        """REWRITTEN (was test_persistent_429_eventually_raises): the
        wrapper is now deadline-bounded, NOT a fixed ``max_attempts``
        cap (that kwarg was removed). A persistent 429 retries until
        the deadline then re-raises the 429 — the all-agents-strict
        caller turns that into status:ERROR (no signals.json/email)."""
        monkeypatch.setattr(
            "agents.langchain_utils.time.sleep", lambda s: None
        )
        # Tiny deadline so the loop gives up fast.
        monkeypatch.setattr(
            langchain_utils, "RATE_LIMIT_RETRY_DEADLINE_SECONDS", 0.01
        )

        calls = {"n": 0}

        def always_429():
            calls["n"] += 1
            raise _make_rate_limit_error()

        with pytest.raises(Exception) as exc:
            invoke_with_rate_limit_retry(always_429, label="t:x")
        from agents.langchain_utils import _is_rate_limit_error

        assert _is_rate_limit_error(exc.value)
        # It DID attempt at least once before the deadline check fired.
        assert calls["n"] >= 1

    def test_deadline_constant_default_is_75_min(self):
        """The locked default: ~75 min wall-clock per-invoke 429
        retry window (overridable via RATE_LIMIT_RETRY_DEADLINE_SECONDS,
        clamped 5 min .. 3 hr)."""
        assert langchain_utils._resolve_deadline_seconds.__module__
        # Default (no env) resolves to 75 minutes.
        import os

        prev = os.environ.pop("RATE_LIMIT_RETRY_DEADLINE_SECONDS", None)
        try:
            assert langchain_utils._resolve_deadline_seconds() == 75 * 60
        finally:
            if prev is not None:
                os.environ["RATE_LIMIT_RETRY_DEADLINE_SECONDS"] = prev

    def test_long_429_then_success_within_deadline(self, monkeypatch):
        """A team that 429s for a while then succeeds BEFORE the
        deadline proceeds with its real output (the long-retry window
        is what makes a 60-90 min org-TPM stall survivable)."""
        monkeypatch.setattr(
            "agents.langchain_utils.time.sleep", lambda s: None
        )
        monkeypatch.setattr(
            langchain_utils, "RATE_LIMIT_RETRY_DEADLINE_SECONDS", 3600.0
        )

        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 8:  # 429 for the first 7 attempts
                raise _make_rate_limit_error()
            return "REAL"

        assert invoke_with_rate_limit_retry(flaky, label="t:x") == "REAL"
        assert calls["n"] == 8


# ── Part B: per-team S3 persistence ───────────────────────────────────────


class TestSectorTeamPersistence:
    def test_save_writes_to_expected_s3_key(self, archive):
        """(2) Team output persisted to the expected S3 key on success."""
        out = _team_output("technology")
        archive.save_sector_team_run("2026-05-16", "technology", out)
        key = "archive/sector_team_runs/2026-05-16/technology.json"
        assert key in archive.s3.store
        envelope = json.loads(archive.s3.store[key].decode())
        assert envelope["run_date"] == "2026-05-16"
        assert envelope["team_id"] == "technology"
        assert envelope["output"]["team_id"] == "technology"

    def test_round_trip_load_returns_output(self, archive):
        out = _team_output("healthcare", recs=[{"ticker": "LLY"}])
        archive.save_sector_team_run("2026-05-16", "healthcare", out)
        loaded = archive.load_sector_team_run("2026-05-16", "healthcare")
        assert loaded is not None
        assert loaded["team_id"] == "healthcare"
        assert loaded["recommendations"] == [{"ticker": "LLY"}]

    def test_load_missing_returns_none(self, archive):
        assert archive.load_sector_team_run("2026-05-16", "nope") is None

    def test_different_run_date_does_not_reuse(self, archive):
        """(6) A new run_date must NOT reuse a prior date's persisted
        team — the key is run-date-scoped."""
        archive.save_sector_team_run(
            "2026-05-16", "technology", _team_output("technology")
        )
        # A different run_date has no persisted object → None → re-run.
        assert (
            archive.load_sector_team_run("2026-05-23", "technology") is None
        )

    def test_corrupt_json_falls_back_to_none(self, archive):
        key = "archive/sector_team_runs/2026-05-16/technology.json"
        archive.s3.store[key] = b"{ this is not valid json"
        # Must not raise — stale/corrupt → re-run that team.
        assert (
            archive.load_sector_team_run("2026-05-16", "technology") is None
        )

    def test_identity_mismatch_falls_back_to_none(self, archive):
        """A cross-wired envelope (wrong run_date inside) is never
        reused even if it parses."""
        key = "archive/sector_team_runs/2026-05-16/technology.json"
        archive.s3.store[key] = json.dumps({
            "run_date": "2026-05-09",  # stale
            "team_id": "technology",
            "output": _team_output("technology"),
        }).encode()
        assert (
            archive.load_sector_team_run("2026-05-16", "technology") is None
        )

    def test_partial_team_is_not_persisted_by_saver(self, archive):
        """config#1822: a PARTIAL team (qual step-budget exhaustion → 0
        assessments) must NOT be written as a resumable checkpoint —
        persisting it poisons every future rerun's resume short-circuit."""
        out = _team_output("healthcare", recs=[])
        out["partial"] = True
        out["partial_reasons"] = ["qual:remaining_steps_exhausted"]
        archive.save_sector_team_run("2026-05-16", "healthcare", out)
        key = "archive/sector_team_runs/2026-05-16/healthcare.json"
        assert key not in archive.s3.store, (
            "a partial team must NOT be persisted"
        )

    def test_errored_team_is_not_persisted_by_saver(self, archive):
        """The saver enforces the no-error guard itself (defense in depth,
        independent of the node-level call-site check)."""
        out = _team_output("financials", recs=[])
        out["error"] = "RateLimitError 429 — org rate limit"
        archive.save_sector_team_run("2026-05-16", "financials", out)
        key = "archive/sector_team_runs/2026-05-16/financials.json"
        assert key not in archive.s3.store

    def test_load_ignores_pre_existing_partial_artifact(self, archive):
        """config#1822: an ALREADY-persisted partial artifact (written
        before the persist-side guard existed) must be treated as absent on
        resume, so the poisoned team re-runs fresh and self-heals — no
        manual S3 surgery. This is the load-bearing recovery guarantee."""
        key = "archive/sector_team_runs/2026-05-16/consumer.json"
        partial = _team_output("consumer", recs=[])
        partial["partial"] = True
        partial["partial_reasons"] = ["qual:remaining_steps_exhausted"]
        archive.s3.store[key] = json.dumps({
            "run_date": "2026-05-16",
            "team_id": "consumer",
            "output": partial,
        }).encode()
        assert (
            archive.load_sector_team_run("2026-05-16", "consumer") is None
        ), "a persisted partial team must NOT short-circuit a rerun"

    def test_load_ignores_pre_existing_errored_artifact(self, archive):
        """Same guarantee for an errored artifact."""
        key = "archive/sector_team_runs/2026-05-16/industrials.json"
        errored = _team_output("industrials", recs=[])
        errored["error"] = "some hard error"
        archive.s3.store[key] = json.dumps({
            "run_date": "2026-05-16",
            "team_id": "industrials",
            "output": errored,
        }).encode()
        assert (
            archive.load_sector_team_run("2026-05-16", "industrials") is None
        )

    def test_successful_team_still_round_trips(self, archive):
        """The guard must NOT regress the load-bearing invariant: a fully
        SUCCESSFUL team is still persisted and resumed (zero re-work)."""
        out = _team_output("technology", recs=[{"ticker": "NVDA"}])
        archive.save_sector_team_run("2026-05-16", "technology", out)
        loaded = archive.load_sector_team_run("2026-05-16", "technology")
        assert loaded is not None
        assert loaded["recommendations"] == [{"ticker": "NVDA"}]


# ── Part C: resume short-circuit + persist-on-success in the node ─────────


def _node_state(team_id: str, am, run_date: str = "2026-05-16") -> dict:
    return {
        "team_id": team_id,
        "run_date": run_date,
        "archive_manager": am,
        "scanner_universe": [],
        "sector_map": {},
        "price_data": {},
        "technical_scores": {},
        "market_regime": "neutral",
        "prior_theses": {},
        "population_tickers": [],
        "news_data_by_ticker": {},
        "analyst_data_by_ticker": {},
        "insider_data_by_ticker": {},
        "prior_sector_ratings": {},
        "sector_ratings": {},
        "episodic_memories": {},
        "semantic_memories": {},
        "regime_substrate": {},
        "focus_list_by_team": {},
    }


class TestSectorTeamNodeResume:
    def test_resume_short_circuits_with_zero_llm_calls(
        self, archive, monkeypatch
    ):
        """(3) On re-invocation with a persisted output present, the
        team is loaded from S3 and run_sector_team is NEVER called
        (zero ChatAnthropic / Haiku work)."""
        from graph import research_graph

        # Pre-seed a persisted technology output (a prior invocation).
        archive.save_sector_team_run(
            "2026-05-16", "technology",
            _team_output("technology", recs=[{"ticker": "NVDA"}]),
        )

        # Tripwire: run_sector_team must NOT be invoked on resume.
        def _boom(*a, **k):
            raise AssertionError(
                "run_sector_team called on resume — re-paid a "
                "completed team (load-bearing invariant violated)"
            )

        monkeypatch.setattr(research_graph, "run_sector_team", _boom)

        state = _node_state("technology", archive)
        out = research_graph.sector_team_node(state)

        assert "sector_team_outputs" in out
        team_out = out["sector_team_outputs"]["technology"]
        assert team_out["recommendations"] == [{"ticker": "NVDA"}]

    def test_success_persists_then_next_run_resumes(
        self, archive, monkeypatch
    ):
        """Run 1: team executes and is persisted. Run 2 (same run_date):
        resumes from S3 with zero re-execution. End-to-end invariant."""
        from graph import research_graph

        run_calls = {"n": 0}

        def _fake_run(team_id, ctx):
            run_calls["n"] += 1
            return _team_output(team_id, recs=[{"ticker": "MSFT"}])

        # Skip schema validation + decision-capture side effects.
        monkeypatch.setattr(research_graph, "run_sector_team", _fake_run)
        monkeypatch.setattr(
            research_graph, "_validate", lambda *a, **k: None
        )
        monkeypatch.setattr(
            research_graph, "_capture_if_enabled", lambda **k: None
        )
        monkeypatch.setattr(
            research_graph, "track_llm_cost",
            lambda *a, **k: _NullCtx(),
        )

        state = _node_state("consumer", archive)

        out1 = research_graph.sector_team_node(state)
        assert run_calls["n"] == 1
        assert out1["sector_team_outputs"]["consumer"]["recommendations"] == [
            {"ticker": "MSFT"}
        ]
        # Persisted to the deterministic key.
        key = "archive/sector_team_runs/2026-05-16/consumer.json"
        assert key in archive.s3.store

        # Run 2 — same run_date → resume, no re-execution.
        out2 = research_graph.sector_team_node(state)
        assert run_calls["n"] == 1, "team re-executed on resume"
        assert out2["sector_team_outputs"]["consumer"]["recommendations"] == [
            {"ticker": "MSFT"}
        ]

    def test_errored_team_is_not_persisted(self, archive, monkeypatch):
        """A team that ERRORs (e.g. 429 survived backoff) is NOT
        persisted, so a re-run gets a fresh attempt at it (the backoff
        / a TPM-window reset may let it succeed next time)."""
        from graph import research_graph

        def _fake_run(team_id, ctx):
            r = _team_output(team_id, recs=[])
            r["error"] = "RateLimitError 429 — org rate limit"
            return r

        monkeypatch.setattr(research_graph, "run_sector_team", _fake_run)
        monkeypatch.setattr(
            research_graph, "_validate", lambda *a, **k: None
        )
        monkeypatch.setattr(
            research_graph, "_capture_if_enabled", lambda **k: None
        )
        monkeypatch.setattr(
            research_graph, "track_llm_cost",
            lambda *a, **k: _NullCtx(),
        )

        state = _node_state("financials", archive)
        research_graph.sector_team_node(state)

        key = "archive/sector_team_runs/2026-05-16/financials.json"
        assert key not in archive.s3.store, (
            "an errored team must NOT be persisted"
        )


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
