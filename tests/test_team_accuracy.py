"""Tests for `evals.team_accuracy` (config#1422 — adaptive slot allocation
producer).

Synthetic SQLite fixture mirrors the research.db `cio_evaluations` +
`score_performance` schema. Each test focuses on one invariant of the
team-accuracy build path.

`_seed_signal` seeds BOTH the wide `score_performance.beat_spy_21d` column
(schema realism) AND the long-format `score_performance_outcomes` store
(config#1483/config#1530 — the ACTUAL source
`evals.outcome_store.load_primary_outcomes` reads) so existing test call
sites exercise the real post-cutover read path unchanged.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path

import pytest

from evals.team_accuracy import (
    SCHEMA_VERSION,
    TEAM_ACCURACY_S3_KEY,
    analyze_team_performance,
    save_team_accuracy,
)


def _make_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE cio_evaluations (
            id INTEGER PRIMARY KEY,
            ticker TEXT NOT NULL,
            eval_date TEXT NOT NULL,
            team_id TEXT,
            quant_score REAL,
            qual_score REAL,
            combined_score REAL,
            macro_shift REAL,
            final_score REAL,
            neutralized_final_score REAL,
            cio_decision TEXT NOT NULL,
            cio_conviction INTEGER,
            cio_rank INTEGER,
            rationale TEXT,
            rule_tags TEXT,
            UNIQUE(ticker, eval_date)
        );
        CREATE TABLE score_performance (
            id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            score_date TEXT NOT NULL,
            score REAL NOT NULL,
            beat_spy_21d INTEGER,
            log_alpha_21d REAL,
            UNIQUE(symbol, score_date)
        );
        CREATE TABLE score_performance_outcomes (
            id INTEGER PRIMARY KEY,
            signal_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            score_date TEXT NOT NULL,
            horizon_days INTEGER NOT NULL,
            beat_spy INTEGER,
            stock_return REAL,
            spy_return REAL,
            log_alpha REAL,
            is_primary INTEGER NOT NULL,
            resolved_at TEXT NOT NULL,
            schema_version INTEGER NOT NULL DEFAULT 1,
            UNIQUE(signal_id, horizon_days)
        );
        """
    )
    conn.commit()
    return conn


# The canonical primary horizon (nousergon_lib.quant.horizons.DEFAULT_POLICY).
_PRIMARY_HORIZON = 21


def _seed_cio(conn, ticker, eval_date, team_id, decision="ADVANCE"):
    conn.execute(
        "INSERT INTO cio_evaluations (ticker, eval_date, team_id, cio_decision) "
        "VALUES (?, ?, ?, ?)",
        (ticker, eval_date, team_id, decision),
    )


def _seed_signal(conn, symbol, score_date, beat_21d):
    conn.execute(
        "INSERT INTO score_performance (symbol, score_date, score, beat_spy_21d) "
        "VALUES (?, ?, ?, ?)",
        (symbol, score_date, 70.0, beat_21d),
    )
    if beat_21d is not None:
        conn.execute(
            "INSERT INTO score_performance_outcomes "
            "(signal_id, symbol, score_date, horizon_days, beat_spy, "
            " stock_return, spy_return, log_alpha, is_primary, resolved_at) "
            "VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, 1, ?)",
            (
                f"{symbol}:{score_date}", symbol, score_date, _PRIMARY_HORIZON,
                beat_21d, f"{score_date}T00:00:00+00:00",
            ),
        )


@pytest.fixture
def empty_db(tmp_path):
    conn = _make_db(tmp_path / "research.db")
    yield conn
    conn.close()


@pytest.fixture
def populated_db(tmp_path):
    conn = _make_db(tmp_path / "research.db")
    # technology: 3 ADVANCEd picks, 2 beats -> 2/3
    _seed_cio(conn, "AAPL", "2026-05-09", "technology")
    _seed_signal(conn, "AAPL", "2026-05-09", 1)
    _seed_cio(conn, "MSFT", "2026-05-09", "technology")
    _seed_signal(conn, "MSFT", "2026-05-09", 1)
    _seed_cio(conn, "GOOG", "2026-05-09", "technology")
    _seed_signal(conn, "GOOG", "2026-05-09", 0)

    # healthcare: 1 ADVANCEd pick, 0 beats -> 0/1
    _seed_cio(conn, "JNJ", "2026-05-09", "healthcare")
    _seed_signal(conn, "JNJ", "2026-05-09", 0)

    # REJECTed pick — must NOT count even though it has a resolved outcome
    # (score_performance is keyed only by BUY-threshold score, independent
    # of CIO decision).
    _seed_cio(conn, "TSLA", "2026-05-09", "technology", decision="REJECT")
    _seed_signal(conn, "TSLA", "2026-05-09", 1)

    # NO_ADVANCE_DEADLOCK — also must not count.
    _seed_cio(conn, "NVDA", "2026-05-09", "technology", decision="NO_ADVANCE_DEADLOCK")
    _seed_signal(conn, "NVDA", "2026-05-09", 1)

    conn.commit()
    yield conn
    conn.close()


class TestEmptyDB:
    def test_zero_data_emits_insufficient_envelope_never_bare_dict(self, empty_db):
        """config#1844: an empty window must NEVER produce a bare `{}` —
        the envelope carries status + the counts that explain why."""
        result = analyze_team_performance(empty_db, as_of_date=date(2026, 5, 23))
        assert result == {
            "schema_version": SCHEMA_VERSION,
            "status": "insufficient",
            "as_of": "2026-05-23",
            "n_teams": 0,
            "n_advance_picks": 0,
            "n_resolved_outcomes": 0,
            "horizon_days": _PRIMARY_HORIZON,
            "teams": {},
        }

    def test_zero_data_warns_with_counts(self, empty_db, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="evals.team_accuracy"):
            analyze_team_performance(empty_db, as_of_date=date(2026, 5, 23))
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "insufficient status must WARN (fail-loud doctrine)"
        msg = warnings[0].getMessage()
        assert "insufficient" in msg
        assert "n_advance_picks=0" in msg
        assert "n_resolved_outcomes=0" in msg


class TestAnalyzeTeamPerformance:
    def test_per_team_accuracy_and_n_obs(self, populated_db):
        result = analyze_team_performance(populated_db, as_of_date=date(2026, 5, 23))
        teams = result["teams"]
        assert teams["technology"] == {"accuracy": pytest.approx(2 / 3), "n_obs": 3}
        assert teams["healthcare"] == {"accuracy": 0.0, "n_obs": 1}

    def test_envelope_counts_and_status_on_real_data(self, populated_db):
        result = analyze_team_performance(populated_db, as_of_date=date(2026, 5, 23))
        assert result["schema_version"] == SCHEMA_VERSION
        assert result["status"] == "ok"
        assert result["as_of"] == "2026-05-23"
        assert result["n_teams"] == 2
        # 4 ADVANCE picks in window (AAPL/MSFT/GOOG/JNJ) — TSLA (REJECT) and
        # NVDA (NO_ADVANCE_DEADLOCK) excluded.
        assert result["n_advance_picks"] == 4
        # All 4 resolved; invariant: n_resolved_outcomes == sum of n_obs.
        assert result["n_resolved_outcomes"] == 4
        assert result["n_resolved_outcomes"] == sum(
            v["n_obs"] for v in result["teams"].values()
        )
        assert result["horizon_days"] == _PRIMARY_HORIZON

    def test_unresolved_advance_counts_in_picks_not_outcomes(self, populated_db):
        # An ADVANCE pick with no resolved outcome raises n_advance_picks
        # but not n_resolved_outcomes — the pair is exactly what
        # distinguishes "not resolved yet" from "join broken".
        _seed_cio(populated_db, "XOM", "2026-05-10", "defensives")
        populated_db.commit()
        result = analyze_team_performance(populated_db, as_of_date=date(2026, 5, 23))
        assert result["n_advance_picks"] == 5
        assert result["n_resolved_outcomes"] == 4
        assert "defensives" not in result["teams"]

    def test_rejected_and_deadlocked_picks_excluded(self, populated_db):
        result = analyze_team_performance(populated_db, as_of_date=date(2026, 5, 23))
        # If TSLA/NVDA (both realized beat_spy_21d=1) leaked in, technology's
        # n_obs would be 5 and accuracy would be 4/5, not 3 and 2/3.
        assert result["teams"]["technology"]["n_obs"] == 3

    def test_team_with_zero_observations_omitted(self, empty_db):
        _seed_cio(empty_db, "XOM", "2026-05-09", "defensives")
        # No matching score_performance row -> unresolved, must not appear.
        empty_db.commit()
        result = analyze_team_performance(empty_db, as_of_date=date(2026, 5, 23))
        assert "defensives" not in result["teams"]
        assert result["status"] == "insufficient"
        assert result["n_advance_picks"] == 1
        assert result["n_resolved_outcomes"] == 0

    def test_unresolved_signal_excluded(self, empty_db):
        _seed_cio(empty_db, "XOM", "2026-05-09", "defensives")
        _seed_signal(empty_db, "XOM", "2026-05-09", None)
        empty_db.commit()
        result = analyze_team_performance(empty_db, as_of_date=date(2026, 5, 23))
        assert "defensives" not in result["teams"]

    def test_null_team_id_excluded(self, empty_db):
        _seed_cio(empty_db, "XOM", "2026-05-09", None)
        _seed_signal(empty_db, "XOM", "2026-05-09", 1)
        empty_db.commit()
        result = analyze_team_performance(empty_db, as_of_date=date(2026, 5, 23))
        assert result["teams"] == {}
        assert result["n_advance_picks"] == 0  # NULL team_id filtered in SQL

    def test_window_excludes_current_cycle_and_stale_history(self, empty_db):
        _seed_cio(empty_db, "AAPL", "2026-05-23", "technology")  # same day as as_of
        _seed_signal(empty_db, "AAPL", "2026-05-23", 1)
        _seed_cio(empty_db, "MSFT", "2020-01-01", "technology")  # far outside lookback
        _seed_signal(empty_db, "MSFT", "2020-01-01", 1)
        empty_db.commit()
        result = analyze_team_performance(empty_db, as_of_date=date(2026, 5, 23))
        assert "technology" not in result["teams"]

    def test_lookback_weeks_override(self, empty_db):
        _seed_cio(empty_db, "AAPL", "2025-01-01", "technology")
        _seed_signal(empty_db, "AAPL", "2025-01-01", 1)
        empty_db.commit()
        # Default 26-week lookback from 2026-05-23 doesn't reach back to
        # 2025-01-01 (that's ~73 weeks prior).
        default = analyze_team_performance(empty_db, as_of_date=date(2026, 5, 23))
        assert default["teams"] == {}
        # A wide-enough explicit window does.
        result = analyze_team_performance(
            empty_db, as_of_date=date(2026, 5, 23), lookback_weeks=104
        )
        assert result["teams"]["technology"]["n_obs"] == 1


class _StubS3Client:
    """Captures put_object calls for assertion."""

    def __init__(self, fail: bool = False):
        self.calls: list[dict] = []
        self._fail = fail

    def put_object(self, **kwargs):
        if self._fail:
            raise RuntimeError("simulated S3 outage")
        self.calls.append(kwargs)
        return {"ETag": '"deadbeef"'}


class TestSaveTeamAccuracy:
    def test_writes_fixed_key(self, populated_db):
        team_accuracy = analyze_team_performance(populated_db, as_of_date=date(2026, 5, 23))
        client = _StubS3Client()
        save_team_accuracy(team_accuracy, s3_client=client, bucket="bkt")
        assert len(client.calls) == 1
        call = client.calls[0]
        assert call["Key"] == TEAM_ACCURACY_S3_KEY == "config/team_accuracy.json"
        assert call["Bucket"] == "bkt"
        payload = json.loads(call["Body"])
        assert payload == team_accuracy

    def test_zero_data_write_is_envelope_never_bare_braces(self, empty_db):
        """config#1844's literal defect: the live artifact was 2 bytes (`{}`).
        The zero-data write must be the full insufficient envelope."""
        team_accuracy = analyze_team_performance(empty_db, as_of_date=date(2026, 5, 23))
        client = _StubS3Client()
        save_team_accuracy(team_accuracy, s3_client=client, bucket="bkt")
        payload = json.loads(client.calls[0]["Body"])
        assert payload != {}
        assert payload["status"] == "insufficient"
        assert payload["n_advance_picks"] == 0
        assert payload["n_resolved_outcomes"] == 0
        assert payload["teams"] == {}

    def test_bare_legacy_dict_rejected(self):
        """The write chokepoint refuses the pre-envelope shape outright."""
        client = _StubS3Client()
        with pytest.raises(ValueError, match="schema_version-1 envelope"):
            save_team_accuracy(
                {"technology": {"accuracy": 0.5, "n_obs": 10}},
                s3_client=client,
                bucket="bkt",
            )
        assert client.calls == []

    def test_custom_key_honored(self, populated_db):
        team_accuracy = analyze_team_performance(populated_db, as_of_date=date(2026, 5, 23))
        client = _StubS3Client()
        save_team_accuracy(team_accuracy, s3_client=client, bucket="bkt", key="custom/path.json")
        assert client.calls[0]["Key"] == "custom/path.json"

    def test_empty_bucket_raises(self, populated_db):
        team_accuracy = analyze_team_performance(populated_db, as_of_date=date(2026, 5, 23))
        client = _StubS3Client()
        with pytest.raises(ValueError, match="non-empty bucket"):
            save_team_accuracy(team_accuracy, s3_client=client, bucket="")

    def test_s3_failure_propagates_per_no_silent_fails(self, populated_db):
        # Producer-side path must raise, not swallow — the Lambda handler's
        # shadow-mode wrapper (_maybe_emit_team_accuracy) is responsible for
        # catching this and logging WARN-not-fatal.
        team_accuracy = analyze_team_performance(populated_db, as_of_date=date(2026, 5, 23))
        client = _StubS3Client(fail=True)
        with pytest.raises(RuntimeError, match="simulated S3 outage"):
            save_team_accuracy(team_accuracy, s3_client=client, bucket="bkt")


class TestEnvelopeSchemaContract:
    """Pins the schema_version-1 envelope (config#1844) — the producer/
    consumer contract for ``config/team_accuracy.json``. Mirrors the repo's
    schema-contract pattern (cf. alpha-engine-data's test_schema_contract):
    key set, types, and value domains are asserted exactly so any shape
    drift fails HERE, not as a silent consumer starvation.
    """

    _ENVELOPE_KEYS = {
        "schema_version",
        "status",
        "as_of",
        "n_teams",
        "n_advance_picks",
        "n_resolved_outcomes",
        "horizon_days",
        "teams",
    }

    @pytest.mark.parametrize("fixture_name", ["empty_db", "populated_db"])
    def test_envelope_keys_and_types(self, fixture_name, request):
        conn = request.getfixturevalue(fixture_name)
        result = analyze_team_performance(conn, as_of_date=date(2026, 5, 23))
        assert set(result.keys()) == self._ENVELOPE_KEYS
        assert result["schema_version"] == 1
        assert result["status"] in ("ok", "insufficient")
        assert date.fromisoformat(result["as_of"])  # parseable ISO date
        for count_key in ("n_teams", "n_advance_picks", "n_resolved_outcomes"):
            assert isinstance(result[count_key], int) and result[count_key] >= 0
        assert result["horizon_days"] == _PRIMARY_HORIZON
        assert isinstance(result["teams"], dict)
        assert result["n_teams"] == len(result["teams"])
        assert (result["status"] == "ok") == bool(result["teams"])
        for team_id, entry in result["teams"].items():
            assert isinstance(team_id, str)
            assert set(entry.keys()) == {"accuracy", "n_obs"}
            assert isinstance(entry["accuracy"], float)
            assert 0.0 <= entry["accuracy"] <= 1.0
            assert isinstance(entry["n_obs"], int)
            assert entry["n_obs"] > 0

    def test_consumer_accuracy_adjustment_integration(self, populated_db):
        from agents.sector_teams.team_config import _accuracy_adjustment

        result = analyze_team_performance(populated_db, as_of_date=date(2026, 5, 23))
        # n_obs=3 is below ADAPTIVE_SLOT_MIN_OBS=8 -> no nudge yet, but the
        # loader/consumer contract round-trips without error.
        assert _accuracy_adjustment(result["teams"].get("technology")) == 0


class TestLoadTeamAccuracyDualShape:
    """`archive/manager.py::load_team_accuracy` must parse BOTH shapes during
    the 1-week dual-shape window (S3 contract-safety rule): the new envelope
    AND the legacy bare per-team dict still live in S3 until the enveloped
    producer's first overwrite. Legacy branch removal: after 2026-07-13.
    """

    @staticmethod
    def _manager_returning(raw):
        from archive.manager import ArchiveManager

        am = ArchiveManager.__new__(ArchiveManager)  # skip boto3 __init__
        am._s3_get = lambda key: raw
        return am

    def test_envelope_shape_unwrapped_to_teams(self, populated_db):
        envelope = analyze_team_performance(populated_db, as_of_date=date(2026, 5, 23))
        am = self._manager_returning(json.dumps(envelope))
        assert am.load_team_accuracy() == envelope["teams"]

    def test_legacy_bare_dict_passthrough(self):
        legacy = {"technology": {"accuracy": 0.6, "n_obs": 12}}
        am = self._manager_returning(json.dumps(legacy))
        assert am.load_team_accuracy() == legacy

    def test_legacy_empty_braces_the_live_defect_artifact(self):
        # The exact 2-byte live object config#1844 found: parses as the
        # legacy shape, degrades to "no accuracy data" rather than erroring.
        am = self._manager_returning("{}")
        result = am.load_team_accuracy()
        assert result in ({}, None)

    def test_insufficient_envelope_yields_empty_teams(self, empty_db):
        envelope = analyze_team_performance(empty_db, as_of_date=date(2026, 5, 23))
        am = self._manager_returning(json.dumps(envelope))
        assert am.load_team_accuracy() == {}

    def test_malformed_envelope_teams_returns_none(self):
        am = self._manager_returning(
            json.dumps({"schema_version": 1, "teams": "not-a-dict"})
        )
        assert am.load_team_accuracy() is None

    def test_missing_artifact_returns_none(self):
        am = self._manager_returning(None)
        assert am.load_team_accuracy() is None

    def test_unparseable_returns_none(self):
        am = self._manager_returning("not json {{{")
        assert am.load_team_accuracy() is None

    def test_consumer_chain_round_trip_both_shapes(self, populated_db):
        """End-to-end: producer envelope -> loader -> compute_team_slots
        accepts the unwrapped mapping (same for a legacy artifact)."""
        from agents.sector_teams.team_config import compute_team_slots

        envelope = analyze_team_performance(populated_db, as_of_date=date(2026, 5, 23))
        for raw in (json.dumps(envelope), json.dumps(envelope["teams"])):
            am = self._manager_returning(raw)
            teams = am.load_team_accuracy()
            allocation = compute_team_slots(6, {}, team_accuracy=teams)
            assert set(allocation) >= {"technology", "healthcare"}
