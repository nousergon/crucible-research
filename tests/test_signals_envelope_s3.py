"""moto-S3 tests for ``scoring/signals_envelope.py`` — read/write + CLI paths.

Covers:
  * universe board present/absent (absent -> RAISE loud, no-silent-fails)
  * regime substrate present/absent (absent -> fail-soft "neutral" + WARN,
    the one documented exception in this module)
  * --target shadow vs --target production S3 key routing
  * production target refuses to run without --i-know-this-is-production
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boto3
import pytest
from moto import mock_aws

from scoring.signals_envelope import (
    _build_arg_parser,
    build_signals_envelope,
    main as envelope_main,
    read_regime_substrate,
    read_universe_board,
    write_envelope,
)

BUCKET = "alpha-engine-research"


@pytest.fixture
def mocked_s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def _sample_board() -> dict:
    return {
        "schema_version": 3,
        "as_of": "2026-07-14",
        "stocks": [
            {
                "ticker": "AAA",
                "sector": "Technology",
                "attractiveness_score": 77.0,
                "pillars": {"quality": 60.0},
            },
            {
                "ticker": "BBB",
                "sector": "Healthcare",
                "attractiveness_score": 55.0,
                "pillars": {"quality": 40.0},
            },
        ],
    }


def _put_board(client, run_date: str, board: dict, *, dated_only: bool = False) -> None:
    body = json.dumps(board).encode("utf-8")
    if not dated_only:
        client.put_object(Bucket=BUCKET, Key="scanner/universe/latest.json", Body=body)
    client.put_object(
        Bucket=BUCKET, Key=f"scanner/universe/{run_date}/universe.json", Body=body,
    )


def _put_regime_substrate(client, intensity_z: float) -> None:
    # Mirrors nousergon_lib.eval_artifacts canonical sidecar->artifact
    # resolution: latest.json is a pointer to the dated artifact key.
    payload = {"composite": {"intensity_z": intensity_z}, "run_id": "20260713-0900"}
    dated_key = "regime/20260713-0900.json"
    client.put_object(Bucket=BUCKET, Key=dated_key, Body=json.dumps(payload).encode("utf-8"))
    client.put_object(
        Bucket=BUCKET, Key="regime/latest.json",
        Body=json.dumps({"artifact_key": dated_key}).encode("utf-8"),
    )


# ── read_universe_board ──────────────────────────────────────────────────────


class TestReadUniverseBoard:
    def test_reads_dated_key(self, mocked_s3):
        _put_board(mocked_s3, "2026-07-14", _sample_board())
        board = read_universe_board(BUCKET, run_date="2026-07-14", s3_client=mocked_s3)
        assert len(board["stocks"]) == 2

    def test_falls_back_to_latest_sidecar(self, mocked_s3):
        # Dated key for a DIFFERENT date only; latest.json present.
        body = json.dumps(_sample_board()).encode("utf-8")
        mocked_s3.put_object(Bucket=BUCKET, Key="scanner/universe/latest.json", Body=body)
        board = read_universe_board(BUCKET, run_date="2026-07-15", s3_client=mocked_s3)
        assert len(board["stocks"]) == 2

    def test_absent_board_raises_loud(self, mocked_s3):
        with pytest.raises(RuntimeError, match="no scanner universe board found"):
            read_universe_board(BUCKET, run_date="2026-07-14", s3_client=mocked_s3)

    def test_stale_latest_fallback_raises(self, mocked_s3):
        # I2880: dated board absent; latest.json is from a PRIOR weekly cycle
        # (as_of weeks old). Must fail loud, not silently trade a stale universe.
        stale = _sample_board()
        stale["as_of"] = "2026-06-01"
        body = json.dumps(stale).encode("utf-8")
        mocked_s3.put_object(
            Bucket=BUCKET, Key="scanner/universe/latest.json", Body=body
        )
        with pytest.raises(RuntimeError, match="trading days stale"):
            read_universe_board(BUCKET, run_date="2026-07-15", s3_client=mocked_s3)

    def test_latest_fallback_missing_as_of_raises(self, mocked_s3):
        # I2880: an unverifiable fallback (no as_of) must not be trusted.
        board = _sample_board()
        board.pop("as_of")
        body = json.dumps(board).encode("utf-8")
        mocked_s3.put_object(
            Bucket=BUCKET, Key="scanner/universe/latest.json", Body=body
        )
        with pytest.raises(RuntimeError, match="carries no"):
            read_universe_board(BUCKET, run_date="2026-07-15", s3_client=mocked_s3)

    def test_stale_latest_fallback_preflight_warns_not_raises(
        self, mocked_s3, caplog,
    ):
        # config-I2916: on the Friday-PM shell run the Scanner runs DRY, so the
        # dated board is intentionally absent and the fallback is ALWAYS the
        # prior-Saturday board (~5 trading days stale by Friday). That is an
        # EXPECTED artefact of the preflight contract, not a real scanner miss:
        # preflight=True must WARN + RETURN the board, NOT raise, so the
        # preflight completes its bootstrap/transport smoke.
        stale = _sample_board()
        stale["as_of"] = "2026-06-01"
        body = json.dumps(stale).encode("utf-8")
        mocked_s3.put_object(
            Bucket=BUCKET, Key="scanner/universe/latest.json", Body=body
        )
        import logging

        with caplog.at_level(logging.WARNING):
            board = read_universe_board(
                BUCKET, run_date="2026-07-15", s3_client=mocked_s3,
                preflight=True,
            )
        assert len(board["stocks"]) == 2
        assert any(
            "PREFLIGHT" in r.message and "stale universe-board" in r.message
            for r in caplog.records
        ), "preflight staleness tolerance must emit a WARN, not swallow silently"

    def test_stale_latest_fallback_real_run_still_raises_with_preflight_false(
        self, mocked_s3,
    ):
        # Regression companion to the preflight case: on the REAL Saturday run
        # (preflight=False, the default) the I2880 staleness bound stays fully
        # in force — a genuinely stale fallback must still hard-fail so no
        # prior-cycle universe is ever silently traded.
        stale = _sample_board()
        stale["as_of"] = "2026-06-01"
        body = json.dumps(stale).encode("utf-8")
        mocked_s3.put_object(
            Bucket=BUCKET, Key="scanner/universe/latest.json", Body=body
        )
        with pytest.raises(RuntimeError, match="trading days stale"):
            read_universe_board(
                BUCKET, run_date="2026-07-15", s3_client=mocked_s3,
                preflight=False,
            )

    def test_preflight_still_raises_on_missing_as_of(self, mocked_s3):
        # config-I2916: preflight relaxes ONLY the staleness bound. A fallback
        # with no ``as_of`` (an unverifiable / possibly corrupt board) is a real
        # integrity fault the smoke is meant to surface — it must still raise
        # even under preflight.
        board = _sample_board()
        board.pop("as_of")
        body = json.dumps(board).encode("utf-8")
        mocked_s3.put_object(
            Bucket=BUCKET, Key="scanner/universe/latest.json", Body=body
        )
        with pytest.raises(RuntimeError, match="carries no"):
            read_universe_board(
                BUCKET, run_date="2026-07-15", s3_client=mocked_s3,
                preflight=True,
            )

    def test_preflight_true_with_fresh_dated_board_is_noop(self, mocked_s3):
        # config-I2916: preflight must not change behaviour when a fresh dated
        # board exists (the guard is only reached on the latest.json fallback);
        # the dated key is read directly and the staleness path never runs.
        _put_board(mocked_s3, "2026-07-14", _sample_board())
        board = read_universe_board(
            BUCKET, run_date="2026-07-14", s3_client=mocked_s3, preflight=True,
        )
        assert len(board["stocks"]) == 2

    def test_preflight_still_raises_when_both_keys_absent(self, mocked_s3):
        # config-I2916: preflight is NOT a blanket bypass — a genuinely absent
        # board (BOTH the dated key and latest.json missing) is a real
        # bootstrap/transport failure that must still raise loud even on the
        # preflight (its whole point is surfacing exactly this).
        with pytest.raises(RuntimeError, match="no scanner universe board found"):
            read_universe_board(
                BUCKET, run_date="2026-07-14", s3_client=mocked_s3,
                preflight=True,
            )


# ── read_regime_substrate ────────────────────────────────────────────────────


class TestReadRegimeSubstrate:
    def test_reads_substrate_when_present(self, mocked_s3):
        _put_regime_substrate(mocked_s3, 0.8)
        substrate = read_regime_substrate(BUCKET, s3_client=mocked_s3)
        assert substrate is not None
        assert substrate["composite"]["intensity_z"] == 0.8

    def test_absent_substrate_returns_none_not_raise(self, mocked_s3, caplog):
        substrate = read_regime_substrate(BUCKET, s3_client=mocked_s3)
        assert substrate is None


# ── End-to-end build + write ─────────────────────────────────────────────────


class TestWriteEnvelope:
    def test_shadow_target_writes_shadow_keys_only(self, mocked_s3):
        _put_board(mocked_s3, "2026-07-14", _sample_board())
        envelope = build_signals_envelope("2026-07-14", _sample_board(), None)
        dated_key, latest_key = write_envelope(
            envelope, "2026-07-14", target="shadow", bucket=BUCKET, s3_client=mocked_s3,
        )
        assert dated_key == "signals_envelope/2026-07-14/signals.json"
        assert latest_key == "signals_envelope/latest.json"
        # Never touches the live signals/ prefix.
        with pytest.raises(mocked_s3.exceptions.NoSuchKey):
            mocked_s3.get_object(Bucket=BUCKET, Key="signals/2026-07-14/signals.json")

    def test_production_target_writes_live_keys(self, mocked_s3):
        envelope = build_signals_envelope("2026-07-14", _sample_board(), None)
        dated_key, latest_key = write_envelope(
            envelope, "2026-07-14", target="production", bucket=BUCKET, s3_client=mocked_s3,
        )
        assert dated_key == "signals/2026-07-14/signals.json"
        assert latest_key == "signals/latest.json"
        obj = mocked_s3.get_object(Bucket=BUCKET, Key=dated_key)
        roundtripped = json.loads(obj["Body"].read())
        assert roundtripped["universe"][0]["ticker"] == "AAA"


# ── CLI ──────────────────────────────────────────────────────────────────────


class TestCli:
    def test_production_without_ack_flag_errors(self, mocked_s3, capsys):
        with pytest.raises(SystemExit) as exc_info:
            envelope_main(["--target", "production", "--bucket", BUCKET, "--date", "2026-07-14"])
        assert exc_info.value.code != 0
        captured = capsys.readouterr()
        assert "--i-know-this-is-production" in captured.err

    def test_shadow_end_to_end_via_cli(self, mocked_s3, capsys, monkeypatch):
        _put_board(mocked_s3, "2026-07-14", _sample_board())
        monkeypatch.setattr(boto3, "client", lambda *a, **kw: mocked_s3)
        rc = envelope_main(["--bucket", BUCKET, "--date", "2026-07-14"])
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["target"] == "shadow"
        assert out["universe_count"] == 2
        assert out["market_regime"] == "bear"  # I2881: no substrate → fail-safe bear

    def test_production_with_ack_flag_writes_live_key(self, mocked_s3, capsys, monkeypatch):
        _put_board(mocked_s3, "2026-07-14", _sample_board())
        monkeypatch.setattr(boto3, "client", lambda *a, **kw: mocked_s3)
        rc = envelope_main([
            "--bucket", BUCKET, "--date", "2026-07-14",
            "--target", "production", "--i-know-this-is-production",
        ])
        assert rc == 0
        obj = mocked_s3.get_object(Bucket=BUCKET, Key="signals/2026-07-14/signals.json")
        payload = json.loads(obj["Body"].read())
        assert payload["universe"]

    def test_arg_parser_defaults_to_shadow(self):
        parser = _build_arg_parser()
        args = parser.parse_args([])
        assert args.target == "shadow"
        assert args.ack_production is False
