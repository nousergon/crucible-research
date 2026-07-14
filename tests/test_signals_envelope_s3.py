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
        assert out["market_regime"] == "neutral"  # no substrate present

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
