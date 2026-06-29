"""Tests for the factor-blend regime-weights S3 override (config#748).

config.get_factor_blend_regime_weights() layers an S3-written
config/factor_blend_params.json override on top of the scoring.yaml
aggregator.factor_blend regime weights (YAML wins only when the S3 key is
absent). Backtester's factor_blend_optimizer writes that override.

S3 is faked; no boto3 network, no real bucket.
"""

from __future__ import annotations

import io
import json

import pytest

import config


@pytest.fixture(autouse=True)
def _reset_cache(monkeypatch, tmp_path):
    # Force a fresh resolve each test + isolate the local cache file.
    monkeypatch.setattr(config, "_factor_blend_regime_cache", None)
    monkeypatch.setattr(
        config, "_FACTOR_BLEND_PARAMS_CACHE_PATH", str(tmp_path / "fb_cache.json")
    )
    yield
    monkeypatch.setattr(config, "_factor_blend_regime_cache", None)


class _FakeS3:
    def __init__(self, body: bytes | None):
        self._body = body

    def get_object(self, Bucket, Key):  # noqa: N803
        if self._body is None:
            raise Exception("NoSuchKey: simulated absent key")
        return {"Body": io.BytesIO(self._body)}


def _patch_s3(monkeypatch, body: bytes | None):
    import boto3

    monkeypatch.setattr(boto3, "client", lambda svc, *a, **k: _FakeS3(body))


def test_absent_s3_key_falls_back_to_yaml(monkeypatch):
    _patch_s3(monkeypatch, None)
    out = config.get_factor_blend_regime_weights()
    # Equals the scoring.yaml-derived defaults.
    assert out == {
        regime: dict(w) for regime, w in config._FB_REGIME_DEFAULTS.items()
    }


def test_s3_override_replaces_named_regime(monkeypatch):
    override = {
        "regime_weights": {
            "bull": {
                "quality_score": 0.40, "momentum_score": 0.30,
                "value_score": 0.20, "low_vol_score": -0.10,
            }
        },
        "updated_at": "2026-06-29",
        "source": "factor_blend_optimizer",
    }
    _patch_s3(monkeypatch, json.dumps(override).encode())
    out = config.get_factor_blend_regime_weights()
    # bull replaced by the override...
    assert out["bull"]["quality_score"] == 0.40
    assert out["bull"]["momentum_score"] == 0.30
    # ...regimes absent from the override keep their YAML defaults.
    assert out["bear"] == config._FB_REGIME_DEFAULTS["bear"]
    assert out["neutral"] == config._FB_REGIME_DEFAULTS["neutral"]


def test_result_is_cached(monkeypatch):
    _patch_s3(monkeypatch, None)
    first = config.get_factor_blend_regime_weights()
    # Even if S3 would now return something, the cached value is returned.
    _patch_s3(monkeypatch, json.dumps({"regime_weights": {"bull": {"x": 1.0}}}).encode())
    second = config.get_factor_blend_regime_weights()
    assert first == second  # cache short-circuits the second resolve


def test_empty_override_ignored(monkeypatch):
    _patch_s3(monkeypatch, json.dumps({"regime_weights": {}}).encode())
    out = config.get_factor_blend_regime_weights()
    assert out == {
        regime: dict(w) for regime, w in config._FB_REGIME_DEFAULTS.items()
    }
