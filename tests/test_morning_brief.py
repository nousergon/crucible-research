"""Unit tests for scoring/morning_brief.py (config-I3290 port of the retired
multi-agent consolidated_report path)."""

from __future__ import annotations

from unittest.mock import MagicMock

import boto3
from moto import mock_aws

from scoring.morning_brief import build_morning_brief_markdown, write_morning_brief

BUCKET = "alpha-engine-research"


def _envelope(**overrides) -> dict:
    base = {
        "run_date": "2026-07-25",
        "date": "2026-07-25",
        "market_regime": "bull",
        "sector_ratings": {
            "Technology": {"rating": "market_weight", "modifier": 1.0},
            "Healthcare": {"rating": "market_weight", "modifier": 1.0},
        },
        "universe": [
            {"ticker": "AAPL", "sector": "Technology", "score": 82.0},
            {"ticker": "JNJ", "sector": "Healthcare", "score": 55.0},
            {"ticker": "XOM", "sector": "Energy", "score": 30.0},
        ],
    }
    base.update(overrides)
    return base


class TestBuildMorningBriefMarkdown:
    def test_includes_run_date_and_regime(self):
        md = build_morning_brief_markdown(_envelope())
        assert "2026-07-25" in md
        assert "Bullish" in md

    def test_unknown_regime_falls_back_to_raw_label(self):
        md = build_morning_brief_markdown(_envelope(market_regime="weird"))
        assert "weird" in md

    def test_top_scores_sorted_descending(self):
        md = build_morning_brief_markdown(_envelope())
        top_section = md.split("## Top attractiveness scores")[1]
        aapl_idx = top_section.index("AAPL")
        jnj_idx = top_section.index("JNJ")
        assert aapl_idx < jnj_idx

    def test_entries_missing_score_are_excluded(self):
        env = _envelope(universe=[
            {"ticker": "AAPL", "sector": "Technology", "score": 82.0},
            {"ticker": "NOSCORE", "sector": "Technology", "score": None},
        ])
        md = build_morning_brief_markdown(env)
        assert "NOSCORE" not in md

    def test_empty_universe_renders_none_placeholders(self):
        md = build_morning_brief_markdown(_envelope(universe=[], sector_ratings={}))
        assert "_none this cycle_" in md
        assert "_no sector data this cycle_" in md

    def test_no_llm_narrative_disclosure_present(self):
        md = build_morning_brief_markdown(_envelope())
        assert "config#2515" in md

    def test_is_pure_function_no_mutation(self):
        env = _envelope()
        universe_before = list(env["universe"])
        build_morning_brief_markdown(env)
        assert env["universe"] == universe_before


class TestWriteMorningBrief:
    def test_writes_expected_key_and_body(self):
        s3 = MagicMock()
        key = write_morning_brief(
            "2026-07-25", "# brief", bucket="alpha-engine-research", s3_client=s3,
        )
        assert key == "consolidated/2026-07-25/morning.md"
        s3.put_object.assert_called_once()
        call = s3.put_object.call_args
        assert call.kwargs["Bucket"] == "alpha-engine-research"
        assert call.kwargs["Key"] == "consolidated/2026-07-25/morning.md"
        assert call.kwargs["Body"] == b"# brief"
        assert call.kwargs["ContentType"] == "text/markdown"

    def test_uses_default_client_when_none_provided(self):
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=BUCKET)
            key = write_morning_brief("2026-07-25", "# brief", bucket=BUCKET)
            body = client.get_object(Bucket=BUCKET, Key=key)["Body"].read()
            assert body == b"# brief"
