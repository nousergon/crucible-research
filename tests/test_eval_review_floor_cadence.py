"""The judged-corpus floor reads each judge tier on its own cadence, and only
over criteria the rubric declares.

Measured 2026-10-07 (nous-ergon-ops "CloudWatch alarm age" red on
``alpha-engine-eval-review-floor``, latched since 2026-09-12):

* the Sonnet tier is a MONTHLY sweep (``force_sonnet_pass`` on the first
  Saturday) plus an escalation tail, and ``thinktank_thesis/*/sonnet`` read
  72 / 17 / 0 / 0 / 36 reviews in slots 2957..2961 — under a one-week floor
  that is a breach most weeks with nothing wrong;
* the judge invented a fifth theme dimension, ``dimension_note``, once on
  2026-10-03, and the emitter published it as a new combo the floor then
  counted as starved.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from evals import control_bands as cb
from evals import rubric_dimensions as rd
from evals.judge_models import HAIKU, SONNET

BASE = datetime(2026, 6, 9, tzinfo=UTC)
REPO = Path(__file__).resolve().parent.parent


def _dims(agent: str, criterion: str, judge: str) -> list[dict]:
    return [
        {"Name": "judged_agent_id", "Value": agent},
        {"Name": "criterion", "Value": criterion},
        {"Name": "judge_model", "Value": judge},
    ]


def _cw_for(combos, reviews):
    """Weekly buckets, the newest closing at ``BASE``; ``reviews[idx]`` is
    oldest-first and every bucket's mean is 4.0."""
    cw = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {"Metrics": [{"Dimensions": d} for d in combos]},
    ]
    cw.get_paginator.return_value = paginator
    results = []
    for idx in range(len(combos)):
        counts = reviews[idx]
        n = len(counts)
        ts = [BASE - timedelta(weeks=(n - i)) for i in range(n)]
        results.append({"Id": f"m{idx}", "Timestamps": ts, "Values": [4.0] * n})
        results.append({
            "Id": f"n{idx}", "Timestamps": ts,
            "Values": [float(c) for c in counts],
        })
    cw.get_metric_data.return_value = {"MetricDataResults": results}
    return cw


def _run(combos, reviews, monkeypatch, declared=None):
    monkeypatch.setattr(
        cb, "declared_criteria_for_agent",
        lambda agent: (declared or {}).get(agent),
    )
    cw = _cw_for(combos, reviews)
    out = cb.compute_and_emit_control_bands(
        end_time=BASE, cloudwatch_client=cw, s3_client=MagicMock(),
    )
    return out


class TestJudgeCadenceWindow:
    def test_sonnet_between_sweeps_is_not_a_breach(self, monkeypatch):
        # The 2026-09-17..10-01 shape: a sweep three weeks back, then 0, 0, 0.
        combos = [_dims("a", "c1", SONNET.logical_key)]
        out = _run(combos, {0: [0, 0, 36, 0, 0, 0]}, monkeypatch)
        assert out["review_floor_breach_count"] == 0, out["review_floor_breaches"]

    def test_sonnet_with_no_sweep_in_five_weeks_is_a_breach(self, monkeypatch):
        # A first-Saturday run recovered on day 8+ never sets
        # force_sonnet_pass: the month's sweep did not happen.
        combos = [_dims("a", "c1", SONNET.logical_key)]
        out = _run(combos, {0: [50, 0, 1, 0, 2, 0]}, monkeypatch)
        assert out["review_floor_breaches"] == [
            f"a/c1/{SONNET.logical_key}=3/5w"
        ]

    def test_a_weekly_tier_keeps_the_one_week_floor(self, monkeypatch):
        # The failure the floor was written from — a week nobody judged —
        # is still caught on the weekly tier.
        combos = [_dims("a", "c1", HAIKU.logical_key)]
        out = _run(combos, {0: [50, 50, 50, 50, 50, 0]}, monkeypatch)
        assert out["review_floor_breaches"] == [f"a/c1/{HAIKU.logical_key}=0"]

    def test_the_sonnet_window_covers_the_longest_gap_between_sweeps(self):
        # First Saturdays 2026-08-01, 09-05, 10-03, 11-07: 5, 4, 5 weeks.
        firsts = [datetime(2026, m, d, tzinfo=UTC)
                  for m, d in ((8, 1), (9, 5), (10, 3), (11, 7))]
        longest = max((b - a).days // 7 for a, b in zip(firsts, firsts[1:], strict=False))
        assert cb.FLOOR_WINDOW_WEEKS_BY_JUDGE[SONNET.logical_key] >= longest
        assert HAIKU.logical_key not in cb.FLOOR_WINDOW_WEEKS_BY_JUDGE


class TestUndeclaredCriterion:
    def test_a_phantom_criterion_is_left_out_of_the_floor(self, monkeypatch):
        combos = [
            _dims("thinktank_theme", "actionability", HAIKU.logical_key),
            _dims("thinktank_theme", "dimension_note", HAIKU.logical_key),
        ]
        out = _run(
            combos, {0: [50] * 6, 1: [0, 0, 0, 0, 0, 1]}, monkeypatch,
            declared={"thinktank_theme": ("actionability",)},
        )
        assert out["review_floor_breach_count"] == 0, out["review_floor_breaches"]
        assert out["review_floor_excluded_undeclared"] == [
            f"thinktank_theme/dimension_note/{HAIKU.logical_key}"
        ]

    def test_an_unknown_rubric_keeps_every_combo(self, monkeypatch):
        combos = [_dims("a", "anything", HAIKU.logical_key)]
        out = _run(combos, {0: [0, 0, 0, 0, 0, 1]}, monkeypatch, declared={})
        assert out["review_floor_breach_count"] == 1
        assert out["review_floor_excluded_undeclared"] == []


class TestRubricDimensions:
    @pytest.mark.parametrize("name, expected", [
        ("eval_rubric_thinktank_theme", (
            "grounding_in_inputs", "churn_discipline", "anchor_fidelity",
            "actionability",
        )),
        ("eval_rubric_thinktank_thesis", (
            "input_groundedness", "moat_and_business_quality",
            "valuation_linkage", "risk_specificity", "context_integration",
            "stance_consistency",
        )),
    ])
    def test_reads_the_headers_of_the_shipped_rubrics(self, name, expected):
        text = (REPO / "config" / "prompts.example" / f"{name}.txt").read_text()
        assert rd.declared_dimensions(text) == expected

    def test_no_header_is_unknown_not_empty_rubric(self):
        assert rd.declared_dimensions("score each dimension 1-5") == ()

    def _scores(self, *names):
        return [SimpleNamespace(dimension=n, score=4) for n in names]

    def test_drops_the_2026_10_03_phantom(self, caplog):
        scores = self._scores(
            "grounding_in_inputs", "churn_discipline", "anchor_fidelity",
            "actionability", "dimension_note",
        )
        kept = rd.keep_declared_dimension_scores(
            scores,
            ("grounding_in_inputs", "churn_discipline", "anchor_fidelity",
             "actionability"),
            agent_id="thinktank_theme", judge_model=HAIKU.logical_key,
            rubric_id="eval_rubric_thinktank_theme",
        )
        assert [s.dimension for s in kept] == [
            "grounding_in_inputs", "churn_discipline", "anchor_fidelity",
            "actionability",
        ]
        assert "dimension_note" in caplog.text

    def test_unknown_declaration_keeps_everything(self):
        scores = self._scores("a", "b")
        kept = rd.keep_declared_dimension_scores(
            scores, (), agent_id="x", judge_model="j", rubric_id="r",
        )
        assert kept == scores

    def test_unmapped_agent_is_unknown(self):
        assert rd.declared_criteria_for_agent("executor:anything") is None

    def test_nothing_declared_matched_keeps_the_eval(self, caplog):
        # A rubric/judge mismatch is not a phantom: dropping every score
        # would silently erase the eval.
        scores = self._scores("dim_0", "dim_1")
        kept = rd.keep_declared_dimension_scores(
            scores, ("actionability",), agent_id="x", judge_model="j",
            rubric_id="r",
        )
        assert kept == scores
        assert "mismatch" in caplog.text
