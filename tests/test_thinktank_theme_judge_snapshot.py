"""The theme judge sees every input the theme keeper saw.

alpha-engine-config-I12403. ``evals/judge.py`` builds the judge's input from
``input_data_snapshot`` and nothing else. The theme snapshot used to carry
kind, key, update_reason, regime, anchor date and prior theme only. The macro
anchor block, the sector rating, the house macro theme and the intraweek
developments were in the prompt but not in the snapshot.

Measured on the 2026-09-25, 10-02 and 10-09 theme artifacts: 128-149 of the
129-150 new figures a week's themes cited were in their prompts, and 1-8 were
visible to the judge. The judge scored those figures as fabricated. That drove
the 2026-10-17 ``thinktank_theme`` control breach on ``grounding_in_inputs``
and ``churn_discipline``.

The invariant pinned here does not depend on any one prompt's field list:
on every theme path (seed, reconcile, daily update), every value the prompt
template is formatted with is present in the snapshot. A future prompt input
that is not recorded fails this test.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from thinktank import themes
from thinktank.schemas import ThemeThesis, ThemeThesisLLM

#: Format kwargs that are not inputs the judge must see. ``mode`` and
#: ``sector`` are in the snapshot as ``update_reason`` and ``key``.
#: ``macro_report`` is the legacy alias of ``macro_anchor`` (``_anchor_kwargs``).
#: ``prior_theme`` is recorded as a dict, not as the JSON string the prompt
#: gets. It is compared separately below.
_NOT_JUDGE_INPUTS = {"mode", "sector", "macro_report", "prior_theme"}

ANCHOR = "QUANTITATIVE REGIME SUBSTRATE: HY OAS 324 bps (z +0.32), VIX 16.39"
MACRO_NARRATIVE = "macro view: credit crossed the 300 bps trigger"


class _Prompt:
    def __init__(self, name: str, calls: list[dict]) -> None:
        self.name = name
        self.version = "test"
        self.hash = "h"
        self._calls = calls

    def format(self, **kwargs) -> str:
        self._calls.append({"prompt": self.name, **kwargs})
        return "\n".join(f"{k}={v}" for k, v in kwargs.items())


class _Client:
    run_id = "run-test"

    def complete(self, tier, **kwargs):
        return SimpleNamespace(
            parsed=ThemeThesisLLM(
                narrative="n", stance="neutral", drivers=["d"], watch_items=["w"],
                material_change=True, change_summary="c",
            ),
            model="m", tier=tier, cost_usd=0.0,
        )


def _theme(kind: str, key: str, *, version: int = 3, narrative: str = "prior") -> ThemeThesis:
    return ThemeThesis(
        kind=kind, key=key, version=version, trading_day="2026-10-01",
        calendar_date="2026-10-01", update_reason="reconcile",
        theme=ThemeThesisLLM(
            narrative=narrative, stance="neutral", drivers=["d0"], watch_items=["w0"],
            material_change=False,
        ),
        weekly_anchor_date="2026-09-25",
    )


@pytest.fixture
def harness(monkeypatch):
    calls: list[dict] = []
    captures: list[dict] = []
    written: dict[tuple[str, str], ThemeThesis] = {}

    monkeypatch.setattr(themes, "load_prompt", lambda name: _Prompt(name, calls))
    monkeypatch.setattr(themes, "emit_theme_capture", lambda **kw: captures.append(kw))

    def _write(store, theme):
        written[(theme.kind, theme.key)] = theme

    monkeypatch.setattr(themes, "_write_theme", _write)

    def _load(store, kind, key):
        if (kind, key) in written:
            return written[(kind, key)]
        if kind == "macro":
            return _theme("macro", "macro", narrative=MACRO_NARRATIVE)
        return _theme("sector", key)

    monkeypatch.setattr(themes, "load_theme", _load)
    monkeypatch.setattr(themes.ThemeKeeper, "_macro_anchor_block", lambda self: ANCHOR)

    ctx = SimpleNamespace(
        market_regime=lambda: "neutral",
        sector_ratings=lambda: {
            "Energy": {"rating": "market_weight", "modifier": 1.0},
            "Utilities": {"rating": "overweight", "modifier": 1.1},
        },
        stale_input_records=lambda: [],
        weekly_signals_date=lambda: "2026-10-02",
    )
    keeper = themes.ThemeKeeper(
        SimpleNamespace(bucket="b", s3=None), _Client(), ctx,
        trading_day="2026-10-05", calendar_date="2026-10-05",
    )
    return SimpleNamespace(keeper=keeper, calls=calls, captures=captures, written=written)


def _assert_every_prompt_input_is_in_the_snapshot(h) -> None:
    assert h.calls, "no prompt was rendered"
    assert len(h.calls) == len(h.captures), "a rendered theme was not captured"
    for call, capture in zip(h.calls, h.captures, strict=True):
        snap = capture["input_data_snapshot"]
        for name, value in call.items():
            if name in _NOT_JUDGE_INPUTS or name == "prompt":
                continue
            assert value in snap.values(), (
                f"{call['prompt']}: prompt input {name!r} is not in the judge's "
                f"input_data_snapshot (keys: {sorted(snap)})"
            )
        prior = call["prior_theme"]
        if prior.startswith("{"):
            assert snap["prior_theme"] == json.loads(prior)


def test_seed_records_every_prompt_input(harness):
    harness.keeper._seed_all("2026-10-02")
    _assert_every_prompt_input_is_in_the_snapshot(harness)


def test_reconcile_records_every_prompt_input(harness):
    harness.keeper._reconcile_all(_theme("macro", "macro"), "2026-10-02")
    _assert_every_prompt_input_is_in_the_snapshot(harness)


def test_daily_update_records_every_prompt_input(harness):
    harness.keeper._daily_update_macro(
        _theme("macro", "macro"), "NVDA guidance raised; CPI print pending"
    )
    _assert_every_prompt_input_is_in_the_snapshot(harness)


def test_reconcile_snapshot_carries_the_figures_the_judge_called_fabricated(harness):
    """The 2026-10-02 failure: sector themes cited HY OAS 324 bps from the house
    macro theme in their prompt. The judge, shown only the prior theme,
    scored that figure as invented."""
    harness.keeper._reconcile_all(_theme("macro", "macro"), "2026-10-02")
    by_key = {c["input_data_snapshot"]["key"]: c["input_data_snapshot"] for c in harness.captures}

    assert by_key["macro"]["macro_anchor"] == ANCHOR
    for sector in ("Energy", "Utilities"):
        snap = by_key[sector]
        assert "324 bps" not in json.dumps(snap["prior_theme"])
        # The house macro theme the sector saw is the one the macro reconcile
        # wrote moments earlier in the same run, not the prior one.
        assert snap["macro_theme"] == harness.keeper.macro_summary()
        assert snap["sector_rating"] == str(
            harness.keeper._ctx.sector_ratings()[sector]
        )
        assert snap["developments"] == "(reconcile to the new weekly analysis)"


def test_daily_update_snapshot_carries_the_developments(harness):
    developments = "x" * 9000
    harness.keeper._daily_update_macro(_theme("macro", "macro"), developments)
    snap = harness.captures[0]["input_data_snapshot"]
    # Exactly the truncated text the prompt got, no more and no less.
    assert snap["developments"] == developments[:8000]
