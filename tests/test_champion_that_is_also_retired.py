"""An arm that is BOTH the live pointer's champion and a retired register row
is scored ONCE (alpha-engine-config-I11426).

MEASURED LIVE 2026-09-22 against `main`, the day it started:

    build_producer_leaderboard(s3, 'alpha-engine-research', '2026-09-22')
    -> status: unmeasurable
       duplicate_arm_rows: 21d:scanner_predictor_directx2,
       126d:scanner_predictor_directx2, 252d:scanner_predictor_directx2,
       top_level:scanner_predictor_directx2

`config/producer_champion.json` has named `scanner_predictor_direct` since
2026-07-13. `crucible-research-PR819` retired its register row with
`retired_date="2026-09-22"` — and §3's trailing window opens ON the retirement
date, so from that moment the arm was emitted twice: once as the champion row
and once as a retired row. The board would have published `unmeasurable` on
every run until the pointer moved or the window closed on 2026-11-17.

`_load_producer_specs` already skipped the champion in the CHALLENGER loop, for
exactly this reason. The RETIRED loop had no such skip — not a different
judgement, just the case nobody had reached yet, because no arm had ever been
retired while still serving.

THE CHAMPION ROW WINS. It is the arm that is SERVING: read from the source the
pointer's arm publishes to, and a promotion consumer filtering on `kind` would
otherwise find the serving arm tagged "retired".

VERIFIED RED. Removing the skip and re-running this file, verbatim:

    AssertionError: {'status': 'unmeasurable', ...
      'duplicate_arm_rows ... The rows are NOT silently merged: a duplicate
      after the crucible-research#658 fix is a producer fault, not the known
      one.'}
    assert 'unmeasurable' == 'ok'

Four of the five fail — every assertion downstream of the board being
measurable at all, which is the point: one arm's double row silences every
other arm on the board.
"""

from __future__ import annotations

import json

import boto3
import pytest
from moto import mock_aws

_BUCKET = "alpha-engine-research"
# Its register row's own retired_date, so the trailing window is open.
_ARM = "scanner_predictor_direct"
_AS_OF = "2026-09-22"
_ENTRY = "2026-06-01"


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=_BUCKET)
        client.put_object(
            Bucket=_BUCKET,
            Key="config/producer_champion.json",
            Body=json.dumps({"schema_version": 1, "champion": _ARM}).encode(),
        )
        yield client


class _Panel:
    def __init__(self):
        self.panel: dict[str, dict[str, float]] = {}

    def put(self, date_str: str, closes: dict) -> _Panel:
        self.panel.setdefault(date_str, {}).update(
            {t: float(c) for t, c in closes.items()}
        )
        return self

    def loader(self):
        return lambda bucket, entry_dates, horizon_days, symbols=None: self.panel


def _seed(s3) -> _Panel:
    """The champion's LIVE signals plus a co-arm, with a matured panel."""
    signals = {
        "signals": {
            "A": {"signal": "ENTER", "score": 90.0},
            "B": {"signal": "ENTER", "score": 70.0},
        }
    }
    s3.put_object(
        Bucket=_BUCKET,
        Key=f"signals/{_ENTRY}/signals.json",
        Body=json.dumps(signals).encode(),
    )
    s3.put_object(
        Bucket=_BUCKET,
        Key=f"signals_shadow/{_ARM}/{_ENTRY}/signals.json",
        Body=json.dumps(signals).encode(),
    )
    # A co-retired arm, so the "only the champion is skipped" assertion below
    # has something to be about. It emits nothing on a per-arm-width board
    # otherwise, and would read as dropped when it was merely silent.
    s3.put_object(
        Bucket=_BUCKET,
        Key=f"signals_shadow/no_agent_quant/{_ENTRY}/signals.json",
        Body=json.dumps({
            "signals": {
                "A": {"signal": "ENTER", "score": 80.0},
                "B": {"signal": "ENTER", "score": 75.0},
            }
        }).encode(),
    )
    s3.put_object(
        Bucket=_BUCKET,
        Key=f"signals_shadow/tech_score_20/{_ENTRY}/signals.json",
        Body=json.dumps({
            "signals": {
                "A": {"signal": "ENTER", "score": 60.0},
                "B": {"signal": "ENTER", "score": 88.0},
            }
        }).encode(),
    )
    panel = _Panel().put(_ENTRY, {"A": 100, "B": 100, "SPY": 100})
    for d in [f"2026-07-{d:02d}" for d in range(1, 25)]:
        panel.put(d, {"A": 110, "B": 102, "SPY": 105})
    return panel


def _board(s3, panel) -> tuple[dict, dict]:
    from scoring.leaderboard_producers import build_producer_leaderboard

    res = build_producer_leaderboard(
        s3, _BUCKET, _AS_OF, closes_panel_loader=panel.loader()
    )
    return res, res.get("leaderboard") or {}


class TestOneArmOneRow:
    def test_the_arm_is_in_both_selectors(self):
        """The precondition. If this stops being true the rest of this file is
        vacuous, so it is asserted rather than assumed."""
        from producers.registry import retired_producers

        assert _ARM in {p.name for p in retired_producers(as_of=_AS_OF)}

    def test_a_champion_that_is_also_retired_is_scored_once(self, s3):
        res, board = _board(s3, _seed(s3))
        assert res["status"] == "ok", res
        for block in board["horizons"]:
            names = [r["name"] for r in block["specs"]]
            assert names.count(_ARM) == 1, (
                f"{_ARM} appears {names.count(_ARM)}x on the "
                f"{block['horizon_days']}d block"
            )

    def test_the_board_is_measurable_when_the_champion_is_retired(self, s3):
        """The live symptom: `duplicate_arm_rows` takes the WHOLE board to
        `unmeasurable`, so one arm's double row silences every other arm."""
        res, board = _board(s3, _seed(s3))
        assert res["status"] == "ok"
        assert board.get("status") != "unmeasurable"

    def test_the_serving_arm_is_tagged_champion_not_retired(self, s3):
        """A promotion consumer filtering on `kind` must not find the arm that
        is actually serving tagged "retired"."""
        _res, board = _board(s3, _seed(s3))
        row = next(r for r in board["specs"] if r["name"] == _ARM)
        assert row["kind"] == "champion"
        assert board["champion"] == _ARM

    def test_the_other_retired_arms_are_still_scored(self, s3):
        """The skip is the CHAMPION's alone. §3's trailing window is why
        retired rows exist at all, and dropping the rest would trade one defect
        for a larger one."""
        _res, board = _board(s3, _seed(s3))
        kinds = {r["name"]: r["kind"] for r in board["specs"]}
        retired = [n for n, k in kinds.items() if k == "retired"]
        assert retired, "no retired arm was scored at all"
        assert _ARM not in retired
