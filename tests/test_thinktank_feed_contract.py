"""Consumer half of the scanner→Think Tank boundary contract.

Producer half: ``tests/test_universe_membership.py`` §7. Together they are the
M0 producer/consumer pair for ``universe_membership/latest.json`` as read by
``thinktank/`` (alpha-engine-config-I7842).

The defect being pinned: Think Tank re-derived its own candidate ranking by
sorting the universe board's ``stocks[]`` on ``attractiveness_score``, so it
hardcoded the ranking BASIS and never consulted the champion pointer. It agreed
with the scanner only for as long as the champion happened to rank on
attractiveness; a promotion to a tech-basis cut would have left it covering the
losing arm's names with nothing raising — alpha-engine-config-I7808's shape,
in a second module.

Each test below fails if one leg of that contract is removed:

  * the producer stops declaring ``funnel.advances_to.thinktank_coverage_window``
  * the declaration names a cut that is not emitted
  * the rank table stops covering what the consumer's configured ceiling needs
  * the consumer stops following the champion pointer
  * the consumer starts reading the board's ranking again (source-level guard)
"""

from __future__ import annotations

import io
import json
import os
import pathlib
import re
import sys
import types
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring.universe_membership import (  # noqa: E402
    CUT_CHAMPION_POINTER_KEY,
    FUNNEL_CONSUMER_THINKTANK,
    UniverseMembershipError,
    rank_table_for_cut,
)
from thinktank.feed import (  # noqa: E402
    build_feed_window,
    join_board_rows,
    load_feed_window,
    unjoinable_ranked_names,
)

_REPO = pathlib.Path(__file__).resolve().parent.parent


def _board(n: int = 300) -> dict:
    return {
        "as_of": "2026-08-20",
        "stocks": [
            {"ticker": f"T{i:04d}", "sector": "Tech", "attractiveness_score": float(2000 - i)} for i in range(n)
        ],
    }


def _with_tech_basis_champion(membership: dict, board: dict) -> dict:
    """``membership`` with a promotable ``tech_score_top_60`` arm AND the
    full-universe rank table its basis is served from (I7843).

    The cut is the head of the tech ranking (Think Tank asserts that), but the
    TAIL is reversed against the attractiveness ordering, so a consumer that
    resolved the wrong table is visible in the assertion rather than
    coincidentally right.
    """
    tickers = [s["ticker"] for s in board["stocks"]]
    membership["cuts"]["tech_score_top_60"] = {
        "basis": "tech_score_rank",
        "size": 60,
        "tickers": sorted(tickers[:60]),
        "source": "test",
    }
    tech_order = tickers[:60] + list(reversed(tickers[60:]))
    membership["tech_score_ranks"] = {
        t: {"tech_score_rank": i + 1, "tech_score": float(len(tech_order) - i)} for i, t in enumerate(tech_order)
    }
    membership["rank_tables"] = {
        "attractiveness_rank": {
            "field": "ranks",
            "rank_key": "attractiveness_rank",
            "score_key": "attractiveness_score",
            "size": len(tickers),
        },
        "tech_score_rank": {
            "field": "tech_score_ranks",
            "rank_key": "tech_score_rank",
            "score_key": "tech_score",
            "size": len(tickers),
        },
    }
    return membership


def _membership(board: dict, *, window_cut: str = "attractiveness_top_60", size: int = 60) -> dict:
    tickers = [s["ticker"] for s in board["stocks"]]
    ranks = {t: {"attractiveness_rank": i + 1, "attractiveness_score": float(2000 - i)} for i, t in enumerate(tickers)}
    return {
        "schema_version": 1,
        "run_date": "2026-08-20",
        "cut_effective_date": "2026-08-20",
        "cut_refresh_cadence": "daily",
        "universe_count": len(ranks),
        "feed_cut": window_cut,
        "funnel": {
            "population": len(ranks),
            "advances_to": {
                "predictor_universe": "attractiveness_top_20",
                "rag_corpus_scope": window_cut,
                FUNNEL_CONSUMER_THINKTANK: window_cut,
            },
        },
        "cuts": {
            window_cut: {
                "basis": "attractiveness_rank",
                "size": size,
                "tickers": sorted(tickers[:size]),
                "source": "test",
            },
            "attractiveness_top_20": {
                "basis": "attractiveness_rank",
                "size": 20,
                "tickers": sorted(tickers[:20]),
                "source": "test",
            },
        },
        "ranks": ranks,
    }


class _Store:
    """Minimal stand-in for ``ThinktankStore`` — only ``bucket``/``s3`` are read."""

    def __init__(self, objects: dict[str, dict]):
        self.bucket = "alpha-engine-research"
        outer = self

        class _S3:
            exceptions = types.SimpleNamespace(NoSuchKey=type("NoSuchKey", (Exception,), {}))

            def get_object(self, Bucket, Key):  # noqa: N803
                if Key in outer.objects:
                    return {"Body": io.BytesIO(json.dumps(outer.objects[Key]).encode())}
                exc = Exception("NoSuchKey")
                exc.response = {"Error": {"Code": "NoSuchKey"}}
                raise exc

        self.objects = objects
        self.s3 = _S3()


def _store(membership: dict | None, champion: str | None = None) -> _Store:
    objects: dict[str, dict] = {}
    if membership is not None:
        objects["universe_membership/latest.json"] = membership
    if champion is not None:
        objects[CUT_CHAMPION_POINTER_KEY] = {"champion": champion}
    return _Store(objects)


# ── The window comes from the DECLARATION, not from a constant ───────────────


def test_window_is_the_cut_the_artifact_declares():
    board = _board()
    window = load_feed_window(_store(_membership(board)), minimum_rank_coverage=200)
    assert window.cut == "attractiveness_top_60"
    assert window.basis == "attractiveness_rank"
    assert window.size == 60
    assert set(window.tickers) == {f"T{i:04d}" for i in range(60)}


def test_window_follows_the_declaration_when_the_producer_moves_it():
    """A hardcoded ``attractiveness_top_60`` on the consumer would pass every
    other test in this file and fail only here."""
    board = _board()
    m = _membership(board, window_cut="attractiveness_top_25", size=25)
    window = load_feed_window(_store(m), minimum_rank_coverage=200)
    assert window.cut == "attractiveness_top_25"
    assert window.size == 25


def test_missing_declaration_fails_loud_rather_than_defaulting():
    m = _membership(_board())
    del m["funnel"]["advances_to"][FUNNEL_CONSUMER_THINKTANK]
    with pytest.raises(UniverseMembershipError, match=FUNNEL_CONSUMER_THINKTANK):
        load_feed_window(_store(m))


def test_a_declaration_naming_a_missing_cut_fails_loud():
    m = _membership(_board())
    m["cuts"].pop("attractiveness_top_60")
    with pytest.raises(UniverseMembershipError):
        load_feed_window(_store(m))


def test_a_missing_membership_artifact_fails_loud_with_no_board_fallback():
    with pytest.raises(UniverseMembershipError, match="cannot fall back"):
        load_feed_window(_store(None))


# ── The window follows the CHAMPION pointer ──────────────────────────────────


def test_the_window_follows_the_champion_pointer_not_the_static_declaration():
    """Brian's ruling 2026-08-20 (alpha-engine-config-I7823): the arms of the
    count-matched slot are promoted weekly and the consumers of that slot read
    whichever is champion. The declaration states the ARRANGEMENT; the pointer
    states which arm serves it.

    `PROMOTABLE_CUTS` is patched to include the tech-basis arm: this test is
    about what the WINDOW does once an arm holds the pointer, not about which
    arms may hold it. `tech_score_top_60` became observe-only on 2026-08-21
    (alpha-engine-config-I8060), and the separate refusal test below is what
    pins that.
    """
    board = _board()
    m = _with_tech_basis_champion(_membership(board), board)
    with patch(
        "scoring.universe_membership.PROMOTABLE_CUTS",
        ("attractiveness_top_60", "tech_score_top_60"),
    ):
        window = load_feed_window(_store(m, champion="tech_score_top_60"), minimum_rank_coverage=200)
    assert window.cut == "tech_score_top_60"
    assert window.provenance["declared_cut"] == "attractiveness_top_60"
    assert window.provenance["basis"] == "tech_score_rank"


def test_a_tech_basis_champion_is_ranked_in_its_own_basis():
    """The basis the pointer names is the basis the consumer ranks by.

    Since alpha-engine-config-I7843 the producer emits a full-universe
    ``tech_score_rank`` table, so a tech-basis champion is SERVED — in its own
    basis. What must never happen is being served out of ``ranks``: the pointer
    would name one arm while the consumer ranked by another (I7808's shape).
    """
    board = _board()
    m = _with_tech_basis_champion(_membership(board), board)
    ranks, basis = rank_table_for_cut(m, "tech_score_top_60", minimum_coverage=200)
    assert basis == "tech_score_rank"
    # Below the cut the two tables disagree by construction, so serving this
    # out of ``ranks`` would be visible here — and is not happening.
    tickers = [s["ticker"] for s in board["stocks"]]
    assert ranks[tickers[-1]] == 61
    assert ranks[tickers[60]] == len(tickers)


def test_a_tech_basis_champion_with_no_full_table_still_refuses():
    """The refusal survives the fix: a cut whose basis has no table is refused,
    never quietly resolved out of the attractiveness table."""
    board = _board()
    m = _with_tech_basis_champion(_membership(board), board)
    m.pop("tech_score_ranks")
    m.pop("rank_tables")
    with patch(
        "scoring.universe_membership.PROMOTABLE_CUTS",
        ("attractiveness_top_60", "tech_score_top_60"),
    ), pytest.raises(UniverseMembershipError, match="I7843"):
        load_feed_window(_store(m, champion="tech_score_top_60"), minimum_rank_coverage=200)


def test_an_unregistered_arm_cannot_hold_the_window_even_if_the_pointer_names_it():
    """The consumer-side half of alpha-engine-config-I8060, rescoped by I9272.

    A pointer naming an arm outside `PROMOTABLE_CUTS` is refused by
    `live_cut_champion` before the window is ever built — the refusal does not
    depend on the promotion engine, the board, or this module being correct.

    `tech_score_top_60` is no longer such an arm (Brian's ruling 2026-08-29
    makes every scored arm promotable), so the property is asserted against a
    name that is genuinely outside the register. That is the only case this
    guard was ever load-bearing for: a value nobody validated.
    """
    board = _board()
    m = _with_tech_basis_champion(_membership(board), board)
    with pytest.raises(UniverseMembershipError) as exc:
        load_feed_window(
            _store(m, champion="some_cut_nobody_registered_60"), minimum_rank_coverage=200
        )
    assert "some_cut_nobody_registered_60" in str(exc.value)
    assert "attractiveness_top_60" in str(exc.value)


def test_an_unvalidated_champion_pointer_is_refused():
    m = _membership(_board())
    with pytest.raises(UniverseMembershipError, match="Refusing to resolve"):
        load_feed_window(_store(m, champion="whatever_i_wrote"))


# ── Rank-table coverage ──────────────────────────────────────────────────────


def test_a_rank_table_short_of_the_configured_ceiling_is_refused():
    board = _board()
    m = _membership(board)
    m["ranks"] = dict(list(m["ranks"].items())[:100])
    with pytest.raises(UniverseMembershipError, match="I7843"):
        load_feed_window(_store(m), minimum_rank_coverage=200)


def test_a_universe_smaller_than_the_ceiling_is_not_an_error():
    """The ceiling not binding is a legitimate state; only a table short of the
    universe it claims to rank is a defect."""
    board = _board(30)
    window = load_feed_window(_store(_membership(board, size=30)), minimum_rank_coverage=200)
    assert window.size == 30


# ── The join: membership from the artifact, row content from the board ───────


def test_a_window_ticker_missing_from_the_board_fails_loud():
    board = _board()
    m = _membership(board)
    board["stocks"] = [s for s in board["stocks"] if s["ticker"] != "T0005"]
    window = load_feed_window(_store(m), minimum_rank_coverage=200)
    with pytest.raises(UniverseMembershipError, match="T0005"):
        join_board_rows(window, board)


def test_a_ranked_name_outside_the_window_with_no_board_row_is_recorded():
    """The live instance: ``EQR`` is ranked 98 and absent from the board's 903
    rows (alpha-engine-config-I7844). It cannot be intake — there is no row to
    underwrite from — but it must not vanish silently either."""
    board = _board()
    m = _membership(board)
    board["stocks"] = [s for s in board["stocks"] if s["ticker"] != "T0098"]
    window = load_feed_window(_store(m), minimum_rank_coverage=200)
    assert join_board_rows(window, board)  # inside the window: unaffected
    assert unjoinable_ranked_names(window, board, 150) == ["T0098"]
    assert unjoinable_ranked_names(window, board, 60) == []


# ── The cut and the ranking must be ONE ordering ─────────────────────────────


def test_a_cut_that_is_not_the_head_of_its_own_ranking_is_refused():
    ranks = {f"T{i:04d}": i + 1 for i in range(100)}
    with pytest.raises(UniverseMembershipError, match="not the head"):
        build_feed_window(
            [f"T{i:04d}" for i in range(50, 60)],  # ranks 51-60, not the head
            ranks,
            {"cut": "bogus_top_10", "basis": "attractiveness_rank", "run_date": "x"},
        )


def test_a_cut_whose_declared_size_disagrees_with_its_tickers_is_refused():
    ranks = {f"T{i:04d}": i + 1 for i in range(100)}
    with pytest.raises(UniverseMembershipError, match="declares size"):
        build_feed_window(
            [f"T{i:04d}" for i in range(10)],
            ranks,
            {"cut": "c", "basis": "attractiveness_rank", "declared_size": 60, "run_date": "x"},
        )


# ── Source-level guard: the re-derivation must not come back ─────────────────


def test_thinktank_never_sorts_the_board_by_attractiveness_again():
    """The regression this whole contract exists to prevent.

    A second, quietly-diverging ranking is invisible to every behavioural test
    while it happens to agree with the first — measured 2026-08-20, the two
    agreed on 60 of 60 window names and still disagreed at ``exit_rank``. So the
    guard is on the SOURCE: nothing under ``thinktank/`` may sort the universe
    board's rows, and nothing may hardcode the window width.
    """
    sort_on_score = re.compile(r"sorted\((?:(?!\)).)*attractiveness_score", re.S)
    offenders = []
    for path in sorted((_REPO / "thinktank").glob("*.py")):
        src = path.read_text()
        if sort_on_score.search(src):
            offenders.append(f"{path.name}: sorts on attractiveness_score")
        if "GAP_FILL_TOP_N" in src:
            offenders.append(f"{path.name}: references the deleted GAP_FILL_TOP_N constant")
        if re.search(r"\branked_universe\b", src) and path.name != "feed.py":
            offenders.append(f"{path.name}: references the deleted ranked_universe()")
    assert not offenders, (
        "thinktank/ is deriving its own ranking or width again instead of "
        "reading the declared contract (alpha-engine-config-I7842): " + "; ".join(offenders)
    )


def test_only_feed_py_reads_the_membership_artifact():
    """One resolver. A second call site would be a second opinion about which
    arm is live the moment the pointer moves."""
    offenders = [
        path.name
        for path in sorted((_REPO / "thinktank").glob("*.py"))
        if path.name != "feed.py"
        and (
            "universe_membership" in path.read_text()
            and "import" in path.read_text().split("universe_membership")[0][-200:]
        )
    ]
    assert not offenders, (
        "thinktank modules importing scoring.universe_membership directly "
        f"instead of going through thinktank.feed: {offenders}"
    )
