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
  * the consumer starts reading the board's ranking again (source-level guard)
  * the consumer starts following the champion POINTER again

AMENDED 2026-09-22 (Brian's ruling, alpha-engine-config-I11393). This file used
to assert the opposite of that last leg: that the window FOLLOWED the champion
pointer, per the 2026-08-20 ruling (I7823). It does not any more, and the
reversal is the point.

Think Tank is an ARM of the ``research`` slot, and §3.1 makes an arm an
immutable RECIPE — "whatever cut won this week" is not one. Measured
2026-09-18: the ``universe_cut`` pointer moved from ``attractiveness_top_60``
to ``tech_score_top_60``, two cuts whose intersection was **0 of 60 names**.
This window's entire population was replaced, coverage fell 60/60 -> 9/60 in a
single cycle, the arm produced no leaderboard evidence for days, and its spec
hash never changed, so its arena series recorded the swap as one continuous
line.

The window is now PINNED to ``producers.registry.PINNED_RESEARCH_PREFILTER``
and resolved by ``scoring.universe_membership.resolve_pinned_cut``. A different
pre-filter is tested by REGISTERING AN ARM, never by a pointer moving
underneath the arms already running.
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
from producers.registry import PINNED_RESEARCH_PREFILTER  # noqa: E402
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


def test_the_declaration_no_longer_steers_the_window():
    """The producer moving ``funnel.advances_to`` must NOT move this arm.

    Inverted from its pre-I11393 form, which asserted the window followed the
    declaration. An arm's pre-filter is part of its immutable recipe; a
    producer-side edit that silently re-pointed it is precisely the class of
    change §3.1 exists to refuse. The declaration still governs non-arm funnel
    consumers through ``resolve_funnel_cut`` — it just no longer reaches here.
    """
    board = _board()
    m = _membership(board)
    # The pinned cut stays emitted; only the DECLARATION is re-pointed, which
    # is exactly the producer-side edit that used to move this arm.
    m["cuts"]["attractiveness_top_25"] = {
        "basis": "attractiveness_rank",
        "size": 25,
        "tickers": sorted(t["ticker"] for t in board["stocks"][:25]),
        "source": "test",
    }
    m["funnel"]["advances_to"][FUNNEL_CONSUMER_THINKTANK] = "attractiveness_top_25"
    window = load_feed_window(_store(m), minimum_rank_coverage=200)
    assert window.cut == PINNED_RESEARCH_PREFILTER
    assert window.size == 60


def test_a_missing_declaration_no_longer_breaks_a_pinned_arm():
    """The arm names its own cut, so the funnel declaration is not its input."""
    m = _membership(_board())
    del m["funnel"]["advances_to"][FUNNEL_CONSUMER_THINKTANK]
    window = load_feed_window(_store(m), minimum_rank_coverage=200)
    assert window.cut == PINNED_RESEARCH_PREFILTER


def test_the_pinned_cut_vanishing_from_the_artifact_fails_loud():
    """The replacement failure mode. A pinned cut the producer stopped
    emitting is a producer-side change an arm must never paper over by
    substituting another cut or an empty window."""
    m = _membership(_board())
    m["cuts"].pop(PINNED_RESEARCH_PREFILTER)
    with pytest.raises(UniverseMembershipError) as exc:
        load_feed_window(_store(m), minimum_rank_coverage=200)
    assert PINNED_RESEARCH_PREFILTER in str(exc.value)
    assert "I11393" in str(exc.value)


def test_a_declaration_naming_a_missing_cut_fails_loud():
    m = _membership(_board())
    m["cuts"].pop("attractiveness_top_60")
    with pytest.raises(UniverseMembershipError):
        load_feed_window(_store(m))


def test_a_missing_membership_artifact_fails_loud_with_no_board_fallback():
    with pytest.raises(UniverseMembershipError, match="cannot fall back"):
        load_feed_window(_store(None))


# ── The window follows the CHAMPION pointer ──────────────────────────────────


def test_the_window_does_not_follow_the_champion_pointer():
    """THE regression this file now exists for (alpha-engine-config-I11393).

    Replays the live 2026-09-18 event: the pointer moves to a cut sharing ZERO
    names with the pinned one. Pre-I11393 the window followed it and the arm's
    entire population was replaced mid-series. It must now be inert.
    """
    board = _board()
    m = _with_tech_basis_champion(_membership(board), board)
    pinned = load_feed_window(_store(m), minimum_rank_coverage=200)
    with patch(
        "scoring.universe_membership.PROMOTABLE_CUTS",
        ("attractiveness_top_60", "tech_score_top_60"),
    ):
        moved = load_feed_window(
            _store(m, champion="tech_score_top_60"), minimum_rank_coverage=200
        )
    assert moved.cut == PINNED_RESEARCH_PREFILTER == pinned.cut
    assert moved.basis == pinned.basis == "attractiveness_rank"
    assert moved.tickers == pinned.tickers, (
        "a universe_cut promotion must not change one single name this arm "
        "researches — on 2026-09-18 it changed all 60"
    )
    assert moved.provenance["pinned"] is True


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


def test_a_pinned_cut_whose_basis_has_no_table_still_refuses():
    """The refusal survives the pinning: a cut whose ranking basis has no table
    is refused, never quietly resolved out of a different basis' table."""
    board = _board()
    m = _membership(board)
    m.pop("ranks", None)
    m.pop("rank_tables", None)
    with pytest.raises(UniverseMembershipError):
        load_feed_window(_store(m), minimum_rank_coverage=200)


def test_an_arbitrary_pointer_value_cannot_reach_this_window_at_all():
    """Stronger than the guard it replaces.

    The old test asserted that an unregistered pointer value was REFUSED —
    which still left the pointer as an input to this window, one validation
    away from steering it. It is no longer an input at all, so a garbage value
    is not refused, it is irrelevant. That is the difference between a
    validated arbitrary-cut-selection primitive and no primitive.
    """
    m = _membership(_board())
    expected = load_feed_window(_store(m), minimum_rank_coverage=200)
    for nonsense in ("some_cut_nobody_registered_60", "whatever_i_wrote", ""):
        window = load_feed_window(_store(m, champion=nonsense), minimum_rank_coverage=200)
        assert window.cut == expected.cut
        assert window.tickers == expected.tickers


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
