"""The cheap detection pass must run before the expensive write tier.

alpha-engine-config-I6650. The events sweep ran LAST in ``_build_and_sweep``,
after new-name intake and staleness refresh. Every Think Tank run from
2026-08-03 aborted partway through the ``med``-group thesis writes on a
provider 402, so the sweep was never reached:
``thinktank/events/2026-07-31.jsonl`` was still the newest key eight days
later, with ``sweep_tickers: 0`` and ``deadline_skipped_sweep: false`` on every
manifest in between — never reached, not skipped, and no freshness row on the
prefix to notice.

Three properties, pinned structurally because the alternative is a full run
harness with a live LLM client:

1. **Detection precedes the write tier.** ``products/thinktank.md`` §2.4 makes
   the sweep the gate on the expensive tier; running the expensive tier first
   inverts that, and an abort in it costs the day's detection.
2. **The macro-theme update precedes the thesis builds.** The sweep feeds
   ``themes.ensure_current(daily_developments=...)``; theses written after it
   read current themes rather than yesterday's.
3. **Deadline truncation lands on refresh, not on the sweep.** The sweep is one
   un-resumable fan-out; per-ticker refresh is the most droppable work in the
   run. Whichever runs last absorbs the truncation.
"""

from __future__ import annotations

import inspect
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thinktank import run as tt_run  # noqa: E402

_SRC = inspect.getsource(tt_run._build_and_sweep)


def _pos(pattern: str) -> int:
    """Character offset of the first match, or fail with the reason."""
    m = re.search(pattern, _SRC)
    assert m, f"anchor not found in _build_and_sweep: {pattern!r}"
    return m.start()


def test_the_events_sweep_runs_before_the_intake_thesis_loop():
    """The load-bearing order. If this inverts again, the next provider
    outage silently costs detection for as long as it lasts."""
    sweep_at = _pos(r"assessments, macro_notes = sweep\(")
    intake_at = _pos(r"for idx, ticker in enumerate\(manifest\.names_added\)")
    assert sweep_at < intake_at, (
        "the events sweep runs AFTER new-name intake — an abort in the thesis "
        "loop costs the whole day's detection, which is what produced eight "
        "consecutive empty days in thinktank/events/ (config-I6650)"
    )


def test_the_events_sweep_runs_before_the_staleness_refresh_loop():
    sweep_at = _pos(r"assessments, macro_notes = sweep\(")
    refresh_at = _pos(r"for idx, ticker in enumerate\(refresh\)")
    assert sweep_at < refresh_at, (
        "the events sweep runs AFTER the staleness refresh loop; refresh is "
        "the most droppable work in the run and must absorb truncation, not "
        "cause the sweep to be skipped"
    )


def test_the_daily_theme_update_precedes_the_thesis_builds():
    """`build_thesis` reads `themes`. The sweep surfaces the day's macro
    developments, so folding them in first means theses are written against
    today's themes rather than yesterday's — a correctness gain the reorder
    produces for free, and one a future re-order would silently undo."""
    theme_update_at = _pos(r"themes\.ensure_current\(daily_developments=macro_notes\)")
    intake_at = _pos(r"for idx, ticker in enumerate\(manifest\.names_added\)")
    assert theme_update_at < intake_at


def test_event_driven_thesis_writes_precede_routine_intake():
    """§2.3 makes event-driven updates the PRIMARY mechanism and the clock the
    floor. Under a deadline or a mid-run abort the primary mechanism must
    complete first — the update_thesis branch is inside the sweep block."""
    event_write_at = _pos(r'update_reason="event"')
    routine_write_at = _pos(r'update_reason="initial"')
    assert event_write_at < routine_write_at


def test_the_sweep_still_has_its_own_deadline_guard():
    """Reordering must not drop the guard: the sweep is a single fan-out that
    cannot be partially completed, so near the deadline it is skipped ENTIRELY
    and that skip is RECORDED (alpha-engine-config-I5208)."""
    assert "manifest.deadline_skipped_sweep = True" in _SRC
    guard_at = _pos(r"if covered_before and _out_of_time\(seconds_remaining\)")
    sweep_at = _pos(r"assessments, macro_notes = sweep\(")
    assert guard_at < sweep_at


def test_ranked_rows_is_available_to_both_paths():
    """The sweep's event-driven `build_thesis` and the refresh loop both read
    `ranked_rows`. Moving the sweep up without moving its definition would
    raise NameError on the first material event — a failure that only appears
    on a day when something actually happened."""
    ranked_at = _pos(r"ranked_rows = \{")
    first_use_at = _pos(r"board_row=ranked_rows\.get\(")
    assert ranked_at < first_use_at


def test_the_documented_stage_order_matches_the_code():
    """The module docstring is the first thing a reader consults, and it
    described the pre-I6650 order. A doc that contradicts the code is worse
    than no doc — it is a confident wrong answer."""
    doc = tt_run.__doc__ or ""
    sweep_line = doc.index("events sweep over all covered names")
    builds_line = doc.index("thesis builds for the intake set")
    assert sweep_line < builds_line, (
        "run.py's docstring still lists the thesis builds before the events "
        "sweep — it contradicts the code as of config-I6650"
    )
