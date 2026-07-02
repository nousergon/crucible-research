"""Burn-down guard — forbid reads of the wide horizon-suffixed
score_performance columns (EPIC config#1483 Phase 3, consumer cutover
config#1530).

This repo's seed of the shared ratchet primitive
(``nousergon_lib.quant.horizon_guard``, lifted to a lib chokepoint on its
second adoption per config#1527 — this is the THIRD adoption after
crucible-backtester and crucible-predictor, so we adopt the shared
primitive directly rather than hand-rolling a repo-local copy of the
scan/ratchet mechanics). See that module's docstring for the full mechanics
(migrating allowlist burns down to {}; exempt files are permanent but must
stay honest).

**This repo seeds its allowlist at {} directly** — the three consumer files
in scope (``evals/last_week_scorecard.py``, ``evals/team_accuracy.py``,
``memory/episodic.py``) were migrated onto ``evals.outcome_store`` /
the long-format store in the SAME PR that adds this guard (config#1530),
so there is no interim "known wide-column reader" set to seed. The
"verifiably SOTA" finish line for this repo is therefore immediate, not a
burn-down over time.

Two files are permanently EXEMPT (not `migrating` — they can never become
"clean" because the wide-column literals they contain are not migratable
reads of score_performance):

  * ``archive/schema.py`` — the authoritative DDL for score_performance
    itself. Its ``ALTER TABLE ... ADD COLUMN beat_spy_21d`` etc. statements
    physically DEFINE the wide columns being retired-from-use elsewhere;
    they are the schema's historical record, not a production read, and
    must never be deleted (dropping a column from a live SQLite ALTER
    history is not how this schema module works — see its migration-log
    convention). New columns are never added here for a horizon that
    already went through this cutover, so this file's hit set is fixed and
    known, not growing.
  * ``evals/last_week_scorecard.py`` — the ``SectorRow.mean_log_alpha_21d``
    dataclass field is a pre-existing, externally-serialized (S3 JSON
    artifact, consumed by the Phase-2 prompt-injection pipeline) field name
    that happens to contain the substring ``log_alpha_21d``. It is NOT a
    read of the wide score_performance column of the same name (the field
    is populated from the long-format store's ``log_alpha`` field, see
    ``_build_sector_rows``) — the substring collision is coincidental
    (both names independently describe "log-domain alpha at the canonical
    21d horizon"). Renaming a live external-contract field name to satisfy
    a source-scanning guard would be a bandaid, not a fix; the guard's
    ``check_burndown`` honesty check (stale-exempt-entry failure) still
    protects against this exemption silently rotting if the field is ever
    removed.
"""

from __future__ import annotations

from pathlib import Path

from nousergon_lib.quant.horizon_guard import assert_burndown

_REPO_ROOT = Path(__file__).resolve().parent.parent

_MIGRATING: frozenset[str] = frozenset()

_EXEMPT = frozenset({
    "archive/schema.py",
    "evals/last_week_scorecard.py",
})


def test_wide_horizon_column_burndown():
    assert_burndown(_REPO_ROOT, migrating=_MIGRATING, exempt=_EXEMPT)
