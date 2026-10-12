"""The dimensions a rubric DECLARES, and the guard that holds a judge to them.

Why this exists (measured 2026-10-07, nous-ergon-ops "CloudWatch alarm age"
red on ``alpha-engine-eval-review-floor``): the Haiku-tier judge returned FIVE
``dimension_scores`` for
``2610031309_thinktank_theme.ed83622b9c35-sector-consumer_discretionary-v23``
(2026-10-03T13:19:45Z) against a rubric that declares four. The fifth was
named ``dimension_note``. Nothing compared the returned names with the rubric,
so ``evals.metrics.emit_eval_metric`` published it as a brand-new
``agent_quality_score`` series
``thinktank_theme/dimension_note/claude-haiku-4-5`` with one review in it.
``control_bands`` discovers its combo matrix from ``ListMetrics``, so that
phantom joined the matrix, sits below ``MIN_REVIEWS_PER_WEEK`` forever, and
holds ``agent_quality_score_weekly_review_floor_breach_count`` at >= 1 until
CloudWatch stops listing it — a page about a criterion that does not exist.

Two consumers, one source of truth (the rubric text itself):

* the judge drops an undeclared dimension BEFORE the artifact is built, so it
  is never persisted, never emitted, and never moves an escalation decision;
* ``control_bands`` leaves an undeclared criterion out of the floor's matrix,
  so a phantom that already reached CloudWatch stops counting at once instead
  of in two weeks.

The rubric files declare their dimensions with one fixed header per dimension
(``═══ DIMENSION 3: anchor_fidelity ═══``) — all eight rubrics in the private
config repo and both ``config/prompts.example`` rubrics use it. A rubric whose
headers this cannot read yields an EMPTY declaration, and an empty
declaration is treated as "unknown", never as "nothing is allowed": both
callers then keep everything and log, because dropping every score of every
eval on a header-format change would be a far worse silent failure than the
one this module fixes.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from typing import Any

logger = logging.getLogger(__name__)

_DIMENSION_HEADER_RE = re.compile(
    r"^\s*═+\s*DIMENSION\s+\d+\s*:\s*([A-Za-z0-9_]+)\s*═+\s*$",
    re.MULTILINE,
)


def declared_dimensions(rubric_text: str) -> tuple[str, ...]:
    """The dimension names a rubric's text declares, in declaration order.

    Empty when the text carries no recognisable header — callers must read
    that as UNKNOWN (see the module docstring), not as an empty rubric.
    """
    seen: list[str] = []
    for name in _DIMENSION_HEADER_RE.findall(rubric_text or ""):
        if name not in seen:
            seen.append(name)
    return tuple(seen)


def keep_declared_dimension_scores(
    dimension_scores: Sequence[Any],
    declared: Iterable[str],
    *,
    agent_id: str,
    judge_model: str,
    rubric_id: str,
) -> list[Any]:
    """Return ``dimension_scores`` minus every entry the rubric does not declare.

    Entries carry the dimension name on ``.dimension`` (the lib's
    ``RubricEvalLLMOutput`` shape). Dropped names are logged at WARNING with
    the artifact identity, so a judge that starts inventing criteria is
    visible in the log rather than in a CloudWatch matrix.
    """
    allowed = set(declared)
    scores = list(dimension_scores)
    if not allowed:
        logger.warning(
            "[rubric_dimensions] rubric %s declares no readable dimension "
            "headers — keeping all %d judged dimension(s) for agent_id=%s "
            "judge=%s unfiltered",
            rubric_id, len(scores), agent_id, judge_model,
        )
        return scores
    kept = [s for s in scores if getattr(s, "dimension", None) in allowed]
    if scores and not kept:
        # Not a phantom beside real scores: NOTHING matched, which is a
        # rubric/judge contract mismatch (a renamed rubric, a stale fixture,
        # a header format this parser does not read). Wiping the whole eval
        # would turn that into silent data loss, so keep it and say so.
        logger.error(
            "[rubric_dimensions] none of the %d judged dimension(s) %s is "
            "declared by rubric %s (declares %s) for agent_id=%s judge=%s — "
            "keeping them unfiltered; this is a rubric/judge mismatch, not "
            "a phantom",
            len(scores), [getattr(s, "dimension", None) for s in scores],
            rubric_id, sorted(allowed), agent_id, judge_model,
        )
        return scores
    dropped = [
        getattr(s, "dimension", None) for s in scores
        if getattr(s, "dimension", None) not in allowed
    ]
    if dropped:
        logger.warning(
            "[rubric_dimensions] dropped %d dimension score(s) the rubric "
            "does not declare: %s (agent_id=%s judge=%s rubric=%s declares "
            "%s) — never persisted or emitted, so no phantom "
            "agent_quality_score series is created",
            len(dropped), dropped, agent_id, judge_model, rubric_id,
            sorted(allowed),
        )
    return kept


def declared_criteria_for_agent(agent_id: str) -> tuple[str, ...] | None:
    """The criteria the live rubric for ``agent_id`` declares, or ``None``
    when that cannot be established (unmapped agent, rubric not loadable,
    unreadable headers).

    ``None`` means UNKNOWN: ``control_bands`` keeps such a combo in the floor
    matrix exactly as before. Imported lazily so ``control_bands`` gains no
    import-time dependency on the judge module.
    """
    try:
        from agents.prompt_loader import load_prompt
        from evals.judge import resolve_rubric_for_agent

        rubric = resolve_rubric_for_agent(agent_id)
        if rubric is None:
            return None
        names = declared_dimensions(load_prompt(rubric).text)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[rubric_dimensions] could not read the rubric for agent_id=%s "
            "(%s: %s) — its combos stay in the floor matrix unfiltered",
            agent_id, type(exc).__name__, exc,
        )
        return None
    return names or None
