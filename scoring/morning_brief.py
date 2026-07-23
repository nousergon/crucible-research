"""Templated, non-LLM morning-brief markdown — dashboard Research Briefing
Archive producer (config-I3290, port of the retired multi-agent
``consolidated_report`` path).

The old ``consolidated_report`` was an LLM-authored narrative assembled by
the six-team+CIO LangGraph's ``consolidator_node`` and persisted via
``archive/manager.py::save_consolidated_report``. config#2515 removed that
whole graph from the weekly SF — ``scoring/signals_envelope.py`` is
explicitly NO-AGENT (no LLM calls, no LangGraph). There is no live
narrative-generation step to port INTO; recreating an LLM narrative here
would silently reintroduce the agentic stage config#2515 was built to
remove. Instead this builds a structured, data-only digest from the SAME
envelope fields ``build_signals_envelope`` already computes — mirrors the
established thin-digest pattern in
``scoring.attractiveness_trajectory.format_digest_markdown``.

Written as a post-step right after ``write_envelope`` in
``lambda/signals_envelope_handler.py``, gated to ``target == "production"``
and fail-soft (a brief-write failure must never fail the primary
signals.json deliverable).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_REGIME_LABELS = {"bull": "Bullish", "neutral": "Neutral", "bear": "Bearish"}

_TOP_N = 10


def build_morning_brief_markdown(envelope: dict[str, Any]) -> str:
    """Pure function: templated markdown brief from a built signals envelope.

    No LLM narrative (see module doc) — regime label, sector ratings table,
    and the top/bottom of the universe by attractiveness score.
    """
    run_date = envelope.get("run_date") or envelope.get("date", "")
    regime = envelope.get("market_regime", "neutral")
    regime_label = _REGIME_LABELS.get(regime, regime)
    universe = envelope.get("universe") or []
    sector_ratings = envelope.get("sector_ratings") or {}

    scored = [
        e for e in universe
        if isinstance(e, dict) and e.get("ticker") and e.get("score") is not None
    ]
    scored.sort(key=lambda e: e["score"], reverse=True)
    top = scored[:_TOP_N]
    bottom = scored[-_TOP_N:][::-1] if len(scored) > _TOP_N else []

    lines = [f"# Morning Research Brief — {run_date}", ""]
    lines.append(f"**Market regime:** {regime_label}")
    lines.append(f"**Universe:** {len(universe)} names, {len(sector_ratings)} sectors")
    lines.append("")
    lines.append(
        "_Quant-only digest — no LLM-authored narrative in this producer "
        "(config#2515 removed the multi-agent Research stage from the "
        "weekly SF)._"
    )
    lines.append("")

    lines.append("## Sector ratings")
    if sector_ratings:
        for sector in sorted(sector_ratings):
            rating = sector_ratings[sector] or {}
            modifier = rating.get("modifier", 1.0)
            lines.append(
                f"- **{sector}** — {rating.get('rating', 'n/a')} (x{modifier:.2f})"
            )
    else:
        lines.append("- _no sector data this cycle_")
    lines.append("")

    lines.append("## Top attractiveness scores")
    if top:
        for e in top:
            lines.append(
                f"- **{e['ticker']}** ({e.get('sector', 'Unknown')}) — {e['score']:.1f}"
            )
    else:
        lines.append("- _none this cycle_")
    lines.append("")

    lines.append("## Bottom attractiveness scores")
    if bottom:
        for e in bottom:
            lines.append(
                f"- **{e['ticker']}** ({e.get('sector', 'Unknown')}) — {e['score']:.1f}"
            )
    else:
        lines.append("- _none this cycle_")

    return "\n".join(lines)


def write_morning_brief(
    run_date: str,
    markdown: str,
    *,
    bucket: str,
    s3_client: Any = None,
) -> str:
    """Write the brief to ``consolidated/{run_date}/morning.md``. Returns the key."""
    from scoring.signals_envelope import _client

    s3 = _client(s3_client)
    key = f"consolidated/{run_date}/morning.md"
    s3.put_object(
        Bucket=bucket, Key=key, Body=markdown.encode("utf-8"), ContentType="text/markdown",
    )
    logger.info("[morning_brief] wrote → s3://%s/%s (%d bytes)", bucket, key, len(markdown))
    return key
