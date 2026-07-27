"""Per-sector tool-usage analysis over the ``analyst_resources`` table.

Addresses config#925 ("Agent tool refinement — analyze ``analyst_resources``
table for per-sector tool usage patterns"). Sector teams log every quant+qual
ReAct tool call to ``analyst_resources`` at *team* grain, with the originating
team encoded as ``agent="team:{team_id}"``. Each team maps deterministically to
a GICS sector cohort (``team_config.TEAM_SECTORS``), so the same table answers
"which tools does each sector team lean on, and how does that differ across
sectors?" — the input to agent-tool refinement (pruning unused tools, adding
sector-specific ones).

This module is pure compute over rows already loaded from the DB
(``ArchiveManager.load_analyst_resources``); it does no I/O itself so it is
trivially testable and reusable from a Lambda, a notebook, or the dashboard.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict

log = logging.getLogger(__name__)

# Sentinel ``ticker`` for tool-call rows recorded at team/sector grain (the
# combined quant+qual ReAct log is not scoped to a single ticker). Kept here so
# both the writer (graph/research_graph.py archive_writer) and this analysis
# import one definition.
TEAM_RESOURCE_TICKER = "__team__"

# Local team_id → display sector label. Imported lazily inside helpers so this
# module stays importable without the gitignored config that team_config pulls
# in transitively; falls back to the raw team_id when unavailable.
_TEAM_PREFIX = "team:"


def team_id_from_agent(agent: str) -> str:
    """Extract the team_id from an ``agent`` value like ``"team:technology"``."""
    if agent and agent.startswith(_TEAM_PREFIX):
        return agent[len(_TEAM_PREFIX):]
    return agent or "unknown"


def sector_label_for_team(team_id: str) -> str:
    """Map a team_id to its primary GICS sector label, falling back to team_id."""
    try:
        from agents.sector_teams.team_config import TEAM_SECTORS

        sectors = TEAM_SECTORS.get(team_id)
        if sectors:
            return sectors[0]
    except Exception as e:
        # Sector lookup failed — return team_id as fallback (recorded: KeyError, None-deref).
        log.debug("Sector lookup failed for team_id=%s: %s", team_id, e)
    return team_id


def aggregate_tool_usage_by_sector(rows: list[dict]) -> dict:
    """Aggregate ``analyst_resources`` rows into per-sector tool-usage patterns.

    Args:
        rows: as returned by ``ArchiveManager.load_analyst_resources`` — dicts
            with at least ``agent`` and ``resource_type`` keys.

    Returns a dict::

        {
          "by_sector": {
             "Technology": {
                "team_id": "technology",
                "total_calls": 42,
                "distinct_tools": 5,
                "tool_counts": {"price_history": 20, "news_search": 12, ...},
                "tool_share": {"price_history": 0.4762, ...},  # fraction of calls
                "top_tool": "price_history",
             }, ...
          },
          "by_tool": {"price_history": 30, ...},   # global counts
          "totals": {"n_rows": 42, "n_sectors": 2, "n_tools": 6},
          "unused_tools_by_sector": {"Technology": ["macro_lookup"], ...},
        }

    ``unused_tools_by_sector`` lists tools used by *some* sector but not the
    given one — the direct signal for per-sector tool refinement.
    """
    counts_by_team: dict[str, Counter] = defaultdict(Counter)
    global_tools: Counter = Counter()

    for r in rows:
        tool = r.get("resource_type")
        if not tool:
            continue
        team_id = team_id_from_agent(r.get("agent", ""))
        counts_by_team[team_id][tool] += 1
        global_tools[tool] += 1

    all_tools = set(global_tools)
    by_sector: dict[str, dict] = {}
    unused_by_sector: dict[str, list[str]] = {}

    for team_id, tool_counts in counts_by_team.items():
        sector = sector_label_for_team(team_id)
        total = sum(tool_counts.values())
        share = {
            t: round(c / total, 4) for t, c in tool_counts.items()
        } if total else {}
        top_tool = tool_counts.most_common(1)[0][0] if tool_counts else None
        by_sector[sector] = {
            "team_id": team_id,
            "total_calls": total,
            "distinct_tools": len(tool_counts),
            "tool_counts": dict(tool_counts),
            "tool_share": share,
            "top_tool": top_tool,
        }
        unused = sorted(all_tools - set(tool_counts))
        if unused:
            unused_by_sector[sector] = unused

    return {
        "by_sector": by_sector,
        "by_tool": dict(global_tools),
        "totals": {
            "n_rows": sum(global_tools.values()),
            "n_sectors": len(by_sector),
            "n_tools": len(all_tools),
        },
        "unused_tools_by_sector": unused_by_sector,
    }
