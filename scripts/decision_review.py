"""
Agent-decision review CLI — read-only review of what the research pipeline
decided about a ticker, and why a ticker was (or was not) chosen.

ROADMAP L4567 Phase 1. Artifact-first: every answer here comes from the
already-persisted decision audit trail in ``research.db`` — no LLM call,
no cost. The Saturday pipeline writes a complete trail every cycle:

  - ``scanner_evaluations``  — ALL ~900 screened tickers (pass + fail),
                               with ``filter_fail_reason`` + gate flags.
  - ``team_candidates``      — the FULL quant-ranked set per sector team,
                               incl. ranked-but-not-recommended
                               (``team_recommended=0``) with ``quant_rank``.
  - ``cio_evaluations``      — per-ticker CIO decision + ``rationale`` +
                               ``rule_tags``.
  - ``investment_thesis``    — thesis summary + rating + conviction for
                               population/entrant tickers.

The ``why-not`` command walks that funnel (scanner gate → team rank → CIO
decision) and reports the first stage that dropped the ticker, with the
comparison context. The richer S3 ``decision_artifacts/`` snapshots (the
full agent input + raw LLM output, gated behind
``ALPHA_ENGINE_DECISION_CAPTURE_ENABLED``) and the LLM-replay fallback are
later phases — this CLI deliberately needs neither.

Usage::

    python -m scripts.decision_review ticker NVDA [--date 2026-05-16]
    python -m scripts.decision_review why-not AAPL [--date 2026-05-16]
    python -m scripts.decision_review date [--date 2026-05-16]

    # DB source: a local file, or pull research.db from S3 (default).
    python -m scripts.decision_review ticker NVDA --db ./research.db
    python -m scripts.decision_review ticker NVDA --pull   # force S3 refresh

    # Machine-readable output for any command:
    python -m scripts.decision_review why-not AAPL --json
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
from typing import Any

# Canonical "this CIO decision admits the ticker into the population"
# predicate. The CIO emits both "ADVANCE" (rubric) and the floor-fill
# "ADVANCE_FORCED" — a consumer that matches only "ADVANCE" silently
# drops forced entrants (the bug class that hid the min_new_entrants
# floor for weeks). Mirror the canonical set from
# ``graph.state_schemas.ADVANCE_DECISIONS``.
_ADVANCE_DECISIONS = {"ADVANCE", "ADVANCE_FORCED"}

# Decision tables this CLI reads. Used to compute the latest eval_date and
# to give a clear error on a DB that predates the evaluation schema (added
# in migration 8 / schema v8).
_DECISION_TABLES = ("scanner_evaluations", "team_candidates", "cio_evaluations")

_DB_S3_KEY = "research.db"


# ── DB helpers ────────────────────────────────────────────────────────────


def _rows(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[dict]:
    """Run a query and return a list of plain dicts.

    Independent of ``conn.row_factory`` so it behaves identically whether
    the caller configured ``sqlite3.Row`` or not (tests pass a bare
    connection)."""
    cur = conn.execute(sql, params)
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]


def _has_decision_tables(conn: sqlite3.Connection) -> bool:
    names = {
        r["name"]
        for r in _rows(
            conn,
            "SELECT name FROM sqlite_master WHERE type='table'",
        )
    }
    return all(t in names for t in _DECISION_TABLES)


def latest_eval_date(conn: sqlite3.Connection) -> str | None:
    """Most recent ``eval_date`` across the decision tables, or None if the
    DB has no decision rows yet."""
    dates: list[str] = []
    for table in _DECISION_TABLES:
        # `table` iterates the hardcoded `_DECISION_TABLES` tuple, never external input.
        rows = _rows(conn, f"SELECT MAX(eval_date) AS d FROM {table}")  # noqa: S608
        if rows and rows[0]["d"]:
            dates.append(rows[0]["d"])
    return max(dates) if dates else None


def _resolve_date(conn: sqlite3.Connection, eval_date: str | None) -> str | None:
    return eval_date or latest_eval_date(conn)


def _parse_rule_tags(raw: Any) -> list[str]:
    """``cio_evaluations.rule_tags`` is a JSON list[str] (or NULL)."""
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    try:
        val = json.loads(raw)
        return val if isinstance(val, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


# ── Query layer (pure; takes a connection, returns plain dicts) ────────────


def review_ticker(
    conn: sqlite3.Connection, ticker: str, eval_date: str | None = None
) -> dict:
    """Everything the pipeline recorded about ``ticker`` on ``eval_date``.

    Returns a dict with keys ``ticker``, ``eval_date``, ``scanner``,
    ``team_candidates`` (list — usually one team, occasionally more),
    ``cio``, and ``thesis`` (each None when no row exists)."""
    ticker = ticker.upper()
    date = _resolve_date(conn, eval_date)

    scanner = _rows(
        conn,
        "SELECT * FROM scanner_evaluations WHERE ticker=? AND eval_date=?",
        (ticker, date),
    )
    teams = _rows(
        conn,
        "SELECT * FROM team_candidates WHERE ticker=? AND eval_date=? "
        "ORDER BY quant_rank",
        (ticker, date),
    )
    cio = _rows(
        conn,
        "SELECT * FROM cio_evaluations WHERE ticker=? AND eval_date=?",
        (ticker, date),
    )
    thesis = _rows(
        conn,
        "SELECT * FROM investment_thesis WHERE symbol=? AND date=? "
        "ORDER BY run_time DESC LIMIT 1",
        (ticker, date),
    )
    cio_row = cio[0] if cio else None
    if cio_row is not None:
        cio_row = dict(cio_row)
        cio_row["rule_tags"] = _parse_rule_tags(cio_row.get("rule_tags"))

    return {
        "ticker": ticker,
        "eval_date": date,
        "scanner": scanner[0] if scanner else None,
        "team_candidates": teams,
        "cio": cio_row,
        "thesis": thesis[0] if thesis else None,
    }


def _team_recommended_context(
    conn: sqlite3.Connection, team_id: str, eval_date: str
) -> dict:
    """Recommended-pick context within one team+cycle, for comparison."""
    recs = _rows(
        conn,
        "SELECT ticker, quant_rank, quant_score FROM team_candidates "
        "WHERE team_id=? AND eval_date=? AND team_recommended=1 "
        "ORDER BY quant_rank",
        (team_id, eval_date),
    )
    total = _rows(
        conn,
        "SELECT COUNT(*) AS n FROM team_candidates WHERE team_id=? AND eval_date=?",
        (team_id, eval_date),
    )
    scores = [r["quant_score"] for r in recs if r["quant_score"] is not None]
    return {
        "recommended_count": len(recs),
        "ranked_total": total[0]["n"] if total else 0,
        "recommended_score_min": min(scores) if scores else None,
        "recommended_score_max": max(scores) if scores else None,
        "recommended_tickers": [r["ticker"] for r in recs],
    }


def explain_why_not(
    conn: sqlite3.Connection, ticker: str, eval_date: str | None = None
) -> dict:
    """Walk the decision funnel and report where ``ticker`` was dropped.

    Returns a dict with ``ticker``, ``eval_date``, ``stage`` (one of
    ``not_screened`` / ``scanner`` / ``team`` / ``cio`` / ``chosen`` /
    ``no_record``), ``verdict`` (human one-liner), and ``detail`` (the
    structured rows behind the verdict)."""
    ticker = ticker.upper()
    date = _resolve_date(conn, eval_date)
    review = review_ticker(conn, ticker, date)
    scanner = review["scanner"]
    teams = review["team_candidates"]
    cio = review["cio"]

    # Stage 0 — was it even in the screened universe?
    if scanner is None and not teams and cio is None:
        return {
            "ticker": ticker,
            "eval_date": date,
            "stage": "no_record",
            "verdict": (
                f"No decision record for {ticker} on {date}. It was not in the "
                f"screened universe that cycle (not an S&P 500+400 constituent, "
                f"or no price/feature data), or no cycle ran on this date."
            ),
            "detail": {},
        }

    # Stage 1 — scanner quant filter.
    if scanner is not None and not scanner.get("quant_filter_pass"):
        reason = scanner.get("filter_fail_reason") or "below_thresholds"
        gates = {
            "liquidity_pass": scanner.get("liquidity_pass"),
            "volatility_pass": scanner.get("volatility_pass"),
            "balance_sheet_pass": scanner.get("balance_sheet_pass"),
        }
        failed = [g for g, v in gates.items() if v == 0]
        return {
            "ticker": ticker,
            "eval_date": date,
            "stage": "scanner",
            "verdict": (
                f"{ticker} was dropped at the SCANNER stage on {date}: "
                f"filter_fail_reason={reason}"
                + (f"; failed gates: {', '.join(failed)}" if failed else "")
                + f" (tech_score={scanner.get('tech_score')})."
            ),
            "detail": {"scanner": scanner},
        }

    # Stage 2 — sector-team quant ranking.
    if teams:
        recommended = [t for t in teams if t.get("team_recommended")]
        if not recommended:
            t0 = teams[0]
            ctx = _team_recommended_context(conn, t0["team_id"], date)
            return {
                "ticker": ticker,
                "eval_date": date,
                "stage": "team",
                "verdict": (
                    f"{ticker} was screened by the '{t0['team_id']}' team on {date} "
                    f"but NOT recommended (team_recommended=0): "
                    f"quant_rank={t0.get('quant_rank')} of {ctx['ranked_total']} "
                    f"ranked, quant_score={t0.get('quant_score')}, "
                    f"qual_score={t0.get('qual_score')}. The team recommended "
                    f"{ctx['recommended_count']} pick(s) "
                    f"({', '.join(ctx['recommended_tickers']) or 'none'}) with "
                    f"quant_score {ctx['recommended_score_min']}–"
                    f"{ctx['recommended_score_max']}."
                ),
                "detail": {"team_candidates": teams, "team_context": ctx},
            }
    elif scanner is not None and scanner.get("quant_filter_pass"):
        # Passed the scanner filter but never surfaced in a team's ranked
        # picks (the team's quant analyst did not rank it).
        return {
            "ticker": ticker,
            "eval_date": date,
            "stage": "team",
            "verdict": (
                f"{ticker} passed the scanner quant filter on {date} but did not "
                f"appear in any sector team's ranked picks — the team's quant "
                f"analyst screened it out without surfacing it."
            ),
            "detail": {"scanner": scanner},
        }

    # Stage 3 — CIO.
    if cio is not None:
        decision = (cio.get("cio_decision") or "").upper()
        if decision in _ADVANCE_DECISIONS:
            return {
                "ticker": ticker,
                "eval_date": date,
                "stage": "chosen",
                "verdict": (
                    f"{ticker} WAS chosen on {date}: CIO decision={decision}, "
                    f"rank={cio.get('cio_rank')}, conviction={cio.get('cio_conviction')}, "
                    f"final_score={cio.get('final_score')}. "
                    f"Rationale: {cio.get('rationale') or '(none recorded)'}"
                ),
                "detail": {"cio": cio},
            }
        return {
            "ticker": ticker,
            "eval_date": date,
            "stage": "cio",
            "verdict": (
                f"{ticker} reached the CIO on {date} but was not advanced: "
                f"decision={decision or '(none)'}"
                + (f", tags={cio.get('rule_tags')}" if cio.get("rule_tags") else "")
                + f", final_score={cio.get('final_score')}. "
                f"Rationale: {cio.get('rationale') or '(none recorded)'}"
            ),
            "detail": {"cio": cio},
        }

    # Recommended by a team but no CIO row — unusual; report what we have.
    return {
        "ticker": ticker,
        "eval_date": date,
        "stage": "no_record",
        "verdict": (
            f"{ticker} was recommended by its sector team on {date} but has no "
            f"CIO evaluation row — the CIO stage may not have run, or the record "
            f"was not persisted."
        ),
        "detail": {"team_candidates": teams},
    }


def review_date(conn: sqlite3.Connection, eval_date: str | None = None) -> dict:
    """Funnel summary for one cycle: counts at each stage + the chosen set."""
    date = _resolve_date(conn, eval_date)
    scanned = _rows(
        conn,
        "SELECT COUNT(*) AS n, SUM(quant_filter_pass) AS passed "
        "FROM scanner_evaluations WHERE eval_date=?",
        (date,),
    )
    team_total = _rows(
        conn,
        "SELECT COUNT(*) AS n, SUM(team_recommended) AS recommended "
        "FROM team_candidates WHERE eval_date=?",
        (date,),
    )
    cio_rows = _rows(
        conn,
        "SELECT ticker, cio_decision, cio_rank, cio_conviction, final_score "
        "FROM cio_evaluations WHERE eval_date=? ORDER BY cio_rank",
        (date,),
    )
    advanced = [
        r for r in cio_rows if (r.get("cio_decision") or "").upper() in _ADVANCE_DECISIONS
    ]
    return {
        "eval_date": date,
        "scanner_screened": (scanned[0]["n"] if scanned else 0),
        "scanner_passed": (scanned[0]["passed"] if scanned and scanned[0]["passed"] else 0),
        "team_ranked": (team_total[0]["n"] if team_total else 0),
        "team_recommended": (
            team_total[0]["recommended"] if team_total and team_total[0]["recommended"] else 0
        ),
        "cio_evaluated": len(cio_rows),
        "cio_advanced": len(advanced),
        "advanced": advanced,
        "cio_all": cio_rows,
    }


# ── LLM-fallback Q&A (Phase 2) ──────────────────────────────────────────────
#
# The artifact-first commands above answer the bulk of "why did/didn't you …"
# questions straight from research.db. ``ask`` is the explicit fallback for
# what the structured store does NOT contain verbatim — the agent's free-text
# reasoning about a specific non-pick, or a question that needs synthesis
# across stages. It grounds an LLM in ALL captured evidence (the research.db
# review plus, optionally, the richer S3 ``decision_artifacts/`` snapshots) and
# instructs it to answer ONLY from that evidence, or to say so plainly and name
# what a fresh agent replay would need. No fabrication; no silent guess.

# Cap the serialized evidence fed to the model so a pathological artifact
# (snapshots run to ~1MB) can't blow the context window or the cost. The
# research.db review is always small; this bounds the optional artifacts.
_EVIDENCE_CHAR_CAP = 60_000

# Sector-team agent_ids whose decision_artifacts are relevant to a single
# ticker's fate, plus the cross-cutting CIO. ``{team}`` is filled from the
# ticker's team_candidates/cio row.
_TEAM_AGENT_TEMPLATES = (
    "sector_quant:{team}",
    "sector_qual:{team}",
    "sector_peer_review:{team}",
)


def _ticker_team_id(review: dict) -> str | None:
    """Best-effort team_id for a ticker from its review rows."""
    for t in review.get("team_candidates") or []:
        if t.get("team_id"):
            return t["team_id"]
    cio = review.get("cio")
    if cio and cio.get("team_id"):
        return cio["team_id"]
    return None


def fetch_decision_artifacts(eval_date: str, agent_ids: list[str]) -> dict:
    """Best-effort fetch of S3 ``decision_artifacts/`` for the given agents.

    Keyed by agent_id → artifact dict (newest under that day's prefix). Returns
    ``{}`` and never raises if capture is disabled, the bucket is unreachable,
    or nothing exists — the caller proceeds on research.db evidence alone. The
    run_id is not assumed (it may be ``run_date`` or a Lambda request id), so we
    list the agent's prefix for the day and take the most recent object."""
    try:
        import boto3  # lazy

        from config import AWS_REGION, S3_BUCKET  # lazy — avoids SSM in tests
    except Exception:  # pragma: no cover — defensive (no creds/config locally)
        return {}

    y, m, d = eval_date.split("-")
    s3 = boto3.client("s3", region_name=AWS_REGION)
    out: dict = {}
    for agent_id in agent_ids:
        prefix = f"decision_artifacts/{y}/{m}/{d}/{agent_id}/"
        try:
            resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
            objs = resp.get("Contents") or []
            if not objs:
                continue
            newest = max(objs, key=lambda o: o["LastModified"])
            body = s3.get_object(Bucket=S3_BUCKET, Key=newest["Key"])["Body"].read()
            out[agent_id] = json.loads(body)
        except Exception as e:  # pragma: no cover — per-agent best-effort
            print(f"(skipping {agent_id}: {e})", file=sys.stderr)
            continue
    return out


def gather_evidence(
    conn: sqlite3.Connection,
    ticker: str,
    eval_date: str | None = None,
    *,
    with_artifacts: bool = False,
) -> dict:
    """Assemble all stored evidence about a ticker for grounded Q&A.

    Always includes the research.db review. When ``with_artifacts`` is set,
    additionally pulls the relevant S3 decision_artifacts (sector team agents
    for the ticker's team + the CIO) — best-effort, omitted on any miss."""
    review = review_ticker(conn, ticker, eval_date)
    evidence: dict = {"review": review, "artifacts": {}}
    if with_artifacts:
        team = _ticker_team_id(review)
        agent_ids = ["ic_cio"]
        if team:
            agent_ids = [t.format(team=team) for t in _TEAM_AGENT_TEMPLATES] + agent_ids
        evidence["artifacts"] = fetch_decision_artifacts(review["eval_date"], agent_ids)
    return evidence


def has_evidence(evidence: dict) -> bool:
    """True if the store recorded anything about this ticker on this date."""
    r = evidence.get("review") or {}
    return bool(
        r.get("scanner") or r.get("team_candidates") or r.get("cio") or r.get("thesis")
        or evidence.get("artifacts")
    )


def build_qa_prompt(
    ticker: str, eval_date: str | None, question: str, evidence: dict
) -> tuple[str, str]:
    """Construct (system, user) messages for the grounded fallback answer."""
    system = (
        "You are an analyst reviewing the recorded decisions of an automated "
        "equity-research pipeline (sector-team quant/qual analysts → peer "
        "review → CIO). You are given the COMPLETE evidence the system stored "
        "about one ticker on one date. Answer the user's question using ONLY "
        "this evidence. Rules:\n"
        "1. Ground every claim in a specific field/value from the evidence; "
        "name it.\n"
        "2. If the evidence does not contain what's needed to answer, say so "
        "explicitly — do NOT speculate or invent reasoning the agents did not "
        "record. State what a fresh agent replay would need to answer it.\n"
        "3. Be concise and concrete. Prefer the recorded numbers (ranks, "
        "scores, gate flags, rationale text) over generic narrative."
    )
    serialized = json.dumps(evidence, default=str, indent=2)
    if len(serialized) > _EVIDENCE_CHAR_CAP:
        serialized = (
            serialized[:_EVIDENCE_CHAR_CAP]
            + "\n…[evidence truncated to fit context cap]…"
        )
    user = (
        f"Ticker: {ticker}\n"
        f"Eval date: {evidence.get('review', {}).get('eval_date', eval_date)}\n\n"
        f"=== RECORDED EVIDENCE ===\n{serialized}\n\n"
        f"=== QUESTION ===\n{question}"
    )
    return system, user


def _default_llm_fn(model: str):
    """Build the production LLM caller: ``(system, user) -> answer_text``."""
    def _call(system: str, user: str) -> str:
        from langchain_anthropic import ChatAnthropic  # lazy
        from langchain_core.messages import HumanMessage, SystemMessage  # lazy

        from config import ANTHROPIC_API_KEY  # lazy — avoids SSM in tests

        llm = ChatAnthropic(
            model=model,
            anthropic_api_key=ANTHROPIC_API_KEY,
            max_tokens=1024,
        )
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return resp.content if isinstance(resp.content, str) else str(resp.content)

    return _call


def answer_question(
    conn: sqlite3.Connection,
    ticker: str,
    question: str,
    eval_date: str | None = None,
    *,
    with_artifacts: bool = False,
    model: str | None = None,
    llm_fn=None,
) -> dict:
    """Grounded LLM fallback. Skips the LLM entirely when nothing is recorded
    about the ticker (returns the no-evidence verdict at $0).

    ``llm_fn`` is injectable as ``(system, user) -> str`` for testing; the
    default builds a ChatAnthropic caller on the configured strategic model."""
    ticker = ticker.upper()
    evidence = gather_evidence(conn, ticker, eval_date, with_artifacts=with_artifacts)
    date = evidence["review"]["eval_date"]

    if not has_evidence(evidence):
        return {
            "ticker": ticker,
            "eval_date": date,
            "question": question,
            "llm_called": False,
            "model": None,
            "answer": (
                f"No decision evidence recorded for {ticker} on {date} — nothing "
                f"to reason over. It was not in the screened universe that cycle, "
                f"or no cycle ran on this date. (No LLM call made.)"
            ),
            "artifacts_used": [],
        }

    chosen_model = model or _default_strategic_model()
    system, user = build_qa_prompt(ticker, date, question, evidence)
    call = llm_fn or _default_llm_fn(chosen_model)
    answer = call(system, user)
    return {
        "ticker": ticker,
        "eval_date": date,
        "question": question,
        "llm_called": True,
        "model": chosen_model,
        "answer": answer,
        "artifacts_used": sorted(evidence.get("artifacts", {}).keys()),
    }


def _default_strategic_model() -> str:
    """The configured Sonnet-tier model (synthesis quality for interactive
    Q&A). Lazy so the non-``ask`` commands never import config."""
    from config import STRATEGIC_MODEL  # lazy — avoids SSM in tests

    return STRATEGIC_MODEL


# ── DB source resolution ───────────────────────────────────────────────────


def open_db(db_path: str | None, pull: bool) -> sqlite3.Connection:
    """Open the research.db connection.

    ``--db PATH`` opens that file directly. Otherwise, with ``--pull`` (or
    when no local ``research.db`` exists), download it from S3 to a temp
    file. The S3 path imports ``config`` lazily so local/test invocations
    with ``--db`` never touch SSM/boto3."""
    if db_path:
        if not os.path.exists(db_path):
            raise SystemExit(f"error: --db path not found: {db_path}")
        return sqlite3.connect(db_path)

    repo_local = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "research.db"
    )
    if os.path.exists(repo_local) and not pull:
        return sqlite3.connect(repo_local)

    # Pull from S3.
    import boto3  # lazy

    from config import AWS_REGION, S3_BUCKET  # lazy — avoids SSM in tests

    tmp = os.path.join(tempfile.gettempdir(), "research_review.db")
    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.download_file(S3_BUCKET, _DB_S3_KEY, tmp)
    print(f"(pulled research.db from s3://{S3_BUCKET}/{_DB_S3_KEY})", file=sys.stderr)
    return sqlite3.connect(tmp)


# ── Rendering ───────────────────────────────────────────────────────────────


def _fmt_kv(d: dict, keys: list[str]) -> str:
    parts = []
    for k in keys:
        if k in d and d[k] is not None:
            parts.append(f"{k}={d[k]}")
    return ", ".join(parts)


def render_ticker(review: dict) -> str:
    lines = [f"=== {review['ticker']}  (eval_date={review['eval_date']}) ==="]

    sc = review["scanner"]
    lines.append("\n[scanner]")
    if sc is None:
        lines.append("  (no scanner_evaluations row — not in screened universe)")
    else:
        verdict = "PASS" if sc.get("quant_filter_pass") else "FAIL"
        lines.append(f"  quant_filter: {verdict}")
        lines.append(
            "  "
            + _fmt_kv(
                sc,
                [
                    "sector", "tech_score", "scan_path", "filter_fail_reason",
                    "liquidity_pass", "volatility_pass", "balance_sheet_pass",
                    "rsi_14", "atr_pct", "current_price", "avg_volume_20d",
                ],
            )
        )

    lines.append("\n[sector team(s)]")
    if not review["team_candidates"]:
        lines.append("  (no team_candidates row — not surfaced in any team's ranking)")
    for t in review["team_candidates"]:
        rec = "RECOMMENDED" if t.get("team_recommended") else "not recommended"
        lines.append(f"  {t.get('team_id')}: {rec}")
        lines.append(
            "    "
            + _fmt_kv(
                t,
                [
                    "quant_rank", "quant_score", "qual_score",
                    "rsi_sub_score", "macd_sub_score", "ma50_sub_score",
                    "ma200_sub_score", "momentum_sub_score",
                ],
            )
        )

    lines.append("\n[CIO]")
    cio = review["cio"]
    if cio is None:
        lines.append("  (no cio_evaluations row — did not reach the CIO)")
    else:
        lines.append(f"  decision={cio.get('cio_decision')}")
        lines.append(
            "  "
            + _fmt_kv(
                cio,
                ["team_id", "cio_rank", "cio_conviction", "final_score",
                 "combined_score", "macro_shift"],
            )
        )
        if cio.get("rule_tags"):
            lines.append(f"  rule_tags={cio['rule_tags']}")
        lines.append(f"  rationale: {cio.get('rationale') or '(none recorded)'}")

    lines.append("\n[thesis]")
    th = review["thesis"]
    if th is None:
        lines.append("  (no investment_thesis row for this date)")
    else:
        lines.append(
            "  "
            + _fmt_kv(th, ["rating", "score", "conviction", "signal",
                           "quant_score", "qual_score"])
        )
        if th.get("thesis_summary"):
            lines.append(f"  summary: {th['thesis_summary']}")

    return "\n".join(lines)


def render_why_not(result: dict) -> str:
    return (
        f"=== why-not {result['ticker']} (eval_date={result['eval_date']}) ===\n"
        f"stage: {result['stage']}\n"
        f"{result['verdict']}"
    )


def render_date(summary: dict) -> str:
    lines = [
        f"=== cycle {summary['eval_date']} ===",
        f"  scanner: {summary['scanner_screened']} screened → "
        f"{summary['scanner_passed']} passed quant filter",
        f"  teams:   {summary['team_ranked']} ranked → "
        f"{summary['team_recommended']} recommended",
        f"  CIO:     {summary['cio_evaluated']} evaluated → "
        f"{summary['cio_advanced']} advanced",
    ]
    if summary["advanced"]:
        lines.append("\n  advanced:")
        for r in summary["advanced"]:
            lines.append(
                f"    #{r.get('cio_rank')} {r['ticker']} "
                f"(decision={r.get('cio_decision')}, "
                f"conviction={r.get('cio_conviction')}, "
                f"final_score={r.get('final_score')})"
            )
    return "\n".join(lines)


def render_ask(result: dict) -> str:
    head = f"=== ask {result['ticker']} (eval_date={result['eval_date']}) ==="
    q = f"Q: {result['question']}"
    if result["llm_called"]:
        meta = f"(model={result['model']}"
        if result["artifacts_used"]:
            meta += f", artifacts={', '.join(result['artifacts_used'])}"
        meta += ")"
    else:
        meta = "(no LLM call — no recorded evidence)"
    return f"{head}\n{q}\n{meta}\n\n{result['answer']}"


# ── CLI ─────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="decision_review",
        description="Review research-pipeline agent decisions from research.db (L4567).",
    )
    p.add_argument("--db", help="Path to a local research.db (skips S3 pull).")
    p.add_argument(
        "--pull", action="store_true",
        help="Force a fresh download of research.db from S3.",
    )
    p.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    sub = p.add_subparsers(dest="command", required=True)

    pt = sub.add_parser("ticker", help="Full decision record for a ticker.")
    pt.add_argument("ticker")
    pt.add_argument("--date", help="eval_date (default: latest cycle).")

    pw = sub.add_parser("why-not", help="Where in the funnel a ticker was dropped.")
    pw.add_argument("ticker")
    pw.add_argument("--date", help="eval_date (default: latest cycle).")

    pd = sub.add_parser("date", help="Funnel summary for one cycle.")
    pd.add_argument("--date", help="eval_date (default: latest cycle).")

    pa = sub.add_parser(
        "ask",
        help="LLM fallback: ask a free-form question, grounded in the recorded "
             "evidence (only stage that may incur an LLM cost).",
    )
    pa.add_argument("ticker")
    pa.add_argument("question", help="Free-form question, e.g. \"why not a higher rank?\"")
    pa.add_argument("--date", help="eval_date (default: latest cycle).")
    pa.add_argument(
        "--with-artifacts", action="store_true",
        help="Also pull S3 decision_artifacts snapshots (needs capture enabled).",
    )
    pa.add_argument("--model", help="Override the LLM model (default: strategic/Sonnet).")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    conn = open_db(args.db, args.pull)
    try:
        if not _has_decision_tables(conn):
            raise SystemExit(
                "error: this research.db predates the evaluation schema "
                "(scanner_evaluations / team_candidates / cio_evaluations "
                "missing). Pull a current DB or run a cycle first."
            )

        if args.command == "ticker":
            result = review_ticker(conn, args.ticker, args.date)
            print(json.dumps(result, default=str, indent=2) if args.json
                  else render_ticker(result))
        elif args.command == "why-not":
            result = explain_why_not(conn, args.ticker, args.date)
            print(json.dumps(result, default=str, indent=2) if args.json
                  else render_why_not(result))
        elif args.command == "date":
            result = review_date(conn, args.date)
            print(json.dumps(result, default=str, indent=2) if args.json
                  else render_date(result))
        elif args.command == "ask":
            result = answer_question(
                conn, args.ticker, args.question, args.date,
                with_artifacts=args.with_artifacts, model=args.model,
            )
            print(json.dumps(result, default=str, indent=2) if args.json
                  else render_ask(result))
        else:  # pragma: no cover — argparse enforces a valid subcommand
            return 2
    finally:
        conn.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
