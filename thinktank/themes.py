"""Theme theses (macro + per-sector) — seed → daily update → weekly reconcile.

Lifecycle (Brian, 2026-07-02, config#1579):
- SEED: first run derives the macro theme from the macro ANCHOR (the weekly
  regime substrate + the news-aggregate backdrop, see ``_macro_anchor_block``)
  plus the market regime, and one theme per sector from ``sector_ratings``.
- DAILY UPDATE (churn-disciplined): intraweek developments (e.g. a Thursday
  jobs report surfaced through the news sweep) go into the theme the same
  day — but ONLY when the model marks ``material_change``; otherwise no new
  version is written and the no-change outcome is logged.
- RECONCILE: when a NEW weekly ``signals.json`` date appears, themes are
  re-anchored to the weekly analysis (authoritative for now), with any
  intraweek divergence noted in ``divergence_from_weekly``.

Per-name thesis calls consume the CURRENT themes as context — sector/macro
work is done once here, not per ticker.
"""

from __future__ import annotations

import logging

from agents.prompt_loader import load_prompt
from thinktank import THEME_KEY_TMPL, THEME_LATEST_TMPL
from thinktank.capture import emit_theme_capture
from thinktank.client import ThinktankClient
from thinktank.context import ContextBundle
from thinktank.schemas import ThemeThesis, ThemeThesisLLM
from thinktank.storage import ThinktankStore

logger = logging.getLogger(__name__)


def _anchor_kwargs(anchor: str) -> dict[str, str]:
    """The macro-anchor text under BOTH the new and legacy placeholder names.

    ``LoadedPrompt.format`` is ``str.format``: a template referencing a name
    the caller did not pass raises ``KeyError``, while an extra kwarg the
    template ignores costs nothing. The prompt template lives in the PRIVATE
    ``alpha-engine-config`` repo and the Think Tank spot box clones BOTH repos'
    ``main`` fresh on every nightly run — so a rename done in one repo alone is
    broken in whichever direction lands first, with no safe merge order
    (mutually exclusive by construction). Passing both names makes the two
    merges independent instead.

    RETIRE with the legacy ``macro_report`` key once the config-side prompt has
    cut over to ``{macro_anchor}`` — tracked as alpha-engine-config-I7962.
    """
    return {"macro_anchor": anchor, "macro_report": anchor}

TIER = "themes"

_MACRO_SYSTEM = (
    "You are the macro strategist of an investment research team. You maintain "
    "a living macro/regime thesis. Be specific, cite the inputs you were given, "
    "and be conservative about declaring change: material_change is true ONLY "
    "when new information genuinely alters positioning-relevant conclusions."
)
_SECTOR_SYSTEM = (
    "You are a sector strategist maintaining a living sector thesis. Be "
    "specific and conservative about declaring change: material_change is "
    "true ONLY when new information genuinely alters the sector view."
)


def _slug(key: str) -> str:
    return key.lower().replace(" ", "_").replace("/", "_")


def load_theme(store: ThinktankStore, kind: str, key: str) -> ThemeThesis | None:
    raw = store.get_json(THEME_LATEST_TMPL.format(kind=kind, key=_slug(key)))
    return ThemeThesis.model_validate(raw) if raw is not None else None


def _write_theme(store: ThinktankStore, theme: ThemeThesis) -> None:
    payload = theme.model_dump()
    slug = _slug(theme.key)
    store.put_json(
        THEME_KEY_TMPL.format(kind=theme.kind, key=slug, version=theme.version), payload
    )
    store.put_json(THEME_LATEST_TMPL.format(kind=theme.kind, key=slug), payload)


class ThemeKeeper:
    """Maintains the macro theme + one theme per sector for a run."""

    def __init__(
        self,
        store: ThinktankStore,
        client: ThinktankClient,
        ctx: ContextBundle,
        *,
        trading_day: str,
        calendar_date: str,
    ) -> None:
        self._store = store
        self._client = client
        self._ctx = ctx
        self._trading_day = trading_day
        self._calendar_date = calendar_date
        self.updates_written = 0
        self.reconciled = False

    # ── public entrypoints ───────────────────────────────────────────────────

    def ensure_current(self, daily_developments: str = "") -> None:
        """Seed themes if absent, reconcile if a new weekly landed, then apply
        the daily churn-gated update when developments were observed."""
        weekly_date = self._ctx.weekly_signals_date()
        macro = load_theme(self._store, "macro", "macro")

        if macro is None:
            self._seed_all(weekly_date)
            macro = load_theme(self._store, "macro", "macro")
        elif weekly_date and macro.weekly_anchor_date != weekly_date:
            self._reconcile_all(macro, weekly_date)
            macro = load_theme(self._store, "macro", "macro")

        if daily_developments.strip() and macro is not None:
            self._daily_update_macro(macro, daily_developments)

    def _macro_anchor_block(self) -> str:
        """The macro ANCHOR as handed to the model: the weekly regime
        substrate plus the news-aggregate backdrop, each prefixed with an
        explicit staleness banner when it is out of date.

        alpha-engine-config-I2638, Brian ruling 2026-08-21: this used to be
        ``archive/macro/macro_report.md``, whose producer lost its call site
        when the multi-agent graph retired. The object froze at 2026-03-16 and
        this prompt kept presenting it as "the weekly macro report" while the
        model reconciled to a five-month-old backdrop. Rather than resurrect a
        retired agent path, Think Tank self-anchors on inputs with live
        producers. The banners are the in-band half of the staleness fix; the
        ``stale_inputs`` field on every written theme is the out-of-band half.
        """
        parts = [
            self._banner_for(
                "regime_substrate",
                "Treat the regime block below as a DATED reference, not as "
                "current conditions: prefer the market regime and intraweek "
                "developments you were given, and say so in change_summary if "
                "the two conflict.",
            ),
            _render_regime_substrate(self._ctx.regime_substrate),
            "",
            self._banner_for(
                "news_aggregates",
                "The news backdrop below is DATED — do not read the absence of "
                "recent events as calm.",
            ),
            _render_news_backdrop(self._ctx.news_by_ticker),
        ]
        return "\n".join(part for part in parts if part)

    def _banner_for(self, source: str, guidance: str) -> str:
        """Staleness banner + handling guidance for one anchor leg, or ``""``
        when that leg is fresh. A source with NO verdict also returns ``""`` —
        ``context.load_context`` records a verdict for every anchor leg, so a
        missing one is a wiring bug, not a silent pass; the loud surface for
        that is the manifest's ``context_source_freshness`` gap, not a banner
        this method would have to invent.
        """
        verdict = self._ctx.freshness.get(source)
        if verdict is None or verdict.is_fresh:
            return ""
        return f"{verdict.banner()}\n{guidance}\n"

    def macro_summary(self) -> str:
        theme = load_theme(self._store, "macro", "macro")
        if theme is None:
            return "No macro theme available."
        stale = (
            f" [DEGRADED: {len(theme.stale_inputs)} stale upstream input(s) — "
            f"{', '.join(sorted(r.get('artifact', '?') for r in theme.stale_inputs))}]"
            if theme.stale_inputs
            else ""
        )
        return (
            f"[macro v{theme.version}, stance={theme.theme.stance}]{stale} "
            f"{theme.theme.narrative}"
        )

    def sector_summary(self, sector: str | None) -> str:
        if not sector:
            return "No sector theme available."
        theme = load_theme(self._store, "sector", sector)
        if theme is None:
            return f"No theme on file for sector {sector}."
        return f"[{sector} v{theme.version}, stance={theme.theme.stance}] {theme.theme.narrative}"

    def theme_versions(self) -> tuple[int | None, dict[str, int]]:
        macro = load_theme(self._store, "macro", "macro")
        sectors: dict[str, int] = {}
        for sector in self._ctx.sector_ratings():
            t = load_theme(self._store, "sector", sector)
            if t is not None:
                sectors[sector] = t.version
        return (macro.version if macro else None, sectors)

    # ── seed ─────────────────────────────────────────────────────────────────

    def _seed_all(self, weekly_date: str | None) -> None:
        logger.info("seeding themes from weekly artifacts (weekly=%s)", weekly_date)
        prompt = load_prompt("thinktank_theme_macro")
        rendered = prompt.format(
            mode="seed",
            market_regime=self._ctx.market_regime(),
            **_anchor_kwargs(self._macro_anchor_block()),
            prior_theme="(none — first seed)",
            developments="(none)",
        )
        result = self._client.complete(
            TIER,
            agent_id="themes_macro",
            system=_MACRO_SYSTEM,
            user=rendered,
            response_model=ThemeThesisLLM,
            prompt_id=prompt.name,
            prompt_version=prompt.version,
            sft_meta={"kind": "macro", "key": "macro", "update_reason": "seed",
                      "theme_version": 1, "trading_day": self._trading_day},
        )
        self._store_new(
            kind="macro", key="macro", prior=None, llm=result, reason="seed",
            weekly_anchor=weekly_date,
            system=_MACRO_SYSTEM, user=rendered, prompt_hash=prompt.hash,
        )

        sector_prompt = load_prompt("thinktank_theme_sector")
        for sector, rating in self._ctx.sector_ratings().items():
            rendered = sector_prompt.format(
                mode="seed",
                sector=sector,
                sector_rating=str(rating),
                market_regime=self._ctx.market_regime(),
                macro_summary=self.macro_summary(),
                prior_theme="(none — first seed)",
                developments="(none)",
            )
            result = self._client.complete(
                TIER,
                agent_id="themes_sector",
                system=_SECTOR_SYSTEM,
                user=rendered,
                response_model=ThemeThesisLLM,
                prompt_id=sector_prompt.name,
                prompt_version=sector_prompt.version,
                sft_meta={"kind": "sector", "key": sector, "update_reason": "seed",
                          "theme_version": 1, "trading_day": self._trading_day},
            )
            self._store_new(
                kind="sector", key=sector, prior=None, llm=result, reason="seed",
                weekly_anchor=weekly_date,
                system=_SECTOR_SYSTEM, user=rendered, prompt_hash=sector_prompt.hash,
            )

    # ── reconcile (weekly is the authoritative anchor) ───────────────────────

    def _reconcile_all(self, macro: ThemeThesis, weekly_date: str) -> None:
        logger.info(
            "reconciling themes to new weekly run %s (prior anchor %s)",
            weekly_date,
            macro.weekly_anchor_date,
        )
        self.reconciled = True
        prompt = load_prompt("thinktank_theme_macro")
        rendered = prompt.format(
            mode="reconcile",
            market_regime=self._ctx.market_regime(),
            **_anchor_kwargs(self._macro_anchor_block()),
            prior_theme=macro.theme.model_dump_json(),
            developments="(reconcile to the new weekly analysis; note any divergence "
            "between your intraweek view and the weekly report)",
        )
        result = self._client.complete(
            TIER,
            agent_id="themes_macro",
            system=_MACRO_SYSTEM,
            user=rendered,
            response_model=ThemeThesisLLM,
            prompt_id=prompt.name,
            prompt_version=prompt.version,
            sft_meta={"kind": "macro", "key": "macro", "update_reason": "reconcile",
                      "theme_version": macro.version + 1,
                      "trading_day": self._trading_day},
        )
        self._store_new(
            kind="macro", key="macro", prior=macro, llm=result, reason="reconcile",
            weekly_anchor=weekly_date,
            divergence=result.parsed.change_summary or None,
            system=_MACRO_SYSTEM, user=rendered, prompt_hash=prompt.hash,
        )

        sector_prompt = load_prompt("thinktank_theme_sector")
        for sector, rating in self._ctx.sector_ratings().items():
            prior = load_theme(self._store, "sector", sector)
            rendered = sector_prompt.format(
                mode="reconcile",
                sector=sector,
                sector_rating=str(rating),
                market_regime=self._ctx.market_regime(),
                macro_summary=self.macro_summary(),
                prior_theme=prior.theme.model_dump_json() if prior else "(none)",
                developments="(reconcile to the new weekly analysis)",
            )
            result = self._client.complete(
                TIER,
                agent_id="themes_sector",
                system=_SECTOR_SYSTEM,
                user=rendered,
                response_model=ThemeThesisLLM,
                prompt_id=sector_prompt.name,
                prompt_version=sector_prompt.version,
                sft_meta={"kind": "sector", "key": sector,
                          "update_reason": "reconcile",
                          "theme_version": (prior.version + 1) if prior else 1,
                          "trading_day": self._trading_day},
            )
            self._store_new(
                kind="sector", key=sector, prior=prior, llm=result, reason="reconcile",
                weekly_anchor=weekly_date,
                divergence=result.parsed.change_summary or None,
                system=_SECTOR_SYSTEM, user=rendered, prompt_hash=sector_prompt.hash,
            )

    # ── daily churn-gated update ─────────────────────────────────────────────

    def _daily_update_macro(self, prior: ThemeThesis, developments: str) -> None:
        prompt = load_prompt("thinktank_theme_macro")
        rendered = prompt.format(
            mode="update",
            market_regime=self._ctx.market_regime(),
            **_anchor_kwargs("(unchanged since weekly anchor — see prior theme)"),
            prior_theme=prior.theme.model_dump_json(),
            developments=developments[:8000],
        )
        result = self._client.complete(
            TIER,
            agent_id="themes_macro",
            system=_MACRO_SYSTEM,
            user=rendered,
            response_model=ThemeThesisLLM,
            prompt_id=prompt.name,
            prompt_version=prompt.version,
            sft_meta={"kind": "macro", "key": "macro", "update_reason": "event",
                      "theme_version": prior.version + 1,
                      "trading_day": self._trading_day},
        )
        if not result.parsed.material_change:
            logger.info("macro theme: no material change today — no version written")
            return
        self._store_new(
            kind="macro", key="macro", prior=prior, llm=result, reason="event",
            weekly_anchor=prior.weekly_anchor_date,
            system=_MACRO_SYSTEM, user=rendered, prompt_hash=prompt.hash,
        )

    # ── shared writer ────────────────────────────────────────────────────────

    def _store_new(
        self,
        *,
        kind: str,
        key: str,
        prior: ThemeThesis | None,
        llm,
        reason: str,
        weekly_anchor: str | None,
        divergence: str | None = None,
        system: str = "",
        user: str = "",
        prompt_hash: str | None = None,
    ) -> None:
        theme = ThemeThesis(
            kind=kind,  # type: ignore[arg-type]
            key=key,
            version=(prior.version + 1) if prior else 1,
            trading_day=self._trading_day,
            calendar_date=self._calendar_date,
            update_reason=reason,  # type: ignore[arg-type]
            theme=llm.parsed,
            weekly_anchor_date=weekly_anchor,
            divergence_from_weekly=divergence,
            # Written unconditionally (empty ⇒ checked and fresh), so a reader
            # of a theme — or the producer leaderboard scoring this arm — can
            # never mistake a degraded run for a healthy one.
            stale_inputs=self._ctx.stale_input_records(),
            model=llm.model,
            tier=llm.tier,
            cost_usd=llm.cost_usd,
        )
        _write_theme(self._store, theme)
        emit_theme_capture(
            base_run_id=self._client.run_id,
            kind=kind,
            key_slug=_slug(key),
            version=theme.version,
            trading_day=self._trading_day,
            result=llm,
            system=system,
            user=user,
            prompt_version_hash=prompt_hash,
            input_data_snapshot={
                "kind": kind,
                "key": key,
                "update_reason": reason,
                "market_regime": self._ctx.market_regime(),
                "weekly_anchor_date": weekly_anchor,
                "prior_theme": prior.theme.model_dump() if prior else None,
            },
            agent_output=theme.model_dump(),
            bucket=self._store.bucket,
            s3_client=self._store.s3,
        )
        self.updates_written += 1


# ── macro anchor renderers ───────────────────────────────────────────────────
#
# Deterministic, model-free rendering of the two anchor legs. Kept at module
# scope (not methods) so they are unit-testable against a payload dict without
# constructing a ThemeManager, and so the sector/per-name prompts can reuse
# them if they ever need the same backdrop.

#: Cap on the news backdrop's event list — the anchor is a BACKDROP, not the
#: sweep. The sweep (``analyst.sweep``) is what reads events per name; this
#: block exists so the seed/reconcile calls, which run before any sweep
#: developments exist, are not handed a regime block with no narrative at all.
_MAX_BACKDROP_EVENTS = 12


def _fmt(value, digits: int = 2) -> str:
    """Render a numeric field, or ``n/a`` when it is absent/non-numeric.

    ``None`` is rendered as ``n/a`` rather than ``0`` deliberately: a composite
    ``intensity_z`` of ``None`` means a total macro-feature outage upstream
    (config-I7272), which is undefined, not a measured calm reading.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "n/a"
    return f"{value:.{digits}f}"


def _render_regime_substrate(substrate: dict | None) -> str:
    """The weekly regime substrate as prompt-safe prose + key/value lines."""
    if not isinstance(substrate, dict):
        return (
            "QUANTITATIVE REGIME SUBSTRATE: not available this run. Treat the "
            "regime picture as UNOBSERVED — not as neutral — and say so in the "
            "thesis rather than asserting a regime you were not given evidence for."
        )

    hmm = substrate.get("hmm") or {}
    probs = hmm.get("probs") or {}
    composite = substrate.get("composite") or {}
    bocpd = substrate.get("bocpd") or {}
    guardrails = substrate.get("guardrails") or {}

    lines = [
        "QUANTITATIVE REGIME SUBSTRATE (weekly, produced by the RegimeSubstrate "
        "stage of the Saturday pipeline):",
        f"- as-of: calendar_date={substrate.get('calendar_date', '?')} "
        f"trading_day={substrate.get('trading_day', '?')} "
        f"run_id={substrate.get('run_id', '?')}",
        f"- HMM state: {hmm.get('argmax', '?')} "
        f"(bear {_fmt(probs.get('bear'))} / neutral {_fmt(probs.get('neutral'))} "
        f"/ bull {_fmt(probs.get('bull'))}), "
        f"{hmm.get('weeks_in_current_state', '?')} week(s) in this state",
        f"- composite intensity_z: {_fmt(composite.get('intensity_z'))} "
        f"(positive = risk-on), implied severity "
        f"{composite.get('implied_severity', 'n/a')}",
    ]

    per_feature = composite.get("per_feature_z")
    if isinstance(per_feature, dict) and per_feature:
        ranked = sorted(
            (
                (name, z)
                for name, z in per_feature.items()
                if isinstance(z, (int, float)) and not isinstance(z, bool)
            ),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )[:5]
        if ranked:
            lines.append(
                "- largest feature deviations (z): "
                + ", ".join(f"{name} {z:+.2f}" for name, z in ranked)
            )

    if bocpd:
        lines.append(
            f"- change-point signal: {bocpd.get('change_signal', 'n/a')} "
            f"(change confidence {_fmt(bocpd.get('change_confidence'))}, "
            f"settled-regime mass {_fmt(bocpd.get('max_runlength_prob'))})"
        )

    # ``guardrails`` mixes threshold-crossing booleans with the string-valued
    # ``active_severity_floor``; ``is True`` keeps the two apart rather than
    # letting a non-empty string render as a tripped flag.
    tripped = sorted(k for k, v in guardrails.items() if v is True)
    lines.append(
        "- guardrails tripped: " + (", ".join(tripped) if tripped else "none")
    )
    floor = guardrails.get("active_severity_floor")
    lines.append(f"- active severity floor: {floor or 'none'}")

    effective = substrate.get("effective_regime")
    if isinstance(effective, dict) and effective:
        lines.append(
            "- blended effective regime: "
            f"{effective.get('regime_score_categorical', 'n/a')} "
            f"(score {_fmt(effective.get('regime_score'))})"
        )

    features = substrate.get("features")
    if isinstance(features, dict) and features:
        lines.append(
            "- raw macro features: "
            + ", ".join(f"{k}={_fmt(v, 4)}" for k, v in sorted(features.items()))
        )

    return "\n".join(lines)


def _render_news_backdrop(news_by_ticker: dict[str, dict]) -> str:
    """A cross-sectional digest of the daily news aggregates.

    The per-name detail belongs to the sweep; what the macro theme needs is the
    shape of the tape — how wide the coverage is, which way sentiment leans,
    and what the most severe events were.
    """
    rows = [r for r in (news_by_ticker or {}).values() if isinstance(r, dict)]
    if not rows:
        return (
            "NEWS BACKDROP: no news aggregates available this run. Treat the "
            "intraweek picture as UNOBSERVED, not as quiet."
        )

    sentiments = [
        r.get("lm_sentiment_trusted_mean")
        if isinstance(r.get("lm_sentiment_trusted_mean"), (int, float))
        else r.get("lm_sentiment_mean")
        for r in rows
    ]
    sentiments = [
        s for s in sentiments if isinstance(s, (int, float)) and not isinstance(s, bool)
    ]
    articles = sum(
        r.get("n_articles") or 0
        for r in rows
        if isinstance(r.get("n_articles"), (int, float))
    )
    dates = sorted({str(r["aggregate_date"]) for r in rows if r.get("aggregate_date")})

    category_counts: dict[str, int] = {}
    for r in rows:
        cats = r.get("event_categories")
        if isinstance(cats, str):
            cats = [c.strip() for c in cats.split(",") if c.strip()]
        if isinstance(cats, (list, tuple)):
            for c in cats:
                category_counts[str(c)] = category_counts.get(str(c), 0) + 1

    lines = [
        "NEWS BACKDROP (daily aggregates, cross-sectional):",
        f"- coverage: {len(rows)} name(s), {int(articles)} article(s), "
        f"aggregate_date {dates[-1] if dates else '?'}"
        + (f" (oldest row {dates[0]})" if len(dates) > 1 else ""),
    ]
    if sentiments:
        mean = sum(sentiments) / len(sentiments)
        negative = sum(1 for s in sentiments if s < 0)
        lines.append(
            f"- sentiment: mean {mean:+.3f} across {len(sentiments)} name(s); "
            f"{negative} negative ({negative / len(sentiments):.0%})"
        )
    if category_counts:
        top = sorted(category_counts.items(), key=lambda kv: kv[1], reverse=True)[:6]
        lines.append(
            "- most common event categories: "
            + ", ".join(f"{name} ({n})" for name, n in top)
        )

    severe = sorted(
        (
            r
            for r in rows
            if isinstance(r.get("event_severity_max"), (int, float))
            and not isinstance(r.get("event_severity_max"), bool)
        ),
        key=lambda r: r["event_severity_max"],
        reverse=True,
    )[:_MAX_BACKDROP_EVENTS]
    if severe:
        lines.append("- most severe events:")
        for r in severe:
            desc = str(r.get("top_event_descriptions") or "").strip().replace("\n", " ")
            lines.append(
                f"  - {r.get('ticker', '?')} "
                f"(severity {_fmt(r['event_severity_max'], 1)}): {desc[:240] or 'n/a'}"
            )

    return "\n".join(lines)
