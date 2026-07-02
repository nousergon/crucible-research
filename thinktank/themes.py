"""Theme theses (macro + per-sector) — seed → daily update → weekly reconcile.

Lifecycle (Brian, 2026-07-02, config#1579):
- SEED: first run derives the macro theme from the weekly macro report +
  market regime, and one theme per sector from ``sector_ratings``.
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
from thinktank.client import ThinktankClient
from thinktank.context import ContextBundle
from thinktank.schemas import ThemeThesis, ThemeThesisLLM
from thinktank.storage import ThinktankStore

logger = logging.getLogger(__name__)

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

    def macro_summary(self) -> str:
        theme = load_theme(self._store, "macro", "macro")
        if theme is None:
            return "No macro theme available."
        return f"[macro v{theme.version}, stance={theme.theme.stance}] {theme.theme.narrative}"

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
            macro_report=(self._ctx.macro_report_md or "(weekly macro report unavailable)")[:12000],
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
        )
        self._store_new(
            kind="macro", key="macro", prior=None, llm=result, reason="seed",
            weekly_anchor=weekly_date,
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
            )
            self._store_new(
                kind="sector", key=sector, prior=None, llm=result, reason="seed",
                weekly_anchor=weekly_date,
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
            macro_report=(self._ctx.macro_report_md or "(weekly macro report unavailable)")[:12000],
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
        )
        self._store_new(
            kind="macro", key="macro", prior=macro, llm=result, reason="reconcile",
            weekly_anchor=weekly_date,
            divergence=result.parsed.change_summary or None,
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
            )
            self._store_new(
                kind="sector", key=sector, prior=prior, llm=result, reason="reconcile",
                weekly_anchor=weekly_date,
                divergence=result.parsed.change_summary or None,
            )

    # ── daily churn-gated update ─────────────────────────────────────────────

    def _daily_update_macro(self, prior: ThemeThesis, developments: str) -> None:
        prompt = load_prompt("thinktank_theme_macro")
        rendered = prompt.format(
            mode="update",
            market_regime=self._ctx.market_regime(),
            macro_report="(unchanged since weekly anchor — see prior theme)",
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
        )
        if not result.parsed.material_change:
            logger.info("macro theme: no material change today — no version written")
            return
        self._store_new(
            kind="macro", key="macro", prior=prior, llm=result, reason="event",
            weekly_anchor=prior.weekly_anchor_date,
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
            model=llm.model,
            tier=llm.tier,
            cost_usd=llm.cost_usd,
        )
        _write_theme(self._store, theme)
        self.updates_written += 1
