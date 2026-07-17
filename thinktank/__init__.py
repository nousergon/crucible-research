"""Research Think Tank (data++) — open-source-model-native intel engine.

Standalone daily research pipeline (EPIC config#1579): maintains a coverage
ledger over the scanner attractiveness ranking, writes versioned company
theses, keeps macro/sector THEME theses on a seed → daily-update → weekly-
reconcile lifecycle, and sweeps news/events over covered names.

Boundaries (plan: alpha-engine-config/private-docs/research-thinktank-plan-260702.md):
- Writes ONLY to the ``thinktank/`` S3 namespace (+ the shared SFT corpus
  prefix). Never touches ``signals/`` or any existing consumer contract.
  ONE narrow, deliberate exception (epic alpha-engine-config-I2515): the
  challenger-selection producer ALSO writes a conforming shadow view to
  ``signals_shadow/thinktank_coverage/{trading_day}/signals.json`` — the
  shape the shared champion/challenger leaderboard scorer
  (``scoring/leaderboard_producers.py``) already knows how to read. This is
  the OBSERVE-ONLY substrate other challenger producers already write to
  (never read by live trading/executor/predictor); see
  ``thinktank/challenger_selection.py``.
- Provider-agnostic by construction: OpenAI-compatible wire contract behind
  a per-tier model registry (``thinktank.yaml``); Anthropic/OpenAI/self-hosted
  vLLM are registry swaps, not code changes.
- Edge stays private: prompts load at runtime via ``agents.prompt_loader``
  (gitignored; real prompts in alpha-engine-config), tuned params live in
  the private ``alpha-engine-config/research/thinktank.yaml``.
"""

__version__ = "0.1.0"

# config#2678: bumped 1 -> 2 for CompanyThesis.pillar_assessment +
# RatingRow.raw_llm_rating (M0 contract-discipline: schema_version bumps on
# any shape change, per this module's own docstring). Additive-only — old
# artifacts (schema_version=1) still parse under both new fields' defaults.
SCHEMA_VERSION = 2

# S3 namespace (single source of truth for key templates)
LEDGER_KEY = "thinktank/coverage_ledger.json"
THESIS_KEY_TMPL = "thinktank/theses/{ticker}/v{version}.json"
THESIS_LATEST_TMPL = "thinktank/theses/{ticker}/latest.json"
THEME_KEY_TMPL = "thinktank/themes/{kind}/{key}/v{version}.json"
THEME_LATEST_TMPL = "thinktank/themes/{kind}/{key}/latest.json"
EVENTS_KEY_TMPL = "thinktank/events/{trading_day}.jsonl"
RATINGS_KEY_TMPL = "thinktank/ratings/{trading_day}.json"
RATINGS_LATEST_KEY = "thinktank/ratings/latest.json"
# Challenger-arm leaderboard submission (epic alpha-engine-config-I2515):
# Think Tank's top-N covered names by independent rating, written at the
# tail of every non-dry run — see thinktank/challenger_selection.py.
CHALLENGER_SELECTION_KEY_TMPL = "thinktank/challenger_selection/{trading_day}.json"
CHALLENGER_SELECTION_LATEST_KEY = "thinktank/challenger_selection/latest.json"
# Conforming shadow view for the shared champion/challenger leaderboard
# scorer (config#1221/#1223 substrate) — OUTSIDE the thinktank/ namespace by
# design, mirroring every other challenger producer's shadow key. Written
# ONLY when coverage_complete (see challenger_selection.py); not registered
# in producers/registry.py yet (registration tracked separately).
CHALLENGER_SHADOW_SIGNALS_KEY_TMPL = "signals_shadow/thinktank_coverage/{trading_day}/signals.json"
MANIFEST_KEY_TMPL = "thinktank/runs/{trading_day}/manifest_{run_id}.json"
COSTS_KEY_TMPL = "thinktank/costs/{month}.json"
# config#2678: per-ticker moat-assessment time series, INSIDE the
# thinktank/ namespace — Think Tank's own equivalent of the legacy
# archive/manager.py::save_moat_profile (archive/universe/{ticker}/...),
# which had no live producer since the qual/CIO graph left the weekly SF
# (config#1580). See thinktank/archive.py.
MOAT_PROFILE_KEY_TMPL = "thinktank/moat_profile/{ticker}.json"
SFT_PRODUCER = "crucible_thinktank"
