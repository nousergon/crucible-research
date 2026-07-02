"""Research Think Tank (data++) — open-source-model-native intel engine.

Standalone daily research pipeline (EPIC config#1579): maintains a coverage
ledger over the scanner attractiveness ranking, writes versioned company
theses, keeps macro/sector THEME theses on a seed → daily-update → weekly-
reconcile lifecycle, and sweeps news/events over covered names.

Boundaries (plan: alpha-engine-config/private-docs/research-thinktank-plan-260702.md):
- Writes ONLY to the ``thinktank/`` S3 namespace (+ the shared SFT corpus
  prefix). Never touches ``signals/`` or any existing consumer contract.
- Provider-agnostic by construction: OpenAI-compatible wire contract behind
  a per-tier model registry (``thinktank.yaml``); Anthropic/OpenAI/self-hosted
  vLLM are registry swaps, not code changes.
- Edge stays private: prompts load at runtime via ``agents.prompt_loader``
  (gitignored; real prompts in alpha-engine-config), tuned params live in
  the private ``alpha-engine-config/research/thinktank.yaml``.
"""

__version__ = "0.1.0"

SCHEMA_VERSION = 1

# S3 namespace (single source of truth for key templates)
LEDGER_KEY = "thinktank/coverage_ledger.json"
THESIS_KEY_TMPL = "thinktank/theses/{ticker}/v{version}.json"
THESIS_LATEST_TMPL = "thinktank/theses/{ticker}/latest.json"
THEME_KEY_TMPL = "thinktank/themes/{kind}/{key}/v{version}.json"
THEME_LATEST_TMPL = "thinktank/themes/{kind}/{key}/latest.json"
EVENTS_KEY_TMPL = "thinktank/events/{trading_day}.jsonl"
MANIFEST_KEY_TMPL = "thinktank/runs/{trading_day}/manifest_{run_id}.json"
COSTS_KEY_TMPL = "thinktank/costs/{month}.json"
SFT_PRODUCER = "crucible_thinktank"
