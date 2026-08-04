"""Think-tank configuration — loader for the private ``thinktank.yaml``.

Mirrors ``config.py``'s private-config-repo discovery (resolve via
``nousergon_lib.config.resolve_experiment_config``) so the real,
proprietary values live in ``alpha-engine-config/research/thinktank.yaml``
and never in this public repo. A tracked ``config/thinktank.sample.yaml``
documents the shape for open-source viewers only — it is NEVER loaded.

Explicit test/dev override: ``THINKTANK_CONFIG_PATH`` env var points at an
alternate YAML. This is an explicit operator/test knob (same spirit as
``ALPHA_ENGINE_SECRETS_SOURCE=env``), not a silent example-file fallback.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from nousergon_lib.config import resolve_experiment_config

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_BUCKET = os.environ.get(
    "RESEARCH_BUCKET", os.environ.get("S3_BUCKET", "alpha-engine-research")
)


@dataclass(frozen=True)
class ProviderSpec:
    """One OpenAI-compatible serving endpoint."""

    name: str
    base_url: str
    key_secret: str  # secret name resolved via nousergon_lib.secrets.get_secret


@dataclass(frozen=True)
class TierSpec:
    """One model tier (sweep / thesis / themes / pillar).

    A tier is addressed EITHER by capability group (``group``) or by a pinned
    ``provider`` + ``model`` — never both, never neither. Group addressing is
    the target state (alpha-engine-config-I6367, Brian's 2026-08-03 ruling:
    no agent directly linked to OpenRouter); the pinned form remains
    expressible so a tier can be held on a specific model deliberately, with
    that choice visible in the config rather than implied by silence.

    ``price_in_per_m`` / ``price_out_per_m`` are meaningful only for a pinned
    tier. Under group addressing the serving model is not known until the
    response returns, so prices come from ``krepis`` keyed on the model that
    actually served — a per-tier literal would bill the wrong card the moment
    the chain fell through.
    """

    name: str
    max_tokens: int
    group: str | None = None
    provider: str | None = None
    model: str | None = None
    price_in_per_m: float | None = None
    price_out_per_m: float | None = None
    structured_outputs: bool = False  # provider/model supports response_format json_schema

    @property
    def is_group_addressed(self) -> bool:
        return self.group is not None


@dataclass(frozen=True)
class ThinktankSettings:
    bucket: str
    daily_new_names: int
    rank_ceiling: int
    sweep_chunk_size: int
    stale_after_days: int
    monthly_budget_usd_default: float
    budget_ssm_param: str
    providers: dict[str, ProviderSpec] = field(default_factory=dict)
    tiers: dict[str, TierSpec] = field(default_factory=dict)

    def tier(self, name: str) -> TierSpec:
        try:
            return self.tiers[name]
        except KeyError:
            raise KeyError(
                f"thinktank.yaml defines no LLM tier '{name}' — "
                f"available: {sorted(self.tiers)}"
            ) from None

    def provider_for(self, tier: TierSpec) -> ProviderSpec:
        try:
            return self.providers[tier.provider]
        except KeyError:
            raise KeyError(
                f"tier '{tier.name}' references unknown provider "
                f"'{tier.provider}' — available: {sorted(self.providers)}"
            ) from None


def _config_path() -> Path:
    override = os.environ.get("THINKTANK_CONFIG_PATH")
    if override:
        path = Path(override)
        if not path.exists():
            raise FileNotFoundError(
                f"THINKTANK_CONFIG_PATH is set but does not exist: {path}"
            )
        return path
    return resolve_experiment_config(
        "research",
        "thinktank.yaml",
        repo_root=_REPO_ROOT,
        repo_local_fallback=_REPO_ROOT / "config" / "thinktank.yaml",
        github_workspace=True,
        resolve=True,
        error_message=(
            "Could not locate research/thinktank.yaml in alpha-engine-config. "
            "Checkout the config repo at ~/alpha-engine-config (local) or "
            "$GITHUB_WORKSPACE/alpha-engine-config (CI), or set "
            "THINKTANK_CONFIG_PATH explicitly."
        ),
    )


def _parse_tier(name: str, t: dict) -> TierSpec:
    """One tier, addressed by group XOR by pinned provider+model.

    Both-or-neither is REJECTED rather than resolved by precedence. A tier
    carrying a group and a pin would have one of them silently ignored, and
    which one is exactly the sort of fact that is discovered in an incident.
    """
    group = t.get("group")
    provider = t.get("provider")
    model = t.get("model")

    if group and (provider or model):
        raise ValueError(
            f"thinktank.yaml tier {name!r} declares BOTH group={group!r} and "
            f"a pinned provider/model — one would be silently ignored. "
            f"Address a tier by capability group OR by a specific model."
        )
    if not group and not (provider and model):
        raise ValueError(
            f"thinktank.yaml tier {name!r} declares neither a `group` nor a "
            f"complete `provider` + `model` pin — there is nothing to call."
        )

    if group:
        # Prices are NOT read for a group-addressed tier: the serving model
        # is a call-time fact, so cost is priced from the model that actually
        # served (krepis PriceCard). Rejecting the keys outright keeps a
        # stale literal from reading as authoritative.
        stale = [k for k in ("price_in_per_m", "price_out_per_m") if k in t]
        if stale:
            raise ValueError(
                f"thinktank.yaml tier {name!r} is group-addressed but still "
                f"carries {stale} — under group addressing the serving model "
                f"is not known until the response returns, and cost is priced "
                f"from it. A per-tier literal here would bill the wrong card "
                f"the moment the chain fell through."
            )
        return TierSpec(
            name=name,
            group=str(group),
            max_tokens=int(t["max_tokens"]),
            structured_outputs=bool(t.get("structured_outputs", False)),
        )

    return TierSpec(
        name=name,
        provider=str(provider),
        model=str(model),
        max_tokens=int(t["max_tokens"]),
        price_in_per_m=float(t["price_in_per_m"]),
        price_out_per_m=float(t["price_out_per_m"]),
        structured_outputs=bool(t.get("structured_outputs", False)),
    )


def load_settings() -> ThinktankSettings:
    """Parse thinktank.yaml into typed settings. Hard-fails on missing keys."""
    path = _config_path()
    with open(path) as f:
        raw = yaml.safe_load(f)
    tt = raw["thinktank"]

    # `providers` is OPTIONAL. A config whose tiers are all group-addressed
    # has no provider endpoints to declare, and requiring an empty block would
    # make the absence of direct provider linkage look like a malformed file
    # (alpha-engine-config-I6367 — no agent directly linked to OpenRouter).
    # Still hard-fails on a MALFORMED entry: a provider missing base_url or
    # key_secret is a mistake, and only its total absence is meaningful.
    providers = {
        name: ProviderSpec(name=name, base_url=p["base_url"], key_secret=p["key_secret"])
        for name, p in (tt["llm"].get("providers") or {}).items()
    }
    tiers = {
        name: _parse_tier(name, t)
        for name, t in tt["llm"]["tiers"].items()
    }

    coverage = tt["coverage"]
    budget = tt["budget"]
    settings = ThinktankSettings(
        bucket=os.environ.get("RESEARCH_BUCKET", os.environ.get("S3_BUCKET", tt.get("bucket", DEFAULT_BUCKET))),
        daily_new_names=int(os.environ.get("THINKTANK_DAILY_NEW_NAMES", coverage["daily_new_names"])),
        rank_ceiling=int(os.environ.get("THINKTANK_RANK_CEILING", coverage["rank_ceiling"])),
        sweep_chunk_size=int(coverage.get("sweep_chunk_size", 25)),
        stale_after_days=int(coverage.get("stale_after_days", 30)),
        monthly_budget_usd_default=float(budget["monthly_usd_default"]),
        budget_ssm_param=str(budget.get("ssm_param", "/thinktank/monthly_budget_usd")),
        providers=providers,
        tiers=tiers,
    )
    logger.info(
        "thinktank settings loaded from %s (tiers=%s, daily_new_names=%d, rank_ceiling=%d)",
        path,
        sorted(tiers),
        settings.daily_new_names,
        settings.rank_ceiling,
    )
    return settings
