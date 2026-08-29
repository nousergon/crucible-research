"""Think-tank configuration — loader for the private ``thinktank.yaml``.

Mirrors ``config.py``'s private-config-repo discovery (resolve via
``nousergon_lib.config.resolve_experiment_config``) so the real,
proprietary values live in ``alpha-engine-config/research/thinktank.yaml``
and never in this public repo. A tracked ``config/thinktank.sample.yaml``
documents the shape for open-source viewers only — it is NEVER loaded.

Explicit test/dev override: ``THINKTANK_CONFIG_PATH`` env var points at an
alternate YAML. This is an explicit operator/test knob (same spirit as
``ALPHA_ENGINE_SECRETS_SOURCE=env``), not a silent example-file fallback.

**alpha-engine-config-I9302 (item 3), 2026-08-29:** every tier is addressed
by capability GROUP only. Through 2026-08-29 this module also accepted a
pinned ``provider`` + ``model`` form (its own ``ProviderSpec``, its own price
literals) that bypassed the krepis router entirely — a second routing plane,
measured DORMANT (the live private config carries no ``providers:`` block,
so it had reinstated direct provider linkage on one config edit, with no
guard between). Brian's ruling that day was unconditional: "the entire nous
ergon system should now be running through the krepis router ... we should
have no other parallel setups." I9302 itself: "Prefer deletion to a guard: a
code path that only a config edit can reach is cheaper to remove than to
police." Removed rather than migrated onto ``krepis.router.resolve_model_spec``
(the pinned-model sibling of ``resolve_group_spec``) because the model IDs
this schema carried (``openai/gpt-oss-120b``, raw OpenRouter slugs) are not
krepis registry entries — wiring that up is a registry change in
``alpha-engine-config``, out of this repo's scope, and every live tier is
already group-addressed with no case pending that needs it.
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

_RETIRED_PINNED_KEYS = ("provider", "model", "price_in_per_m", "price_out_per_m")


@dataclass(frozen=True)
class TierSpec:
    """One model tier (sweep / thesis / themes / pillar).

    Addressed by capability GROUP only — the ``group`` field names a
    ``low|med|high|ultra`` registry tier resolved through
    ``krepis.router.resolve_group_spec``. Which model serves it, at which
    endpoint, with which credential, is a registry decision resolved above
    this file (alpha-engine-config-I6367, I9302).
    """

    name: str
    max_tokens: int
    group: str
    structured_outputs: bool = False  # provider/model supports response_format json_schema


@dataclass(frozen=True)
class ThinktankSettings:
    bucket: str
    daily_new_names: int
    rank_ceiling: int
    sweep_chunk_size: int
    stale_after_days: int
    monthly_budget_usd_default: float
    budget_ssm_param: str
    tiers: dict[str, TierSpec] = field(default_factory=dict)
    # alpha-engine-config-I6648 / products/thinktank.md §2.1. `rank_ceiling`
    # is the ENTER threshold; this is the strictly-wider EXIT threshold that
    # makes the covered set a declared set rather than a monotonic ratchet.
    # Optional so an un-migrated config keeps today's behaviour (no exit path)
    # rather than silently de-covering on first load.
    exit_rank: int | None = None

    def tier(self, name: str) -> TierSpec:
        try:
            return self.tiers[name]
        except KeyError:
            raise KeyError(
                f"thinktank.yaml defines no LLM tier '{name}' — "
                f"available: {sorted(self.tiers)}"
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


def _parse_exit_rank(coverage: dict) -> int | None:
    """`coverage.exit_rank`, validated STRICTLY wider than the enter rank.

    Fails LOUD at config load rather than at the first drop
    (alpha-engine-config-I6648). An exit rank at or inside the enter rank is
    not a narrow band, it is a de-covering loop: a name enters at rank N and
    is dropped on the same ranking, every run, forever — and the symptom would
    be churn in the coverage ledger a long way from the config that caused it.

    Absent ⇒ ``None`` ⇒ no exit path, which is the pre-I6648 behaviour. An
    un-migrated config must keep working; it must not start de-covering names
    because a default appeared.
    """
    raw = os.environ.get("THINKTANK_EXIT_RANK", coverage.get("exit_rank"))
    if raw is None or raw == "":
        return None
    exit_rank = int(raw)
    enter_rank = int(
        os.environ.get("THINKTANK_RANK_CEILING", coverage["rank_ceiling"])
    )
    if exit_rank <= enter_rank:
        raise ValueError(
            f"thinktank coverage.exit_rank={exit_rank} must be STRICTLY "
            f"greater than rank_ceiling={enter_rank} — an exit rank at or "
            "inside the enter rank de-covers every name it just admitted, "
            "every run (products/thinktank.md §2.1, config-I6648)"
        )
    return exit_rank


def _parse_tier(name: str, t: dict) -> TierSpec:
    """One tier, addressed by capability group. Fails loud, never silently.

    A tier authored against the retired pinned-provider schema (``provider``
    / ``model`` / ``price_in_per_m`` / ``price_out_per_m``, removed
    alpha-engine-config-I9302) is REJECTED rather than having those keys
    silently dropped — a silent drop would read as "group addressing was
    chosen" when nobody chose it, and the config would keep loading with the
    model choice it documents quietly discarded.
    """
    stale = [k for k in _RETIRED_PINNED_KEYS if k in t]
    if stale:
        raise ValueError(
            f"thinktank.yaml tier {name!r} carries {stale} — the pinned "
            f"provider/model addressing mode was removed (alpha-engine-"
            f"config-I9302, Brian's 2026-08-29 ruling: no parallel routing "
            f"plane outside the krepis router). Address the tier by "
            f"`group` (low|med|high|ultra) only."
        )

    group = t.get("group")
    if not group:
        raise ValueError(
            f"thinktank.yaml tier {name!r} declares no `group` — there is "
            f"nothing to call. Every tier is addressed by capability group."
        )

    return TierSpec(
        name=name,
        group=str(group),
        max_tokens=int(t["max_tokens"]),
        structured_outputs=bool(t.get("structured_outputs", False)),
    )


def load_settings() -> ThinktankSettings:
    """Parse thinktank.yaml into typed settings. Hard-fails on missing keys."""
    path = _config_path()
    with open(path) as f:
        raw = yaml.safe_load(f)
    tt = raw["thinktank"]

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
        exit_rank=_parse_exit_rank(coverage),
        sweep_chunk_size=int(coverage.get("sweep_chunk_size", 25)),
        stale_after_days=int(coverage.get("stale_after_days", 30)),
        monthly_budget_usd_default=float(budget["monthly_usd_default"]),
        budget_ssm_param=str(budget.get("ssm_param", "/thinktank/monthly_budget_usd")),
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
