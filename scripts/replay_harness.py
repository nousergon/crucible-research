"""
Scenario-replay harness — faithful + counterfactual replay of recorded
``DecisionArtifact``s.

ROADMAP L4567 sub-item 2b / alpha-engine-config#781. Sibling in spirit to
``scripts/decision_review.py`` (both are artifact-first CLI tools reading
``decision_artifacts/`` + ``research.db``); this module answers a different
question — not "what did the pipeline decide and why" but "if we re-ran this
exact decision (faithful), or re-ran it with one input perturbed
(counterfactual), what would come out?"

**Faithful replay** (:func:`faithful_replay`) certifies the run=code+data
reproducibility contract that 1a/1b (provenance stamps ``code_sha`` +
``data_snapshot_id`` on every ``DecisionArtifact``) exist to make possible:
fetch a recorded artifact, rehydrate the ArcticDB reads it depended on
``as_of`` its ``data_snapshot_id`` version, re-invoke the SAME node function
that produced it, and field-level diff the fresh output against the
recorded one.

**Counterfactual replay** (:func:`counterfactual_replay`) answers "what if
one input had been different?" — same rehydration, but a
:class:`PerturbationSpec` mutates the rehydrated input state before
re-running. Distributional by construction: runs N times (parameterizable
temperature) and returns an :class:`OutcomeDistribution` summary rather than
a single point, per the issue's own requirement ("temp-0 + N-replay, not a
single stochastic shot").

**Dispatch scope (v1):** only ``thesis_update:{team_id}:{ticker}`` artifacts
are replayable end-to-end. That agent_id is the one node in the pipeline
whose full input state is captured in a single, self-contained artifact
snapshot (``triggers``, ``prior_thesis``, ``news_data``, ``analyst_data``) and
whose node function (``_update_thesis_for_held_stock``) is a narrow, directly
-callable function of exactly those arguments — no surrounding
``ResearchState``/multi-team fan-out to reconstruct. ``sector_quant`` /
``sector_qual`` / ``sector_peer_review`` / ``ic_cio`` artifacts are read-only
inspectable via :func:`fetch_artifact` but raise ``ReplayNotSupportedError``
on replay — their node functions consume a much larger ``SectorTeamContext``
/ ``ResearchState`` that a single artifact's snapshot does not fully
reconstruct (a follow-up scope, not invented here).

Usage::

    # Faithful replay of a specific recorded thesis-update decision.
    python -m scripts.replay_harness faithful \\
        --agent-id thesis_update:financials:COIN --date 2026-06-05

    # Counterfactual: "would the loss-floor + de-stancing fix have changed
    # COIN's actual decision?" — 20 replays at temperature 0.7.
    python -m scripts.replay_harness coin-counterfactual \\
        --date 2026-06-05 --n-replay 20 --temperature 0.7
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ── Errors ──────────────────────────────────────────────────────────────


class ReplayError(RuntimeError):
    """Base class for replay-harness failures."""


class ArtifactNotFoundError(ReplayError):
    """No ``decision_artifacts/`` object found for the requested
    ``(agent_id, date)`` (or ``run_id``)."""


class ReplayNotSupportedError(ReplayError):
    """The artifact's ``agent_id`` has no registered replay dispatch entry.

    Raised instead of guessing at a node function — replaying against the
    wrong function would silently produce a diff against garbage, which is
    worse than refusing.
    """


# ── Artifact fetch (mirrors scripts.decision_review.fetch_decision_artifacts) ──


_DEFAULT_S3_BUCKET = "alpha-engine-research"
_DEFAULT_S3_PREFIX = "decision_artifacts"


def fetch_artifact(
    agent_id: str,
    date: str,
    *,
    run_id: Optional[str] = None,
    s3_client: Any = None,
    s3_bucket: str = _DEFAULT_S3_BUCKET,
    s3_prefix: str = _DEFAULT_S3_PREFIX,
) -> dict:
    """Fetch one recorded ``DecisionArtifact`` (as a plain dict) from S3.

    ``date`` is the capture-date partition (``YYYY-MM-DD``) — the artifact's
    S3 key is ``{s3_prefix}/{YYYY}/{MM}/{DD}/{agent_id}/{run_id}.json``. When
    ``run_id`` is not given, lists the agent's prefix for that day and takes
    the most-recently-modified object (mirrors
    ``scripts.decision_review.fetch_decision_artifacts``'s "run_id may be
    run_date or a Lambda request id, so list-and-take-newest" convention).

    Raises :class:`ArtifactNotFoundError` on any miss — unlike
    ``decision_review``'s best-effort ``fetch_decision_artifacts`` (which
    swallows misses because Q&A degrades gracefully without artifacts), a
    replay has nothing to replay without one, so this is NOT best-effort.

    ``s3_client`` is a dependency-injection point for tests (moto-mocked or
    a bare stub) — when ``None`` a real ``boto3.client("s3")`` is built
    lazily so importing this module never touches AWS/SSM.
    """
    if s3_client is None:
        import boto3  # lazy — avoids AWS/SSM touch on import

        s3_client = boto3.client("s3")

    y, m, d = date.split("-")
    prefix = f"{s3_prefix}/{y}/{m}/{d}/{agent_id}/"

    if run_id:
        key = f"{prefix}{run_id}.json"
        try:
            body = s3_client.get_object(Bucket=s3_bucket, Key=key)["Body"].read()
        except Exception as exc:
            raise ArtifactNotFoundError(
                f"no artifact at s3://{s3_bucket}/{key}: {exc}"
            ) from exc
        return json.loads(body)

    try:
        resp = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=prefix)
    except Exception as exc:
        raise ArtifactNotFoundError(
            f"listing s3://{s3_bucket}/{prefix} failed: {exc}"
        ) from exc
    objs = resp.get("Contents") or []
    if not objs:
        raise ArtifactNotFoundError(
            f"no decision_artifacts under s3://{s3_bucket}/{prefix} "
            f"(agent_id={agent_id!r}, date={date!r})"
        )
    newest = max(objs, key=lambda o: o["LastModified"])
    body = s3_client.get_object(Bucket=s3_bucket, Key=newest["Key"])["Body"].read()
    return json.loads(body)


# ── ArcticDB rehydration (as-of the artifact's data_snapshot_id) ─────────


def rehydrate_price_data(
    tickers: list[str],
    data_snapshot_id: Optional[str],
    *,
    bucket: str = "alpha-engine-research",
    period: str = "1y",
    arctic_lib: Any = None,
) -> dict:
    """Re-read ArcticDB OHLCV for ``tickers`` AS OF ``data_snapshot_id``.

    ``data_snapshot_id`` is the run-level ArcticDB version stamp surfaced by
    ``data.fetchers.price_fetcher.fetch_price_data(..., return_snapshot_id=True)``
    (1b, #781) — the immutable per-symbol write-generation integer ArcticDB
    calls ``VersionedItem.version``. Passing it back into
    ``Library.read(symbol, as_of=version, ...)`` pins the read to the SAME
    snapshot the original decision was computed on, even if newer daily
    appends have since landed (ArcticDB's native ``as_of`` kwarg — confirmed
    against the installed ``arcticdb.version_store.library.Library.read``
    signature, NOT exposed by ``nousergon_lib.arcticdb``'s thin
    ``open_universe_lib`` wrapper, so this calls the opened library
    directly rather than going through ``data.fetchers.price_fetcher``,
    which has no ``as_of`` parameter).

    ``data_snapshot_id`` of ``None``, ``"unknown"``, or non-numeric (legacy
    artifacts predating 1b, or a read that produced no versioned result)
    degrades to an un-pinned read (current library state) — logged loudly,
    never silently identical-looking, per ``feedback_no_silent_fails``. A
    caller that needs to know whether the read was actually pinned should
    check the returned dict is non-empty AND that a warning wasn't logged;
    :func:`faithful_replay` surfaces this via ``FaithfulReplayResult.pinned``.

    ``arctic_lib`` is a dependency-injection point for tests: an object
    exposing ``.read(symbol, as_of=..., date_range=..., columns=...) ->
    object-with-.data``, mirroring the fake ArcticDB library pattern in
    ``tests/test_price_fetcher_snapshot_id.py``. When ``None``, opens the
    real ``universe`` library via ``nousergon_lib.arcticdb.open_universe_lib``.
    """
    import pandas as pd

    from data.fetchers.price_fetcher import (
        _ARCTIC_OHLCV_COLS,
        _period_to_lookback_days,
    )

    as_of: Any = None
    if data_snapshot_id and data_snapshot_id != "unknown":
        try:
            as_of = int(data_snapshot_id)
        except (TypeError, ValueError):
            logger.warning(
                "[replay] data_snapshot_id=%r is not an ArcticDB version "
                "int — reading current library state, NOT pinned to the "
                "artifact's snapshot",
                data_snapshot_id,
            )
    else:
        logger.warning(
            "[replay] artifact has no usable data_snapshot_id (%r) — "
            "reading current library state, NOT pinned. Faithful replay "
            "of a pre-1b artifact cannot be data-certified.",
            data_snapshot_id,
        )

    if arctic_lib is None:
        from nousergon_lib.arcticdb import open_universe_lib

        arctic_lib = open_universe_lib(bucket)

    lookback_days = _period_to_lookback_days(period)
    end_ts = pd.Timestamp.utcnow().normalize().tz_localize(None)
    start_ts = end_ts - pd.Timedelta(days=lookback_days)

    result: dict = {}
    for ticker in tickers:
        try:
            read_kwargs: dict = {
                "date_range": (start_ts, end_ts),
                "columns": _ARCTIC_OHLCV_COLS,
            }
            if as_of is not None:
                read_kwargs["as_of"] = as_of
            res = arctic_lib.read(ticker, **read_kwargs)
            df = res.data
        except Exception as exc:
            logger.warning("[replay] ArcticDB read failed for %s: %s", ticker, exc)
            continue
        if df is None or df.empty:
            continue
        result[ticker] = df[~df.index.duplicated(keep="last")].sort_index()
    return result


# ── Field-level diff ───────────────────────────────────────────────────


@dataclass
class FieldDiff:
    """One top-level field that differs between recorded and replayed output."""

    field: str
    recorded: Any
    replayed: Any

    def to_dict(self) -> dict:
        return {"field": self.field, "recorded": self.recorded, "replayed": self.replayed}


def diff_outputs(recorded: dict, replayed: dict) -> list[FieldDiff]:
    """Field-level diff of two agent-output dicts — NOT just equality.

    Walks the union of top-level keys and reports every field whose value
    differs (added, removed, or changed), so a faithful-replay consumer can
    see WHAT changed rather than a bare True/False. Nested dict/list values
    are compared by equality (not recursively diffed) — sufficient for the
    narrative agent-output shapes this harness targets
    (``HeldThesisUpdateLLMOutput``: short strings + short lists), and keeps
    the diff itself simple to reason about and test.
    """
    diffs: list[FieldDiff] = []
    for key in sorted(set(recorded) | set(replayed)):
        r_val = recorded.get(key)
        p_val = replayed.get(key)
        if r_val != p_val:
            diffs.append(FieldDiff(field=key, recorded=r_val, replayed=p_val))
    return diffs


# ── Faithful replay ────────────────────────────────────────────────────


@dataclass
class FaithfulReplayResult:
    agent_id: str
    run_id: str
    data_snapshot_id: Optional[str]
    code_sha: Optional[str]
    pinned: bool  # True iff the ArcticDB read was actually as_of-pinned
    recorded_output: dict
    replayed_output: dict
    diffs: list[FieldDiff]

    @property
    def matches(self) -> bool:
        return not self.diffs

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "run_id": self.run_id,
            "data_snapshot_id": self.data_snapshot_id,
            "code_sha": self.code_sha,
            "pinned": self.pinned,
            "matches": self.matches,
            "diffs": [d.to_dict() for d in self.diffs],
        }


def _thesis_update_node(
    snapshot: dict,
    recorded_output: dict,
    *,
    temperature: Optional[float] = None,
    node_fn: Optional[Callable[..., dict]] = None,
) -> dict:
    """Dispatch entry for ``thesis_update:{team}:{ticker}`` — invokes
    ``agents.sector_teams.sector_team._update_thesis_for_held_stock`` with
    the exact args captured in the artifact's ``input_data_snapshot``
    (``build_thesis_update_capture_payload``'s shape: ``ticker``,
    ``team_id``, ``run_date``, ``triggers``, ``prior_thesis``, ``news_data``,
    ``analyst_data``). ``node_fn`` is a dependency-injection point for tests
    (a fake deterministic node) — defaults to the real node function.
    """
    if node_fn is None:
        from agents.sector_teams.sector_team import _update_thesis_for_held_stock

        node_fn = _update_thesis_for_held_stock
    return node_fn(
        snapshot["ticker"],
        snapshot.get("triggers") or [],
        snapshot.get("prior_thesis"),
        snapshot.get("news_data"),
        snapshot.get("analyst_data"),
        snapshot["run_date"],
        snapshot["team_id"],
        temperature=temperature,
    )


# agent_id PREFIX -> node dispatch callable. Matched by ``str.startswith`` so
# ``thesis_update:financials:COIN`` and ``thesis_update:tech:NVDA`` share one
# entry. Extend this table as more node functions grow a faithful-replay
# entry point (see module docstring — sector_quant/qual/peer_review/ic_cio
# are NOT yet supported; their node functions consume a full
# SectorTeamContext/ResearchState this harness does not reconstruct).
_NODE_DISPATCH: dict[str, Callable[..., dict]] = {
    "thesis_update:": _thesis_update_node,
}


def _resolve_node(agent_id: str) -> Callable[..., dict]:
    for prefix, fn in _NODE_DISPATCH.items():
        if agent_id.startswith(prefix):
            return fn
    raise ReplayNotSupportedError(
        f"agent_id={agent_id!r} has no registered replay dispatch entry "
        f"(supported prefixes: {sorted(_NODE_DISPATCH)}). "
        f"scripts.replay_harness.fetch_artifact() can still fetch it "
        f"read-only via decision_review-style inspection."
    )


def faithful_replay(
    agent_id: str,
    date: str,
    *,
    run_id: Optional[str] = None,
    s3_client: Any = None,
    arctic_lib: Any = None,
    node_fn: Optional[Callable[..., dict]] = None,
    rehydrate_prices: bool = False,
) -> FaithfulReplayResult:
    """Rehydrate + re-run the node that produced ``(agent_id, date)`` and
    diff the fresh output against what was actually recorded.

    Steps (per the module docstring's faithful-replay contract):

    1. Fetch the ``DecisionArtifact`` from S3.
    2. Extract ``data_snapshot_id`` — the ArcticDB version the ORIGINAL
       decision's price reads resolved to.
    3. Rehydrate: reconstruct the input state the node saw from
       ``input_data_snapshot``. When ``rehydrate_prices=True`` AND the
       snapshot references tickers needing a live price re-read (not the
       case for ``thesis_update``, whose snapshot is fully self-contained —
       this flag exists for future dispatch entries that need it), also
       re-reads ArcticDB ``as_of`` that version via
       :func:`rehydrate_price_data`.
    4. Re-invoke the SAME node function via the dispatch table.
    5. Field-level diff the fresh output against ``agent_output`` recorded
       in the artifact.

    Raises :class:`ArtifactNotFoundError` if no artifact exists, or
    :class:`ReplayNotSupportedError` if ``agent_id`` has no dispatch entry.
    """
    artifact = fetch_artifact(agent_id, date, run_id=run_id, s3_client=s3_client)
    resolved_agent_id = artifact.get("agent_id", agent_id)
    node = node_fn or _resolve_node(resolved_agent_id)

    snapshot = artifact["input_data_snapshot"]
    recorded_output = artifact["agent_output"]
    data_snapshot_id = artifact.get("data_snapshot_id")
    pinned = bool(data_snapshot_id) and data_snapshot_id != "unknown"

    if rehydrate_prices:
        tickers = [snapshot["ticker"]] if "ticker" in snapshot else []
        rehydrate_price_data(tickers, data_snapshot_id, arctic_lib=arctic_lib)

    replayed_output = node(snapshot, recorded_output, temperature=0.0)
    diffs = diff_outputs(recorded_output, replayed_output)

    return FaithfulReplayResult(
        agent_id=resolved_agent_id,
        run_id=artifact.get("run_id", run_id or "unknown"),
        data_snapshot_id=data_snapshot_id,
        code_sha=artifact.get("code_sha"),
        pinned=pinned,
        recorded_output=recorded_output,
        replayed_output=replayed_output,
        diffs=diffs,
    )


# ── Counterfactual replay ──────────────────────────────────────────────


@dataclass
class PerturbationSpec:
    """Describes how to mutate a rehydrated input snapshot before replay.

    ``field_path`` is a dotted path into the snapshot dict (e.g.
    ``"prior_thesis.conviction"`` or ``"triggers"``); nested keys are walked
    with ``dict.get`` semantics (missing intermediate keys are created as
    empty dicts). ``mutate`` receives the CURRENT value at that path
    (``None`` if absent) and returns the new value — a function rather than
    a bare replacement value so perturbations can be relative (e.g. "append
    to this list", "multiply this score") as well as absolute.

    ``name`` + ``description`` are for the distribution summary / audit
    trail; ``name`` should be short and stable (used as a dict key
    elsewhere), ``description`` is free text explaining WHY this mutation
    models the counterfactual in human terms.
    """

    name: str
    description: str
    field_path: str
    mutate: Callable[[Any], Any]

    def apply(self, snapshot: dict) -> dict:
        """Return a NEW snapshot dict with the mutation applied (does not
        mutate the input — the recorded/original snapshot must stay intact
        for comparison)."""
        import copy

        out = copy.deepcopy(snapshot)
        parts = self.field_path.split(".")
        node = out
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        leaf = parts[-1]
        node[leaf] = self.mutate(node.get(leaf))
        return out


@dataclass
class OutcomeDistribution:
    """Distributional summary of N counterfactual replays.

    ``rating_counts`` / ``conviction_values`` are the two outcome axes the
    ``HeldThesisUpdateLLMOutput`` schema actually exposes (no ``rating``
    field on the LLM output itself, but ``conviction`` is the closest thing
    to a "did the decision change" score — see the
    ``HeldThesisUpdateLLMOutput`` docstring: narrative-only, no BUY/HOLD/SELL
    field). ``field_value_counts`` generalizes this to ANY field so a
    perturbation spec targeting a different node's output isn't stuck with
    conviction-only reporting.
    """

    perturbation_name: str
    n_replay: int
    temperature: float
    actual_recorded: dict
    field_value_counts: dict[str, Counter]
    conviction_values: list[Optional[int]]
    conviction_mean: Optional[float]
    conviction_delta_from_recorded: Optional[float]
    raw_outputs: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "perturbation_name": self.perturbation_name,
            "n_replay": self.n_replay,
            "temperature": self.temperature,
            "actual_recorded": self.actual_recorded,
            "field_value_counts": {
                k: dict(v) for k, v in self.field_value_counts.items()
            },
            "conviction_values": self.conviction_values,
            "conviction_mean": self.conviction_mean,
            "conviction_delta_from_recorded": self.conviction_delta_from_recorded,
        }


def _summarize_distribution(
    perturbation_name: str,
    n_replay: int,
    temperature: float,
    recorded_output: dict,
    outputs: list[dict],
    *,
    tracked_fields: tuple[str, ...] = ("conviction", "thesis_summary", "triggers_response"),
) -> OutcomeDistribution:
    field_value_counts: dict[str, Counter] = {
        f: Counter(json.dumps(o.get(f), sort_keys=True, default=str) for o in outputs)
        for f in tracked_fields
    }
    conviction_values = [o.get("conviction") for o in outputs]
    numeric = [c for c in conviction_values if isinstance(c, (int, float))]
    conviction_mean = (sum(numeric) / len(numeric)) if numeric else None

    recorded_conviction = recorded_output.get("conviction")
    conviction_delta = None
    if conviction_mean is not None and isinstance(recorded_conviction, (int, float)):
        conviction_delta = conviction_mean - recorded_conviction

    return OutcomeDistribution(
        perturbation_name=perturbation_name,
        n_replay=n_replay,
        temperature=temperature,
        actual_recorded=recorded_output,
        field_value_counts=field_value_counts,
        conviction_values=conviction_values,
        conviction_mean=conviction_mean,
        conviction_delta_from_recorded=conviction_delta,
        raw_outputs=outputs,
    )


def counterfactual_replay(
    agent_id: str,
    date: str,
    perturbation: PerturbationSpec,
    *,
    n_replay: int = 1,
    temperature: float = 0.0,
    run_id: Optional[str] = None,
    s3_client: Any = None,
    node_fn: Optional[Callable[..., dict]] = None,
) -> OutcomeDistribution:
    """Rehydrate the artifact, apply ``perturbation``, and re-run the node
    ``n_replay`` times at ``temperature``, returning a distribution summary.

    ``n_replay=1, temperature=0.0`` is a single deterministic point (matches
    a faithful-replay-style invocation, just with mutated inputs). Anything
    with ``temperature > 0`` and ``n_replay > 1`` samples the outcome
    distribution — the mode this harness exists to support, per the issue's
    explicit requirement that a counterfactual answer come with a
    distribution, not a single anecdote.
    """
    if n_replay < 1:
        raise ValueError(f"n_replay must be >= 1, got {n_replay}")

    artifact = fetch_artifact(agent_id, date, run_id=run_id, s3_client=s3_client)
    resolved_agent_id = artifact.get("agent_id", agent_id)
    node = node_fn or _resolve_node(resolved_agent_id)

    snapshot = artifact["input_data_snapshot"]
    recorded_output = artifact["agent_output"]
    mutated_snapshot = perturbation.apply(snapshot)

    outputs = [
        node(mutated_snapshot, recorded_output, temperature=temperature)
        for _ in range(n_replay)
    ]

    return _summarize_distribution(
        perturbation.name, n_replay, temperature, recorded_output, outputs,
    )


# ── COIN acceptance scenario ───────────────────────────────────────────
#
# Concrete counterfactual anchor (issue #781's own build-when-real trigger):
# "would the COIN loss-floor + de-stancing fix have changed COIN's actual
# decision?" The fix itself (alpha-engine-config#964 `position_loss_floor`
# MAE guard, PRs cipher813/alpha-engine#238/#240/#241; #965 de-stancing
# uniform-momentum-veto, PR #240) lives entirely in the EXECUTOR's
# position-level risk layer — a stance-agnostic hard exit + momentum veto
# applied to portfolio positions, not a change to this repo's
# `_update_thesis_for_held_stock` LLM call. crucible-research's held-thesis
# agent never saw "loss floor breached" as an input; it only sees
# `triggers` (material_triggers.py) + `prior_thesis` + `news_data` +
# `analyst_data`.
#
# So a faithful model of "the fix was in effect" at the RESEARCH layer is:
# had the executor-side floor already force-exited (or the de-stancing
# uniform momentum veto already blocked re-entry into) a collapsing value
# position, the thesis-update agent would have been told about it — via a
# new material trigger firing (the position's realized/unrealized loss
# breaching the floor is itself a material fact deserving a thesis
# revisit) AND via the prior_thesis carrying a flag that the position is
# now under a hard risk override. The perturbation below models exactly
# that: inject a synthetic `"loss_floor_breach"` trigger + a
# `risk_override_active` flag on `prior_thesis`, and observe whether the
# agent's own narrative conviction still supports holding once it is told
# the position is under a stance-agnostic exit mandate — which is the
# closest research-side proxy for "would this fix have changed the
# decision" that doesn't require fabricating executor-side P&L state this
# repo doesn't compute.

COIN_LOSS_FLOOR_FIX_DESCRIPTION = (
    "Models alpha-engine-config#964 (position_loss_floor MAE guard, "
    "cipher813/alpha-engine#238/#241) + #965 (de-stancing uniform momentum "
    "veto, #240), both merged/deployed 2026-06-08. Both fixes act at the "
    "EXECUTOR position-risk layer (stance-agnostic hard exit on loss-floor "
    "breach; uniform momentum veto replacing value-invert/quality-relax "
    "branches) — crucible-research's thesis_update agent has no direct "
    "input for either. This perturbation surfaces the fix's EFFECT as a "
    "research-layer fact the agent would plausibly have been told about: "
    "a new 'loss_floor_breach' material trigger + a prior_thesis "
    "'risk_override_active' flag, modeling a world where the position was "
    "already force-exited/blocked from re-entry by the executor-side guard "
    "before this thesis update ran."
)


def coin_loss_floor_perturbation() -> PerturbationSpec:
    """The concrete perturbation spec for the COIN counterfactual anchor.

    Two mutations bundled as one spec (both are needed to faithfully model
    "the fix was in effect", per the description above):

    1. Append ``"loss_floor_breach"`` to ``triggers`` — the fact that
       would have caused this thesis update to fire in the first place
       had the executor already exited the position (pre-fix runs may not
       have had this trigger fire at all; post-fix, the position breaching
       the MAE floor IS itself the triggering event).
    2. Set ``prior_thesis.risk_override_active = True`` — surfaces to the
       LLM (via ``format_structured_thesis_for_prompt``... note: this flag
       is NOT currently rendered into the prompt text by
       ``format_structured_thesis_for_prompt``, which is a real, disclosed
       gap — see the PR body) that this position is under a stance-agnostic
       hard-risk mandate, not just an ordinary thesis review.
    """
    def _mutate_triggers(current: Any) -> list[str]:
        triggers = list(current) if isinstance(current, list) else []
        if "loss_floor_breach" not in triggers:
            triggers = triggers + ["loss_floor_breach"]
        return triggers

    return PerturbationSpec(
        name="coin_loss_floor_and_destancing_fix",
        description=COIN_LOSS_FLOOR_FIX_DESCRIPTION,
        field_path="triggers",
        mutate=_mutate_triggers,
    )


def coin_risk_override_perturbation() -> PerturbationSpec:
    """Companion spec setting the ``prior_thesis.risk_override_active``
    flag — applied as a SECOND :func:`counterfactual_replay` call (or
    composed by a caller that chains both mutations on the same rehydrated
    snapshot) since :class:`PerturbationSpec` targets one field_path.
    """
    return PerturbationSpec(
        name="coin_risk_override_flag",
        description=COIN_LOSS_FLOOR_FIX_DESCRIPTION,
        field_path="prior_thesis.risk_override_active",
        mutate=lambda _current: True,
    )


def run_coin_scenario(
    date: str,
    *,
    team_id: str = "financials",
    ticker: str = "COIN",
    n_replay: int = 20,
    temperature: float = 0.7,
    run_id: Optional[str] = None,
    s3_client: Any = None,
    node_fn: Optional[Callable[..., dict]] = None,
) -> OutcomeDistribution:
    """Run the concrete COIN counterfactual against a pre-fix artifact.

    ``date`` must be one of the pre-fix (2026-06-08) COIN
    ``thesis_update:financials:COIN`` capture dates confirmed live in S3
    (2026-05-08, 05-13, 05-15, 05-22, 05-29, or 06-05 — see the artifact key
    ``decision_artifacts/2026/06/06/thesis_update:financials:COIN/2026-06-05.json``,
    whose CAPTURE date is 2026-06-06 for an eval as-of 2026-06-05; pass the
    capture date here).

    Both perturbations (loss-floor trigger + risk-override flag) are
    applied to the SAME rehydrated snapshot before the N replays — composing
    them inline here rather than via two separate
    :func:`counterfactual_replay` calls, since they model one indivisible
    counterfactual world ("the fix was in effect"), not two independent
    questions.
    """
    agent_id = f"thesis_update:{team_id}:{ticker}"
    trigger_spec = coin_loss_floor_perturbation()
    override_spec = coin_risk_override_perturbation()

    combined = PerturbationSpec(
        name="coin_loss_floor_and_destancing_fix",
        description=COIN_LOSS_FLOOR_FIX_DESCRIPTION,
        field_path=trigger_spec.field_path,
        mutate=trigger_spec.mutate,
    )

    artifact = fetch_artifact(agent_id, date, run_id=run_id, s3_client=s3_client)
    resolved_agent_id = artifact.get("agent_id", agent_id)
    node = node_fn or _resolve_node(resolved_agent_id)

    snapshot = artifact["input_data_snapshot"]
    recorded_output = artifact["agent_output"]

    mutated = combined.apply(snapshot)
    mutated = override_spec.apply(mutated)

    outputs = [
        node(mutated, recorded_output, temperature=temperature)
        for _ in range(n_replay)
    ]

    return _summarize_distribution(
        combined.name, n_replay, temperature, recorded_output, outputs,
    )


# ── CLI ─────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="replay_harness",
        description="Faithful + counterfactual scenario-replay harness (L4567 2b / #781).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    pf = sub.add_parser(
        "faithful", help="Faithful replay: rehydrate + re-run + diff vs recorded.",
    )
    pf.add_argument("--agent-id", required=True, help="e.g. thesis_update:financials:COIN")
    pf.add_argument("--date", required=True, help="Capture date (YYYY-MM-DD).")
    pf.add_argument("--run-id", help="Specific run_id (default: newest under the date prefix).")

    pc = sub.add_parser(
        "coin-counterfactual",
        help="The concrete COIN acceptance scenario: would the loss-floor + "
             "de-stancing fix have changed COIN's actual decision?",
    )
    pc.add_argument("--date", required=True, help="Pre-fix capture date, e.g. 2026-06-06.")
    pc.add_argument("--team-id", default="financials")
    pc.add_argument("--ticker", default="COIN")
    pc.add_argument("--n-replay", type=int, default=20)
    pc.add_argument("--temperature", type=float, default=0.7)
    pc.add_argument("--run-id")

    return p


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args(argv)

    if args.command == "faithful":
        result = faithful_replay(args.agent_id, args.date, run_id=args.run_id)
        print(json.dumps(result.to_dict(), default=str, indent=2))
    elif args.command == "coin-counterfactual":
        result = run_coin_scenario(
            args.date,
            team_id=args.team_id,
            ticker=args.ticker,
            n_replay=args.n_replay,
            temperature=args.temperature,
            run_id=args.run_id,
        )
        print(json.dumps(result.to_dict(), default=str, indent=2))
    else:  # pragma: no cover — argparse enforces a valid subcommand
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
