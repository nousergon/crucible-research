#!/usr/bin/env python3
"""Lambda code deploy-drift detector for every ``alpha-engine-research-*``
function (alpha-engine-config-I7840).

WHY THIS EXISTS
---------------
Measured 2026-08-20. ``crucible-research@2fc065fc`` ("count each arm once",
alpha-engine-config-I7645) merged 2026-08-18 13:26 PT. Its deploy run
``32182305032`` failed at the Docker build step (``gitleaks`` download, exit
127) and every subsequent attempt through 2026-08-19T20:06:26Z failed or never
ran. The last successful deploy predated the merge. For two days the
``alpha-engine-research-scanner`` Lambda ran pre-fix code while ``main``
carried the fix; ``research/cuts_leaderboard/2026-08-19.json`` reproduced the
exact defect the merged commit had fixed, and a full session was spent proving
that a merged fix was not live. Third instance of the class in eight days
(alpha-engine-config-I7411; the 2026-08-19 Docker-build failure).

``deploy.yml``'s ``notify-main-failure`` job already pages on a FAILED deploy.
That signal exists and fired. It says "a deploy failed"; it does not say "the
running Lambda is behind main", and it does not persist — once a later
unrelated deploy succeeds, nothing stays red even though the intervening
commits may never have shipped. The gap is the STANDING drift STATE, not the
transient event, which is why this runs on a schedule of its own rather than
as a step of the workflow whose failure it is watching. A detector that only
runs when a deploy runs cannot see a deploy that never ran.

WHAT IT COMPARES, AND WHY NOT A BARE SHA EQUALITY
-------------------------------------------------
This mirrors ``crucible-predictor/inference/deploy_drift.py`` (the Step
Functions counterpart, alpha-engine-config-I7799) including the lesson that
cost the fleet a trading session on 2026-08-20: comparing a deployed STAMP
against repo HEAD answers "has anything merged since the deploy", which is
strictly broader than "is the running code different from what main
describes". Three merges that touched nothing in the preopen definition halted
preopen trading anyway — unmanaged session, open positions carried.

The Lambda analogue of that false positive is built in by construction:
``deploy.yml`` is PATH-FILTERED. A merge touching only ``tests/``, ``*.md``,
or ``.github/workflows/ci.yml`` never triggers a publish, so bare
stamped-SHA != HEAD would be red on most merges and permanently ignored.

So the comparison is CONTENT-SHAPED: the verdict is drift only if a commit in
``stamped_sha..main`` touched a path inside ``deploy.yml``'s own ``paths:``
filter — read out of ``deploy.yml`` itself, never re-listed here, because a
second copy of that list is a fork that would go stale in exactly the
direction that produces false pages. A merge that changed no deployed code
path reports ``in_sync_no_deployable_change`` and is not drift.

WHAT COUNTS AS UNMEASURED (never as a pass)
-------------------------------------------
A verdict the detector could not actually make is reported under its own
reason code and is never rendered as ``in_sync``:

  ``stamp_absent``          the function predates the stamp, or was published
                            by something other than ``deploy.sh``.
  ``stamp_unknown_commit``  the stamped SHA is not a commit this checkout can
                            resolve (force-push, shallow clone, foreign repo).
  ``config_read_failed``    the AWS read raised.

Each is a WARNING regardless of the function's criticality: it is a hole in
the measurement, not evidence of drift, and grading it ERROR would page for
the detector's own blind spot. It is still SAID, on the console and in the
alert, because a function nothing can measure is unobserved, not healthy.

THE GRACE WINDOW
----------------
A deploy takes ~12-18 minutes and the detector runs on a clock that knows
nothing about it, so a run landing inside a deploy's own window would see real,
correct, in-flight drift and page for it. A relevant commit younger than
``--grace-minutes`` (default 45) therefore reports ``pending_deploy``, not
drift. Only when the OLDEST unshipped relevant commit is past the window does
the verdict harden — the oldest, not the newest, because one fresh merge on
top of a two-day-old undeployed one must not reset the clock.

SEVERITY
--------
Drift on a function on the trading critical path is an ERROR (red check +
alert, and it stays red until deployed). Drift on an observe-only function is
a WARNING (alert + annotation + job summary, green check). Both are visible;
neither is silent. Criticality is declared per function in ``REGISTRY`` below
with a written reason — an undeclared function is a hard failure of the
registry contract test, not a silent default.

MEASURABILITY (principles.md §2.7)
----------------------------------
The number is ``measured/total`` and ``drifted`` on the final ``SUMMARY:``
line of every run, repeated in the job summary and in the alert body. Its
ABSENCE — this workflow not running at all — is covered by
``alpha-engine-config``'s ``scheduled-workflow-health`` reconciler, which
grades ``nousergon/crucible-research`` (matrix row added with this change).
A detector whose own silence nobody grades is the failure mode this file
exists to remove, so it is not permitted to be one.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
It does not enumerate live functions with ``lambda:ListFunctions`` — the
``github-actions-lambda-deploy`` role does not hold it (it is an account-level
action needing ``Resource: *``), and adding it is an IAM change no merge
applies. The realistic drift direction — a function added to ``deploy.sh``
and never registered here — is closed offline instead, by
``tests/test_lambda_deploy_drift_registry.py``, which reconciles ``REGISTRY``
against ``deploy.sh``'s own ``FUNCTION_*`` variables.

Usage:
    python3 infrastructure/lambda_deploy_drift.py                 # report; exit 1 on ERROR
    python3 infrastructure/lambda_deploy_drift.py --json          # machine-readable
    python3 infrastructure/lambda_deploy_drift.py --emit-alert    # + krepis.alerts publish
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEPLOY_YML = _REPO_ROOT / ".github" / "workflows" / "deploy.yml"
_DEPLOY_SH = _REPO_ROOT / "infrastructure" / "deploy.sh"

_REGION = os.environ.get("AWS_REGION", "us-east-1")

#: Environment variable ``infrastructure/deploy.sh`` stamps onto every
#: function it publishes, BEFORE ``publish-version`` so the ``live`` alias
#: serves a version that carries it.
#:
#: Deliberately the SAME name the Dockerfile already bakes as an image ENV
#: (``ARG GIT_SHA`` -> ``ENV ALPHA_ENGINE_CODE_SHA``, Dockerfile L69-70) and
#: that ``producers/experiment_record.py`` and the retired research graph
#: already read at runtime. A second stamp key would be a fork of the one fact
#: this file exists to read, and the two copies would diverge in exactly the
#: direction that yields a confident wrong answer. What I7840 added is not a
#: new fact — it is the same fact set at the FUNCTION level, where
#: ``get-function-configuration`` can read it without invoking the Lambda. An
#: image ENV is not surfaced in the Lambda configuration at all, which is why
#: the 2026-08-18 scanner could be two days stale with the SHA present in the
#: image and unreadable from outside.
GIT_SHA_ENV = "ALPHA_ENGINE_CODE_SHA"
#: Companion stamp: ``true`` when the deploy was built from a working tree
#: with uncommitted changes, i.e. the SHA does not describe what is running.
DIRTY_ENV = "DEPLOY_STAMP_DIRTY"

CRITICAL = "critical"
OBSERVE = "observe"
RETIRED = "retired"

ERROR = "error"
WARNING = "warning"
OK = "ok"


@dataclass(frozen=True)
class FunctionSpec:
    """One declared ``alpha-engine-research-*`` function."""

    criticality: str
    reason: str
    #: Alias the invokers actually call. Read the stamp AT THIS QUALIFIER —
    #: ``$LATEST`` can carry a configuration the alias does not serve, which
    #: is precisely the class ``_verify_live_alias`` in deploy.sh exists for
    #: (config#2766). ``None`` means the function has no alias and is invoked
    #: unqualified.
    alias: str | None = "live"


#: Every function ``infrastructure/deploy.sh`` can publish. Reconciled
#: against deploy.sh's ``FUNCTION_*`` variables by
#: ``tests/test_lambda_deploy_drift_registry.py`` — this list may not silently
#: fall behind the deploy script.
REGISTRY: dict[str, FunctionSpec] = {
    "alpha-engine-research-runner": FunctionSpec(
        CRITICAL,
        "Produces signals.json + DecisionArtifacts — the R slot of the M0 "
        "contract. Everything the predictor and executor believe starts here.",
    ),
    "alpha-engine-research-scanner": FunctionSpec(
        CRITICAL,
        "Produces the weekly universe cut the sector teams and the live "
        "producers read. The 2026-08-18 two-day stale-code incident that "
        "produced I7840 was this function.",
    ),
    "alpha-engine-research-signals-envelope": FunctionSpec(
        CRITICAL,
        "Weekly SF SignalsEnvelope state — the envelope downstream sizing "
        "reads. Wrong envelope is wrong risk.",
    ),
    "alpha-engine-research-alerts": FunctionSpec(
        CRITICAL,
        "The research alerting path itself. Stale code here is a silent "
        "alerter, which is worse than a stale observer: it removes the "
        "signal that every other failure is reported through.",
        alias=None,
    ),
    "alpha-engine-research-eval-judge": FunctionSpec(
        OBSERVE,
        "LLM-as-judge grading. Feeds evaluation, not execution — a stale "
        "judge mis-grades a run; it does not mis-trade one.",
    ),
    "alpha-engine-research-eval-judge-submit": FunctionSpec(
        OBSERVE,
        "Eval-judge plan/submit leg — the only Lambda left of what was the "
        "batch chain. Poll and Process retired by "
        "alpha-engine-config-I9329; Process runs on a spot box now, so it "
        "has no Lambda to drift.",
    ),
    "alpha-engine-research-eval-rolling-mean": FunctionSpec(
        OBSERVE,
        "Rolling-mean quality floor. Observe-only — and the function whose "
        "5-week silent rot (2026-06-11) is why deploy.yml has a step per "
        "target at all.",
    ),
    "alpha-engine-research-rationale-clustering": FunctionSpec(
        OBSERVE, "Cross-week rationale clustering. Reporting surface only.",
    ),
    "alpha-engine-research-aggregate-costs": FunctionSpec(
        OBSERVE, "Daily cost aggregation. Spend attribution, not execution.",
    ),
    "alpha-engine-research-openrouter-shadow": FunctionSpec(
        OBSERVE, "Shadow judge arm. Challenger-side measurement only.",
    ),
    "alpha-engine-research-perturbation-battery": FunctionSpec(
        OBSERVE, "Weekly judge-sensitivity scorecard. Measurement only.",
    ),
    "alpha-engine-research-thinktank": FunctionSpec(
        RETIRED,
        "Retired 2026-08-04 (alpha-engine-config-I5777): both invokers were "
        "repointed onto alpha-engine-thinktank-spot-dispatcher and deploy.sh "
        "no longer defines a thinktank target. Verified 2026-08-20: zero "
        "invocations since 2026-08-05. Declared rather than deleted so a "
        "reader is told it is deliberately unwatched, not overlooked.",
        alias=None,
    ),
}


# ── deploy.yml path filter ───────────────────────────────────────────────────

def load_deploy_paths(deploy_yml: Path = _DEPLOY_YML) -> list[str]:
    """The ``on.push.paths`` list from ``deploy.yml``, verbatim.

    Read from the workflow rather than re-listed here on purpose: a second
    copy would drift, and it would drift toward paging for merges that never
    deploy (the I7799 false-halt shape) or toward missing merges that do.

    Raises rather than defaulting — an empty or unreadable filter would make
    every verdict ``no_deployable_change``, i.e. a detector that reports
    healthy because it cannot see.
    """
    import yaml

    doc = yaml.safe_load(deploy_yml.read_text())
    # PyYAML resolves the bare key ``on:`` to the boolean True (YAML 1.1).
    triggers = doc.get("on", doc.get(True))
    if not isinstance(triggers, dict):
        raise ValueError(f"{deploy_yml}: no parseable `on:` block")
    paths = (triggers.get("push") or {}).get("paths")
    if not paths:
        raise ValueError(
            f"{deploy_yml}: on.push.paths is empty or absent. Refusing to "
            "run: with no path filter every verdict would be 'no deployable "
            "change', which reads as healthy and measures nothing."
        )
    return list(paths)


def path_matches(path: str, pattern: str) -> bool:
    """GitHub Actions path-filter glob semantics.

    ``**`` crosses ``/``; ``*`` and ``?`` do not. ``fnmatch`` is not usable
    here — its ``*`` matches ``/``, so ``config.py`` would be matched by
    ``config/**`` and ``agents/x/y.py`` by ``agents/*``.
    """
    out: list[str] = []
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif ch == "*":
            out.append("[^/]*")
            i += 1
        elif ch == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(ch))
            i += 1
    return re.fullmatch("".join(out), path) is not None


def is_deployable(path: str, patterns: list[str]) -> bool:
    return any(path_matches(path, p) for p in patterns)


# ── git ──────────────────────────────────────────────────────────────────────

#: Resolved once, absolutely. ruff S607 is enforced in this repo (pyproject
#: L24-25) and a partial executable path is resolved against whatever PATH the
#: runner happens to have.
_GIT = shutil.which("git")


def _git(*args: str, cwd: Path = _REPO_ROOT) -> str:
    """Run git, raising on failure. No silent swallow: a git call that fails
    here means the comparison cannot be made, and the caller must say so."""
    if _GIT is None:
        raise RuntimeError(
            "git is not on PATH. The detector cannot resolve the deployed "
            "stamp against main and must not report anything as in sync."
        )
    return subprocess.run(
        [_GIT, *args], cwd=cwd, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _commit_exists(sha: str) -> bool:
    try:
        _git("cat-file", "-e", f"{sha}^{{commit}}")
        return True
    except subprocess.CalledProcessError:
        return False


@dataclass
class RelevantChange:
    commits: list[str] = field(default_factory=list)
    files: set[str] = field(default_factory=set)
    oldest_commit_ts: int | None = None


def relevant_changes(
    stamped_sha: str, head_sha: str, patterns: list[str],
) -> RelevantChange:
    """Commits in ``stamped..head`` that touched a deploy-triggering path.

    Returns the commits, the union of matching files, and the OLDEST such
    commit's author timestamp — oldest, because the grace window asks how
    long the earliest unshipped change has been waiting, and a newer merge
    stacked on top of it must not reset that clock.
    """
    raw = _git(
        "log", "--format=__C__%H %ct", "--name-only", "--no-merges",
        f"{stamped_sha}..{head_sha}",
    )
    result = RelevantChange()
    cur_sha: str | None = None
    cur_ts: int | None = None
    cur_files: list[str] = []

    def flush() -> None:
        nonlocal cur_sha, cur_ts, cur_files
        if cur_sha is None:
            return
        hits = [f for f in cur_files if is_deployable(f, patterns)]
        if hits:
            result.commits.append(cur_sha)
            result.files.update(hits)
            if result.oldest_commit_ts is None or (
                cur_ts is not None and cur_ts < result.oldest_commit_ts
            ):
                result.oldest_commit_ts = cur_ts
        cur_sha, cur_ts, cur_files = None, None, []

    for line in raw.splitlines():
        if line.startswith("__C__"):
            flush()
            sha, _, ts = line[len("__C__"):].partition(" ")
            cur_sha = sha
            cur_ts = int(ts) if ts.isdigit() else None
        elif line.strip():
            cur_files.append(line.strip())
    flush()
    return result


# ── AWS ──────────────────────────────────────────────────────────────────────

def read_stamp(function_name: str, alias: str | None) -> dict:
    """``{git_sha, dirty}`` from the function's environment at ``alias``.

    Reads AT THE INVOKED QUALIFIER. A stamp on ``$LATEST`` that the ``live``
    alias does not serve describes code nothing runs.
    """
    import boto3

    client = boto3.client("lambda", region_name=_REGION)
    kwargs = {"FunctionName": function_name}
    if alias:
        kwargs["Qualifier"] = alias
    cfg = client.get_function_configuration(**kwargs)
    env = (cfg.get("Environment") or {}).get("Variables") or {}
    return {
        "git_sha": env.get(GIT_SHA_ENV),
        "dirty": str(env.get(DIRTY_ENV, "")).lower() == "true",
        "last_modified": cfg.get("LastModified"),
    }


# ── verdicts ─────────────────────────────────────────────────────────────────

def _severity_for(spec: FunctionSpec) -> str:
    return ERROR if spec.criticality == CRITICAL else WARNING


def check_function(
    name: str,
    spec: FunctionSpec,
    head_sha: str,
    patterns: list[str],
    grace_minutes: int,
    now_ts: int | None = None,
    reader=read_stamp,
) -> dict:
    """One function's verdict. Never returns ``in_sync`` for something it
    could not measure — see the module docstring's unmeasured taxonomy."""
    now_ts = int(time.time()) if now_ts is None else now_ts
    row: dict = {
        "function": name,
        "criticality": spec.criticality,
        "alias": spec.alias,
        "head_sha": head_sha,
    }

    if spec.criticality == RETIRED:
        row.update(verdict="retired", severity=OK, measured=False,
                   detail=spec.reason)
        return row

    try:
        stamp = reader(name, spec.alias)
    except Exception as exc:  # noqa: BLE001 — reported, never swallowed
        row.update(
            verdict="config_read_failed", severity=WARNING, measured=False,
            detail=f"{exc.__class__.__name__}: {str(exc)[:200]}",
        )
        return row

    row["stamped_sha"] = stamp["git_sha"]
    row["last_modified"] = stamp.get("last_modified")

    if not stamp["git_sha"]:
        row.update(
            verdict="stamp_absent", severity=WARNING, measured=False,
            detail=(
                f"No {GIT_SHA_ENV} on {name}"
                + (f" at alias '{spec.alias}'" if spec.alias else "")
                + ". Expected until the first deploy carrying the I7840 "
                  "stamp lands; a WARNING and not an ERROR because an "
                  "absent stamp is an unmeasured state, not proven drift."
            ),
        )
        return row

    if stamp["dirty"]:
        row.update(
            verdict="deployed_from_dirty_tree",
            severity=_severity_for(spec), measured=True,
            detail=(
                f"{name} was published from a working tree with uncommitted "
                f"changes, so {stamp['git_sha'][:8]} does not describe the "
                "running code. Redeploy from a clean checkout."
            ),
        )
        return row

    if not _commit_exists(stamp["git_sha"]):
        row.update(
            verdict="stamp_unknown_commit", severity=WARNING, measured=False,
            detail=(
                f"Stamped SHA {stamp['git_sha'][:12]} is not resolvable in "
                "this checkout (force-push, shallow clone, or a build from "
                "another repo). Cannot compare — reported, not passed."
            ),
        )
        return row

    if stamp["git_sha"] == head_sha:
        row.update(verdict="in_sync", severity=OK, measured=True,
                   detail="Stamped SHA is main HEAD.")
        return row

    changes = relevant_changes(stamp["git_sha"], head_sha, patterns)
    row["unshipped_commits"] = changes.commits
    row["unshipped_files"] = sorted(changes.files)

    if not changes.commits:
        row.update(
            verdict="in_sync_no_deployable_change", severity=OK, measured=True,
            detail=(
                f"{stamp['git_sha'][:8]}..{head_sha[:8]} touches no path in "
                "deploy.yml's filter, so no publish was due. This is the "
                "I7799 lesson made explicit: a merge that changed no deployed "
                "code path is not drift."
            ),
        )
        return row

    age_min = (
        None if changes.oldest_commit_ts is None
        else (now_ts - changes.oldest_commit_ts) // 60
    )
    row["oldest_unshipped_age_minutes"] = age_min

    if age_min is not None and age_min < grace_minutes:
        row.update(
            verdict="pending_deploy", severity=OK, measured=True,
            detail=(
                f"{len(changes.commits)} deployable commit(s), oldest "
                f"{age_min}m old, inside the {grace_minutes}m grace window — "
                "a deploy is plausibly still in flight."
            ),
        )
        return row

    row.update(
        verdict="drifted", severity=_severity_for(spec), measured=True,
        detail=(
            f"{name} is running {stamp['git_sha'][:8]}; main is "
            f"{head_sha[:8]}. {len(changes.commits)} deployable commit(s) "
            f"unshipped, oldest {age_min}m old, touching: "
            + ", ".join(sorted(changes.files)[:8])
            + ("…" if len(changes.files) > 8 else "")
        ),
    )
    return row


def run(
    head_sha: str | None = None,
    grace_minutes: int = 45,
    registry: dict[str, FunctionSpec] | None = None,
    reader=read_stamp,
) -> dict:
    registry = REGISTRY if registry is None else registry
    patterns = load_deploy_paths()
    head_sha = head_sha or _git("rev-parse", "origin/main")

    rows = [
        check_function(name, spec, head_sha, patterns, grace_minutes,
                       reader=reader)
        for name, spec in sorted(registry.items())
    ]
    watched = [r for r in rows if r["verdict"] != "retired"]
    return {
        "head_sha": head_sha,
        "grace_minutes": grace_minutes,
        "deploy_paths": patterns,
        "functions": rows,
        "total": len(watched),
        "measured": sum(1 for r in watched if r["measured"]),
        "drifted": sum(1 for r in watched if r["verdict"] in
                       ("drifted", "deployed_from_dirty_tree")),
        "errors": sum(1 for r in watched if r["severity"] == ERROR),
        "warnings": sum(1 for r in watched if r["severity"] == WARNING),
    }


# ── rendering ────────────────────────────────────────────────────────────────

_ICON = {ERROR: "ERROR", WARNING: "WARN ", OK: "ok   "}


def render(report: dict) -> str:
    lines = [
        f"Lambda deploy-drift — crucible-research @ {report['head_sha'][:8]} "
        f"(grace {report['grace_minutes']}m)",
        "",
    ]
    for row in report["functions"]:
        lines.append(
            f"  {_ICON.get(row['severity'], row['severity'])}  "
            f"{row['function']:<46} {row['verdict']}"
        )
        lines.append(f"        {row['detail']}")
    lines += [
        "",
        # The measurability line (§2.7). Grepped by nothing today; read by a
        # human and carried verbatim into the alert body and job summary, so
        # the same three numbers appear on every surface.
        f"SUMMARY: measured={report['measured']}/{report['total']} "
        f"drifted={report['drifted']} errors={report['errors']} "
        f"warnings={report['warnings']}",
    ]
    return "\n".join(lines)


def render_markdown(report: dict) -> str:
    lines = [
        "### Lambda deploy-drift",
        "",
        f"`crucible-research@{report['head_sha'][:8]}` · grace "
        f"{report['grace_minutes']}m",
        "",
        "| severity | function | verdict | detail |",
        "|---|---|---|---|",
    ]
    for row in report["functions"]:
        detail = row["detail"].replace("|", "\\|")
        lines.append(
            f"| {row['severity']} | `{row['function']}` | "
            f"{row['verdict']} | {detail} |"
        )
    lines += [
        "",
        f"**measured={report['measured']}/{report['total']} "
        f"drifted={report['drifted']} errors={report['errors']} "
        f"warnings={report['warnings']}**",
    ]
    return "\n".join(lines)


def alert_body(report: dict) -> str:
    bad = [r for r in report["functions"] if r["severity"] in (ERROR, WARNING)]
    head = (
        f"Lambda deploy-drift on crucible-research@{report['head_sha'][:8]}: "
        f"measured={report['measured']}/{report['total']} "
        f"drifted={report['drifted']} errors={report['errors']} "
        f"warnings={report['warnings']}."
    )
    detail = "\n".join(
        f"- [{r['severity']}] {r['function']}: {r['verdict']} — {r['detail']}"
        for r in bad
    )
    return f"{head}\n{detail}"


def emit_alert(report: dict) -> None:
    """Publish one rolled-up alert. Never one alert per function — that is
    the storm shape config#2855 already produced once (one latent defect, 11
    red deploys, 11 pages). ``dedup_key`` is keyed on main HEAD so a drift
    that persists across runs pages once per new HEAD, not once per cron
    tick.

    Raises on failure. An alerting path that fails quietly is the exact
    defect this detector exists to remove.
    """
    from krepis.alerts import publish

    severity = ERROR if report["errors"] else WARNING
    publish(
        message=alert_body(report),
        severity=severity,
        source="alpha-engine-research/infrastructure/lambda_deploy_drift.py",
        dedup_key=f"lambda-deploy-drift-{report['head_sha'][:12]}-{severity}",
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--head-sha", default=None,
                    help="main HEAD to compare against (default: origin/main)")
    ap.add_argument("--grace-minutes", type=int, default=45,
                    help="a deployable commit younger than this is "
                         "'pending_deploy', not drift (default: 45)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--emit-alert", action="store_true",
                    help="publish one rolled-up krepis alert when any "
                         "function is ERROR or WARNING")
    args = ap.parse_args(argv)

    report = run(head_sha=args.head_sha, grace_minutes=args.grace_minutes)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render(report))

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(render_markdown(report) + "\n")

    # Annotations go to stdout, where GitHub reads workflow commands — so they
    # must NOT be interleaved with --json output, or the "machine-readable"
    # mode emits something no parser can read. The gate step (no --json) is
    # what annotates; the informational --json step is what a machine consumes.
    for row in ([] if args.json else report["functions"]):
        if row["severity"] == ERROR:
            print(f"::error title={row['function']}::{row['detail']}")
        elif row["severity"] == WARNING:
            print(f"::warning title={row['function']}::{row['detail']}")

    if args.emit_alert and (report["errors"] or report["warnings"]):
        emit_alert(report)

    # ERROR (a trading-critical function is behind main) fails the job so the
    # condition stays red until it is deployed. WARNING is loud on the console
    # and in the alert but does not hold a check red, per I7840 deliverable 3.
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
