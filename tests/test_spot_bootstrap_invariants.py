"""``infrastructure/spot_research_weekly.sh`` must not carry a hand-written
spot bootstrap — inline or as a silently-reappearing interpreter fallback.

## Why this test exists

Before this cutover (alpha-engine-config-I7372), the ``bootstrap`` `run_ssm`
step was a hand-written heredoc: a systemd transient timer, a `dnf install`
interpreter install, and a `git clone` with a spot-expanded `${BRANCH}`. It
duplicated the SAME shape independently built in `nousergon-data` and
`crucible-predictor`'s `infrastructure/_spot_common.sh`, which diverged and
cost three hand-carried fixes in two days (alpha-engine-config-I6922). This
repo is the first of five cutovers onto the shared renderer,
`krepis.spot_bootstrap` — its shape is the reference the rest are checked
against.

Every downstream `run_ssm` step (deps, preflight-only, weekly) also carried a
SEPARATE defect: a shared `ENV_SOURCE` heredoc resolving the interpreter as
``command -v python3.12 >/dev/null && PYTHON_BIN=python3.12 || PYTHON_BIN=python3``
— a silent fallback to the AMI's system python3 when 3.12 is absent, even
though ``requirements.txt`` is resolved against 3.12 and the wheels differ.
This class survived the two already-merged sibling cutovers
(`nousergon-data-PR1388`, `crucible-predictor-PR504`): both left the identical
fallback standing in their `install_deps()` steps, because
`scan_for_inline_bootstraps` cannot see it — a file carrying only the
fallback trips fewer than `MIN_CATEGORIES` signature categories, so the
fork-detector is silent on it by design. This file's second assertion group
exists because that detector alone is not enough.

## What this test asserts, and why it is not a second hardcoded copy

1. `scan_for_inline_bootstraps` against the repo root returns nothing — the
   anti-regression against the heredoc (watchdog / dnf install / git clone)
   creeping back anywhere under `infrastructure/`, `scripts/` or `bin/`,
   under any filename.
2. The CLI arguments the launcher actually passes to `krepis.spot_bootstrap
   render` are extracted from the live script and asserted against the spec
   this repo owes the renderer (repo, checkout, branch, region,
   max-runtime-seconds, the `S3_STAGING` export) — a wrong argument fails
   here, not on a Saturday.
3. Those extracted arguments are fed through `render_bootstrap` and the
   watchdog/interpreter invariants are re-asserted against the RENDERED
   output — unchanged from before the cutover; only their subject moved from
   a heredoc literal to a rendered string.
4. A SEPARATE, derived scan over every `.sh` file under `infrastructure/`,
   `scripts/` and `bin/` for the silent-fallback SELECTION pattern
   (``command -v python3.12 ... && VAR=python3.12 || VAR=python3``) — this is
   what the fork detector cannot see, and what let the class survive two
   already-shipped sibling cutovers.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from krepis.spot_bootstrap import (
    PYTHON,
    SYSTEMCTL_ENABLE_TIMEOUT_SEC,
    SpotBootstrapSpec,
    render_bootstrap,
    scan_for_inline_bootstraps,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "infrastructure" / "spot_research_weekly.sh"

#: The silent-fallback SELECTION the fork detector cannot see (fewer than
#: MIN_CATEGORIES signature categories fire on this pattern alone). Matches
#: both the `&&`/`||` shell-selection form and a ternary-shaped equivalent —
#: anything that lands on a bare `python3` when `python3.12` is absent.
_SILENT_FALLBACK_RE = re.compile(
    r"command\s+-v\s+python3\.12[^\n]*\|\|[^\n]*\bPYTHON(?:_BIN)?=python3\b"
)


# ── No inline bootstrap anywhere in the tree ─────────────────────────────────


def test_no_inline_spot_bootstraps_anywhere_in_the_repo():
    findings = scan_for_inline_bootstraps(_REPO_ROOT)
    assert findings == [], (
        "inline spot-bootstrap heredoc(s) found outside krepis.spot_bootstrap "
        f"dispatch: {[str(f) for f in findings]}"
    )


# ── The dispatch itself ───────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def script_text() -> str:
    return _SCRIPT.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def render_args(script_text: str) -> list[str]:
    """The literal argv passed to ``krepis.spot_bootstrap render``, extracted
    from the live ``_BOOTSTRAP_SCRIPT="$("$LIB_PYTHON" -m krepis.spot_bootstrap
    render \\ ... )"`` assignment — this test reads the actual call, not a
    restatement of it.
    """
    import shlex

    m = re.search(
        r'"\$LIB_PYTHON"\s+-m\s+krepis\.spot_bootstrap\s+render\s*\\(.*?)\)"',
        script_text,
        re.S,
    )
    assert m, "spot_research_weekly.sh no longer dispatches krepis.spot_bootstrap render"
    joined = m.group(1).replace("\\\n", " ")
    return shlex.split(joined)


def _flag_value(args: list[str], flag: str) -> str:
    assert flag in args, f"{flag} missing from the krepis.spot_bootstrap render call"
    return args[args.index(flag) + 1]


@pytest.fixture(scope="module")
def rendered(render_args: list[str]) -> str:
    """The script this launcher's bootstrap step actually sends, rendered
    exactly as the launcher would. ``--branch`` is ``"${BRANCH:-main}"`` — a
    runtime shell default, asserted separately below — and ``--export
    S3_STAGING=...`` embeds the runtime ``$S3_STAGING`` value; neither affects
    the watchdog or interpreter blocks asserted against here.
    """
    spec = SpotBootstrapSpec(
        repo_url=_flag_value(render_args, "--repo-url"),
        checkout=_flag_value(render_args, "--checkout"),
        region=_flag_value(render_args, "--region"),
        branch="main",
        max_runtime_seconds=None,  # asserted separately via the flag's presence
        exports={"S3_STAGING": "s3://placeholder/staging"},
    )
    return render_bootstrap(spec)


def test_bootstrap_no_longer_inlines_the_heredoc(script_text: str):
    """Anti-regression: the whole point of the cutover is this heredoc is gone."""
    assert "systemctl enable" not in script_text, (
        "the launcher still inlines the systemd watchdog unit — the cutover to "
        "krepis.spot_bootstrap (alpha-engine-config-I7372) is meant to delete it, "
        "not duplicate it"
    )
    assert "dnf install" not in script_text, (
        "the launcher still inlines a dnf install — that belongs in "
        "krepis.spot_bootstrap._interpreter_block() now"
    )
    assert not re.search(r"git\s+clone\b[^\n]*\$\{?BRANCH", script_text), (
        "the launcher still names a shell variable in a git clone line — "
        "the renderer bakes repo/branch in as launcher-side literals"
    )


def test_repo_url_and_checkout(render_args: list[str]):
    assert _flag_value(render_args, "--repo-url") == (
        "https://github.com/nousergon/crucible-research.git"
    )
    assert _flag_value(render_args, "--checkout") == "/home/ec2-user/research"


def test_branch_is_a_launcher_side_literal_with_the_main_default(script_text: str):
    """crucible-predictor#463: a value interpolated into the heredoc but never
    exported resolved to an empty string on the spot; passing it through the
    renderer's argv removes that class of bug entirely.
    """
    assert re.search(r'--branch\s+"\$\{BRANCH:-main\}"', script_text), (
        'the launcher must pass --branch "${BRANCH:-main}" — a launcher-side '
        "literal preserving the pre-cutover default, not a remote expansion"
    )


def test_s3_staging_export_is_load_bearing(script_text: str):
    """The appended tarball-staging block's `${S3_STAGING}/...` needs this in
    the remote environment."""
    assert re.search(r'--export\s+"S3_STAGING=\$\{S3_STAGING\}"', script_text), (
        'the launcher must pass --export "S3_STAGING=${S3_STAGING}" — the '
        "appended tarball-staging block's aws s3 cp references ${S3_STAGING} "
        "in the remote environment and dies without it"
    )


def test_max_runtime_seconds_hard_cap_is_carried_over(script_text: str):
    """The pre-cutover heredoc armed `systemd-run --on-active=${MAX_RUNTIME_
    SECONDS}` as an on-box hard-timeout self-destruct. Dropping the cap on
    cutover is the silent un-shipping failure the fleet-wide arc exists to
    prevent — it must still be passed to the renderer, not merely left as a
    dead shell variable.
    """
    assert re.search(r'--max-runtime-seconds\s+"\$MAX_RUNTIME_SECONDS"', script_text), (
        'the launcher must pass --max-runtime-seconds "$MAX_RUNTIME_SECONDS" — '
        "the pre-cutover heredoc armed a hard on-box timeout at this value and "
        "the cutover must not silently drop that guarantee"
    )


def test_region_is_the_pre_cutover_variable(render_args: list[str]):
    """Unlike the sibling repos (which hardcoded us-east-1), this repo's
    pre-cutover heredoc always interpolated the launcher's own $AWS_REGION —
    preserved here rather than hardcoded, to avoid silently changing behavior
    for a launcher that overrides AWS_REGION.
    """
    assert _flag_value(render_args, "--region") == "$AWS_REGION"


# ── The watchdog unit (asserted against the RENDERED output) ────────────────


def _service_directives(script: str) -> set[str]:
    m = re.search(r"\[Service\]\n(.*?)(?=\n\[|\nUNIT\b)", script, re.S)
    assert m, "no [Service] section found in the systemd unit"
    return {
        line.strip()
        for line in m.group(1).splitlines()
        if "=" in line and not line.strip().startswith("#")
    }


def test_watchdog_unit_is_never_oneshot(rendered: str):
    assert "Type=oneshot" not in _service_directives(rendered), (
        "the ec2-spot-watchdog unit declares Type=oneshot; its ExecStart never "
        "returns, so `systemctl enable --now` blocks forever (nousergon-data#1294, "
        "crucible-predictor#461)"
    )


def test_enabling_the_watchdog_is_bounded_by_the_canonical_timeout(rendered: str):
    assert re.search(
        rf"timeout\s+{SYSTEMCTL_ENABLE_TIMEOUT_SEC}\s+systemctl\s+enable\s+--now\s+ec2-spot-watchdog",
        rendered,
    ), (
        "`systemctl enable --now ec2-spot-watchdog` must be wrapped in "
        f"`timeout {SYSTEMCTL_ENABLE_TIMEOUT_SEC}` "
        "(krepis.spot_bootstrap.SYSTEMCTL_ENABLE_TIMEOUT_SEC)"
    )


def test_the_interpreter_is_installed_not_merely_asserted(rendered: str):
    assert re.search(rf"dnf install[^\n]*{re.escape(PYTHON)}\b", rendered), (
        f"the rendered bootstrap must install {PYTHON}, not merely assert it is present"
    )
    assert re.search(rf"command\s+-v\s+{re.escape(PYTHON)}\s*>/dev/null\s*\|\|", rendered), (
        f"the rendered bootstrap must assert {PYTHON} is present as a POST-condition"
    )


def test_the_interpreter_has_no_fallback_in_the_rendered_bootstrap(rendered: str):
    assert not _SILENT_FALLBACK_RE.search(rendered), (
        "the rendered bootstrap must never silently fall back to a bare python3"
    )


def test_clone_uses_no_spot_expanded_variables(rendered: str):
    assert "${BRANCH}" not in rendered and "${REPO_URL}" not in rendered, (
        "the rendered clone must bake repo/branch in as literals, not read "
        "them from the remote shell (crucible-predictor#463)"
    )


# ── The tarball-staging tail (repo-specific; renderer cannot express it) ────


def test_tarball_staging_tail_is_appended_after_the_rendered_script(script_text: str):
    """This repo's tarball extract + prompts-present assertion is not
    expressible by `--config-copy` (a plain `aws s3 cp`) — it must remain, but
    appended after the rendered script into the SAME run_ssm "bootstrap" call,
    never as a second inline bootstrap dispatched separately.
    """
    assert "research-config.tgz" in script_text
    assert "tar -xzf /tmp/research-config.tgz -C /home/ec2-user/alpha-engine-config/research" in script_text
    assert "staged prompts missing after extract" in script_text
    # Exactly one run_ssm "bootstrap" dispatch (excluding comment mentions) —
    # the tail is concatenated in, not sent as a second SSM step.
    code_lines = [
        line for line in script_text.splitlines() if not line.lstrip().startswith("#")
    ]
    assert sum(line.count('run_ssm "bootstrap"') for line in code_lines) == 1


# ── The silent interpreter fallback — NOT caught by scan_for_inline_bootstraps ─


def test_no_silent_interpreter_fallback_anywhere_in_the_repo():
    """Derived, not enumerated: scans every `.sh` file under `infrastructure/`,
    `scripts/` and `bin/` for the fallback SELECTION shape, independent of
    scan_for_inline_bootstraps (which cannot see a file carrying only this
    one signature — it never reaches MIN_CATEGORIES). Two already-merged
    sibling cutovers (nousergon-data-PR1388, crucible-predictor-PR504) each
    left this exact pattern standing in their install_deps() step; this
    assertion is what stops that recurrence here.
    """
    offenders: list[str] = []
    for subdir in ("infrastructure", "scripts", "bin"):
        base = _REPO_ROOT / subdir
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.sh")):
            text = path.read_text(encoding="utf-8", errors="replace")
            code = "\n".join(
                line for line in text.splitlines() if not line.lstrip().startswith("#")
            )
            if _SILENT_FALLBACK_RE.search(code):
                offenders.append(str(path.relative_to(_REPO_ROOT)))
    assert offenders == [], (
        f"silent python3.12->python3 fallback found in: {offenders} — every "
        "interpreter resolution must be strict (exit non-zero when 3.12 is "
        "absent), never a silent selection onto the AMI's system python3"
    )


# ── A heredoc `git clone` beside the delegate marker — scan_for_inline_ ─────
# ── bootstraps() clears a WHOLE FILE on one delegate match (alpha-engine- ──
# ── config-I7378), so it cannot see a heredoc reintroduced next to the ─────
# ── krepis.spot_bootstrap render call. This assertion does not depend on ──
# ── that scanner and catches exactly that case. ─────────────────────────────

_HEREDOC_RE = re.compile(r"<<-?\s*'?([A-Za-z_][A-Za-z0-9_]*)'?\n(.*?)\n\1\b", re.S)


def test_no_heredoc_git_clone_into_home_ec2_user_anywhere_in_the_repo():
    """krepis-I7378: `scan_for_inline_bootstraps` clears a whole file the
    instant it sees ANY `-m krepis.spot_bootstrap render` line, before
    evaluating signatures elsewhere in the same file — so a heredoc
    reintroduced beside a legitimate render call (this file's own bootstrap
    dispatch, permanently) would be invisible to it. This assertion is
    file-scoped to heredoc BODIES only, independent of that scanner, and
    fires on exactly the shape that bug would miss: a `git clone` into
    `/home/ec2-user/` inside any heredoc under `infrastructure/`, `scripts/`
    or `bin/`. Do NOT edit krepis to fix the scanner itself — filed as
    alpha-engine-config-I7378 (P1/mid); this is the repo-local backstop.
    """
    offenders: list[str] = []
    for subdir in ("infrastructure", "scripts", "bin"):
        base = _REPO_ROOT / subdir
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.sh")):
            text = path.read_text(encoding="utf-8", errors="replace")
            for _marker, body in _HEREDOC_RE.findall(text):
                if re.search(r"git\s+clone\b[^\n]*/home/ec2-user/", body):
                    offenders.append(str(path.relative_to(_REPO_ROOT)))
                    break
    assert offenders == [], (
        f"heredoc containing a git clone into /home/ec2-user/ found in: "
        f"{offenders} — a reintroduced inline bootstrap can hide beside a "
        "legitimate krepis.spot_bootstrap render call (krepis-I7378); every "
        "clone into that path belongs to the renderer, never a heredoc"
    )
