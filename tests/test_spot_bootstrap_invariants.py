"""Repo-WIDE spot-bootstrap negative invariants — no hand-written bootstrap,
no silently-reappearing interpreter fallback, anywhere under `infrastructure/`,
`scripts/` or `bin/`.

## What changed on 2026-08-20 (`alpha-engine-config-I7856`)

This file used to take `infrastructure/spot_research_weekly.sh` as its subject.
That launcher was this repo's only spot launcher and it was deleted with the
retired research graph it drove (`crucible-research-PR685`). Its
`krepis.spot_bootstrap render` DISPATCH-SHAPE assertions — the extracted CLI
argv, the rendered watchdog/interpreter blocks, the staging tail — were
properties of that one script and had no other subject here, so they are gone;
live copies stand over live launchers in `nousergon-data` and
`crucible-predictor` (`tests/test_spot_bootstrap_invariants.py` in each).

What is KEPT is the half that never had a single subject: three sweeps derived
over the whole tree. They still cover `infrastructure/thinktank_spot_bootstrap.sh`
and every `.sh` added after this file was written, so nothing here went vacuous.
See `infrastructure/README.md` for the full five-invariant disposition table.

## Why these three exist

Before the `alpha-engine-config-I7372` cutover, the `bootstrap` step was a
hand-written heredoc: a systemd transient timer, a `dnf install` interpreter
install, and a `git clone` with a spot-expanded `${BRANCH}`. It duplicated the
SAME shape independently built in `nousergon-data` and `crucible-predictor`'s
`infrastructure/_spot_common.sh`, which diverged and cost three hand-carried
fixes in two days (`alpha-engine-config-I6922`).

Every downstream `run_ssm` step also carried a SEPARATE defect: a shared
`ENV_SOURCE` heredoc resolving the interpreter as
``command -v python3.12 >/dev/null && PYTHON_BIN=python3.12 || PYTHON_BIN=python3``
— a silent fallback to the AMI's system python3 when 3.12 is absent, even
though ``requirements.txt`` is resolved against 3.12 and the wheels differ.
This class survived the two already-merged sibling cutovers
(`nousergon-data-PR1388`, `crucible-predictor-PR504`): both left the identical
fallback standing in their `install_deps()` steps, because
`scan_for_inline_bootstraps` cannot see it — a file carrying only the fallback
trips fewer than `MIN_CATEGORIES` signature categories, so the fork-detector is
silent on it by design.

## The three assertions

1. `scan_for_inline_bootstraps` against the repo root returns nothing — the
   anti-regression against the heredoc (watchdog / dnf install / git clone)
   creeping back anywhere under `infrastructure/`, `scripts/` or `bin/`, under
   any filename.
2. A derived scan for the silent-fallback SELECTION pattern, which the fork
   detector structurally cannot see.
3. A derived scan for a heredoc `git clone` into `/home/ec2-user/`, which
   `scan_for_inline_bootstraps` also cannot see once a file contains any
   legitimate `krepis.spot_bootstrap render` line (`krepis-I7378`).
"""

from __future__ import annotations

import re
from pathlib import Path

from krepis.spot_bootstrap import scan_for_inline_bootstraps

_REPO_ROOT = Path(__file__).resolve().parents[1]

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
