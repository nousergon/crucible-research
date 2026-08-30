"""The two retired eval-judge Lambdas must stay retired
(alpha-engine-config-I9329).

``eval_judge_poll_handler`` and ``eval_judge_process_handler`` are gone:
Poll was residue of the retired async batch API (with no batch rung there is
nothing to poll), and Process moved substrate — it runs to completion on a
spot box via ``evals/judge_spot_run.py``, under no execution ceiling.

The failure this guards against is not someone re-adding the files. It is a
DANGLING REFERENCE: a ``COPY`` line in the Dockerfile, a ``deploy.sh`` call, a
workflow step, or an import that names a handler which no longer exists. Each
of those fails at a different time — image build, deploy, or Lambda cold start
in production — and the last one is the expensive one.

Derived by walking TRACKED files, not an enumerated list: an enumeration
cannot see the file someone adds tomorrow. Prose is exempt on purpose — the
history of why these existed is worth keeping, so only EXECUTABLE source is
scanned: Python with comments and string literals tokenized away, and
shell/YAML/Dockerfile with comment lines stripped.
"""

from __future__ import annotations

import io
import subprocess
import tokenize
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: The retired handler module names. A reference to either from executable
#: source is a dangling reference by definition — there is nothing to resolve.
RETIRED_HANDLERS = ("eval_judge_poll_handler", "eval_judge_process_handler")

#: Extensions whose executable content is scanned. Markdown and anything else
#: is prose and exempt.
_PY = {".py"}
_LINE_COMMENT = {".sh", ".yml", ".yaml", ".toml", ".cfg"}
_DOCKERFILE_NAMES = {"Dockerfile"}


def _tracked_files() -> list[Path]:
    out = subprocess.run(  # noqa: S607 — git on PATH, fixture-side
        ["git", "ls-files"], cwd=_REPO_ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    return [_REPO_ROOT / line for line in out if line.strip()]


def _executable_text(path: Path) -> str | None:
    """The file's content with prose removed, or None if it is not scanned."""
    name = path.name
    suffix = path.suffix

    if suffix in _PY:
        try:
            with open(path, "rb") as fh:
                toks = list(tokenize.tokenize(io.BytesIO(fh.read()).readline))
        except (tokenize.TokenError, SyntaxError, UnicodeDecodeError):
            # A file that will not tokenize is a different failure; do not
            # silently exempt it — scan it raw rather than skipping it.
            return path.read_text(encoding="utf-8", errors="replace")
        return " ".join(
            t.string for t in toks
            if t.type not in (tokenize.COMMENT, tokenize.STRING)
        )

    if suffix in _LINE_COMMENT or name in _DOCKERFILE_NAMES or name.startswith("Dockerfile"):
        return "\n".join(
            line for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if not line.lstrip().startswith("#")
        )

    return None


def test_the_retired_handler_files_are_gone():
    for handler in RETIRED_HANDLERS:
        path = _REPO_ROOT / "lambda" / f"{handler}.py"
        assert not path.exists(), (
            f"{path} is back — the SF no longer has a state for it "
            "(alpha-engine-config-I9329)"
        )


def test_no_executable_reference_to_a_retired_handler_anywhere():
    offenders: list[str] = []
    for path in _tracked_files():
        if not path.is_file():
            continue
        if path.resolve() == Path(__file__).resolve():
            continue  # this file names them on purpose
        text = _executable_text(path)
        if text is None:
            continue
        for handler in RETIRED_HANDLERS:
            if handler in text:
                offenders.append(f"{path.relative_to(_REPO_ROOT)} -> {handler}")

    assert offenders == [], (
        "dangling reference(s) to a retired eval-judge Lambda handler: "
        f"{offenders}. Each of these fails at a different time — image "
        "build, deploy, or a cold start in production."
    )


def test_the_live_function_names_are_gone_from_the_deploy_surface():
    """The deploy script and the drift registry are the two places a retired
    function can keep being published from or reported on.
    """
    surfaces = {
        "infrastructure/deploy.sh",
        "infrastructure/lambda_deploy_drift.py",
        ".github/workflows/deploy.yml",
    }
    offenders: list[str] = []
    for rel in sorted(surfaces):
        text = _executable_text(_REPO_ROOT / rel) or ""
        for fn in ("alpha-engine-research-eval-judge-poll",
                   "alpha-engine-research-eval-judge-process"):
            if fn in text:
                offenders.append(f"{rel} -> {fn}")
    assert offenders == [], (
        f"retired live function name(s) still on the deploy surface: {offenders}"
    )
