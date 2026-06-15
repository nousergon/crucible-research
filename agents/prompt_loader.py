"""
Prompt loader — reads agent prompt templates from the alpha-engine-config repo.

Returns a ``LoadedPrompt`` dataclass carrying ``name``, ``version``, ``hash``,
``text``, and ``source_path``. Existing callers can keep using ``.format(**kw)``
as a drop-in replacement for ``str.format`` — ``LoadedPrompt`` exposes the
same method.

Search order mirrors ``config.py::_find_config`` so prompts and YAML configs
resolve through the same private-config-repo discovery logic:

  1. ~/alpha-engine-config/research/prompts/<name>.txt        (local dev, sibling)
  2. <repo>/../alpha-engine-config/research/prompts/<name>.txt (local dev, parent)
  3. $GITHUB_WORKSPACE/alpha-engine-config/research/prompts/<name>.txt (CI)
  4. <repo>/config/prompts/<name>.txt                         (Lambda image —
     deploy.sh stages from the config repo into this directory)

Hard-fail on miss: there is **no** fallback to ``config/prompts.example/``.
That fallback violated ``feedback_no_example_fallback.md`` — a Lambda boot
with example prompts after a deploy bug would silently degrade signal quality.

Optional version frontmatter — first non-blank line may be:

    # version: 1.2.3

The parsed semver is exposed as ``LoadedPrompt.version``. Missing frontmatter
means version = "0.0.0" (graceful upgrade — prompts pre-versioning still load).

The body hash (``LoadedPrompt.hash``) is sha256 over the post-frontmatter,
trailing-whitespace-normalized body. Used downstream to stamp LangSmith run
metadata for prompt-vs-prompt drift detection.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Experiment package id (HARNESS_EXPERIMENT_CLASSIFICATION.md §3)
_EXPERIMENT_ID = os.environ.get("ALPHA_ENGINE_EXPERIMENT_ID", "reference")
_VERSION_LINE_RE = re.compile(r"^#\s*version\s*:\s*(\S+)\s*$")
_DEFAULT_VERSION = "0.0.0"


@dataclass(frozen=True)
class LoadedPrompt:
    """A loaded prompt template with provenance metadata."""

    name: str
    text: str
    version: str
    hash: str
    source_path: Path

    def format(self, **kwargs: Any) -> str:
        """Drop-in for ``str.format`` — render the template with kwargs."""
        return self.text.format(**kwargs)

    def langsmith_metadata(self) -> dict[str, str]:
        """Return the prompt-provenance metadata dict for LangSmith stamping.

        Pass into ``llm.invoke(messages, config={"metadata": prompt.langsmith_metadata()})``
        at every LLM call site so the LangSmith trace UI exposes which prompt
        revision produced an output. Closes audit finding F8 — the version +
        hash are computed at load time and propagated via this helper rather
        than being looked up ad-hoc at each call site.

        ``hash`` is truncated to its first 12 hex chars so the LangSmith UI
        stays readable; full sha256 lives in ``LoadedPrompt.hash`` for any
        consumer that needs the full digest.
        """
        return {
            "prompt_name": self.name,
            "prompt_version": self.version,
            "prompt_hash": self.hash[:12],
        }


_cache: dict[str, LoadedPrompt] = {}


def load_prompt(name: str) -> LoadedPrompt:
    """Locate, parse, and cache the prompt template named ``name``.

    Args:
        name: filename without ``.txt`` extension (e.g. ``"macro_agent"``).

    Returns:
        ``LoadedPrompt`` carrying text + version + content hash.

    Raises:
        FileNotFoundError: if the prompt is not found in any search path.
            The error names every path that was tried so an operator can
            diagnose which staging step (deploy.sh, sibling clone, CI
            checkout) is missing.
    """
    if name in _cache:
        return _cache[name]

    path = _resolve_prompt_path(name)
    raw = path.read_text(encoding="utf-8")
    version, body = _split_frontmatter(raw)
    body_hash = _hash_body(body)

    loaded = LoadedPrompt(
        name=name,
        text=body,
        version=version,
        hash=body_hash,
        source_path=path,
    )
    _cache[name] = loaded
    logger.debug(
        "Loaded prompt %s v%s hash=%s from %s (%d chars)",
        name, version, body_hash[:12], path, len(body),
    )
    return loaded


def clear_cache() -> None:
    """Drop the in-process cache. Test-only — production loads at import."""
    _cache.clear()


def _resolve_prompt_path(name: str) -> Path:
    """Return the first existing path; raise ``FileNotFoundError`` otherwise."""
    filename = f"{name}.txt"
    ws = os.environ.get("GITHUB_WORKSPACE")
    search = [
        Path.home() / "alpha-engine-config" / "experiments" / _EXPERIMENT_ID / "research" / "prompts" / filename,
        Path.home() / "alpha-engine-config" / "research" / "prompts" / filename,
        _REPO_ROOT.parent / "alpha-engine-config" / "experiments" / _EXPERIMENT_ID / "research" / "prompts" / filename,
        _REPO_ROOT.parent / "alpha-engine-config" / "research" / "prompts" / filename,
    ]
    if ws:
        search.append(
            Path(ws) / "alpha-engine-config" / "experiments" / _EXPERIMENT_ID / "research" / "prompts" / filename
        )
        search.append(
            Path(ws) / "alpha-engine-config" / "research" / "prompts" / filename
        )
    search.append(_REPO_ROOT / "config" / "prompts" / filename)

    for p in search:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Prompt '{name}' not found. Searched:\n  "
        + "\n  ".join(str(p) for p in search)
        + "\nFix: clone nousergon/alpha-engine-config at "
        + "~/alpha-engine-config (local dev) or stage via "
        + "infrastructure/deploy.sh (Lambda build). "
        + "There is no .example fallback by design — see "
        + "feedback_no_example_fallback.md."
    )


def _split_frontmatter(raw: str) -> tuple[str, str]:
    """Parse optional ``# version: X.Y.Z`` frontmatter from the first non-blank line.

    Returns ``(version, body_without_frontmatter)``. If no frontmatter, returns
    ``("0.0.0", raw)``. Frontmatter line + the single newline that follows it
    are stripped from the body so callers see clean template text.
    """
    lines = raw.split("\n")
    idx = 0
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    if idx >= len(lines):
        return _DEFAULT_VERSION, raw

    match = _VERSION_LINE_RE.match(lines[idx])
    if match is None:
        return _DEFAULT_VERSION, raw

    version = match.group(1)
    body_lines = lines[:idx] + lines[idx + 1:]
    return version, "\n".join(body_lines)


def _hash_body(body: str) -> str:
    """Hash the body after rstripping each line + trailing blank lines.

    Insulates the hash from editor-driven trailing-whitespace + final-newline
    drift, while still detecting content changes.
    """
    normalized = "\n".join(line.rstrip() for line in body.split("\n")).rstrip("\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
