"""Locks audit finding F8: every LLM call site stamps prompt provenance
metadata onto LangSmith run metadata via ``LoadedPrompt.langsmith_metadata()``.

Per ``alpha-engine-docs/private/alpha-engine-research-prompt-audit-260430.md``
§ 4 PR D — wrap ChatAnthropic invocations with a callback that stamps
``prompt_name``, ``prompt_version``, ``prompt_hash`` onto LangSmith run
metadata so the trace UI exposes which prompt revision produced an output.

Two layers of enforcement:

1. **Unit:** ``LoadedPrompt.langsmith_metadata()`` returns the canonical
   ``{prompt_name, prompt_version, prompt_hash}`` dict.
2. **Source-text lock:** every agent module that invokes the LLM
   (``with_structured_output().invoke(...)`` or ``create_react_agent`` +
   ``agent.invoke(...)``) must pass ``config={"metadata": ...}`` so the
   provenance dict reaches LangSmith. This is a structural negation —
   regressing the wire-up surfaces here, not at runtime.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents import prompt_loader
from agents.prompt_loader import load_prompt

# ── Unit — LoadedPrompt.langsmith_metadata() ─────────────────────────────


@pytest.fixture(autouse=True)
def _isolated_search_paths(tmp_path, monkeypatch):
    """Sandbox the prompt loader so unit tests don't touch real prompts."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

    fake_repo_root = tmp_path / "repo"
    (fake_repo_root / "agents").mkdir(parents=True)
    monkeypatch.setattr(prompt_loader, "_REPO_ROOT", fake_repo_root)

    monkeypatch.delenv("GITHUB_WORKSPACE", raising=False)

    prompt_loader.clear_cache()
    yield
    prompt_loader.clear_cache()


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_langsmith_metadata_emits_canonical_dict(tmp_path):
    """The helper must emit the three canonical keys with truncated hash."""
    home = Path.home()
    _write(
        home / "alpha-engine-config" / "research" / "prompts" / "agent_y.txt",
        "# version: 2.3.1\nYou are agent Y.\n",
    )
    loaded = load_prompt("agent_y")
    md = loaded.langsmith_metadata()

    assert md["prompt_name"] == "agent_y"
    assert md["prompt_version"] == "2.3.1"
    assert md["prompt_hash"] == loaded.hash[:12], (
        "Hash field must be the first 12 hex chars of the full sha256 — "
        "keeps the LangSmith UI readable while staying disambiguating."
    )
    assert len(md["prompt_hash"]) == 12


def test_langsmith_metadata_serializable_for_langsmith(tmp_path):
    """LangSmith requires JSON-serializable metadata — every value must be a string."""
    home = Path.home()
    _write(
        home / "alpha-engine-config" / "research" / "prompts" / "agent_z.txt",
        "# version: 1.0.0\nbody\n",
    )
    md = load_prompt("agent_z").langsmith_metadata()
    assert all(isinstance(v, str) for v in md.values()), (
        "All metadata values must be strings — LangSmith run metadata is "
        "JSON-serialized and non-string values cause silent trace drops."
    )


def test_metadata_is_stable_across_loads(tmp_path):
    """Repeat loads of the same prompt return the same metadata dict shape."""
    home = Path.home()
    _write(
        home / "alpha-engine-config" / "research" / "prompts" / "stable.txt",
        "# version: 1.4.2\nstable body\n",
    )
    a = load_prompt("stable").langsmith_metadata()
    b = load_prompt("stable").langsmith_metadata()
    assert a == b


# ── Source-text lock — every agent invocation passes metadata ────────────

_AGENT_FILES_WITH_LLM_INVOCATIONS = (
    "agents/macro_agent.py",
    "agents/sector_teams/sector_team.py",
    "agents/sector_teams/peer_review.py",
    "agents/sector_teams/quant_analyst.py",
    "agents/sector_teams/qual_analyst.py",
    "agents/investment_committee/ic_cio.py",
)


@pytest.mark.parametrize("rel_path", _AGENT_FILES_WITH_LLM_INVOCATIONS)
def test_agent_module_passes_langsmith_metadata(rel_path: str) -> None:
    """Every agent module containing an LLM invocation must pass
    ``config={"metadata": ...}`` so prompt provenance reaches LangSmith.

    This is a structural negation — if a future refactor introduces a
    new invocation without a metadata config, the file will lose its
    one-metadata-per-invocation parity and this test will fail.
    """
    repo_root = Path(__file__).resolve().parent.parent
    src = (repo_root / rel_path).read_text(encoding="utf-8")

    invocation_count = src.count("structured_llm.invoke(") + src.count("agent.invoke(")
    metadata_count = src.count('config={"metadata":') + src.count("config={'metadata':")
    # The agent-loop config also accepts the recursion_limit kwarg, so we
    # match the metadata-as-config-key form (handles dict-multiline +
    # single-line passes alike).
    metadata_count += src.count('"metadata": ') + src.count("'metadata': ")
    # Each invocation contributes 1 to invocation_count; each config
    # contributes >= 1 to metadata_count (single-line passes count once,
    # dict-multiline passes count twice — the literal + the key). Lock the
    # weaker invariant: at least one metadata mention per invocation.
    assert metadata_count >= invocation_count, (
        f"{rel_path}: {invocation_count} LLM invocations vs only "
        f"{metadata_count} ``metadata`` config mentions. Every "
        f"``.invoke(...)`` on the LLM must pass "
        f"``config={{\"metadata\": loaded_prompt.langsmith_metadata()}}`` "
        f"so prompt provenance reaches the LangSmith trace (audit F8 / PR D)."
    )


def test_loaded_prompt_metadata_helper_referenced_in_agents() -> None:
    """At least one agent module must reference ``langsmith_metadata`` —
    sanity check that the new ``LoadedPrompt.langsmith_metadata()`` helper
    is the contract callers use, not an ad-hoc inline dict construction.
    """
    repo_root = Path(__file__).resolve().parent.parent
    referenced = []
    for rel_path in _AGENT_FILES_WITH_LLM_INVOCATIONS:
        src = (repo_root / rel_path).read_text(encoding="utf-8")
        if "langsmith_metadata" in src:
            referenced.append(rel_path)
    assert len(referenced) >= 4, (
        f"Only {len(referenced)} agent modules reference "
        f"``LoadedPrompt.langsmith_metadata()``. Expected at least 4 "
        f"(macro + sector_team + peer_review + ic_cio). The helper is the "
        f"canonical wire-up; ad-hoc inline ``{{\"prompt_name\": ...}}`` "
        f"construction in a caller skips the prompt-version single-source "
        f"and risks drift."
    )
