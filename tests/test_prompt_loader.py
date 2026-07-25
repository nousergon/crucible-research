"""Tests for ``agents.prompt_loader.load_prompt`` and ``LoadedPrompt``.

Locks the version-frontmatter parsing, hash stability, hard-fail-on-miss
behavior, and the ``.format(**kwargs)`` drop-in compatibility that PR A
of the prompt-versioning workstream depends on. Subsequent PRs (B/C/D) +
the LangSmith metadata propagation work read these fields directly.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from agents import prompt_loader
from agents.prompt_loader import LoadedPrompt, load_prompt


@pytest.fixture(autouse=True)
def _isolated_search_paths(tmp_path, monkeypatch):
    """Redirect every search root to a sandbox so tests can't pick up real prompts."""
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


def test_load_with_version_frontmatter(tmp_path):
    home = Path.home()
    _write(
        home / "alpha-engine-config" / "research" / "prompts" / "agent_x.txt",
        "# version: 1.4.2\nYou are agent X.\nProduce JSON.\n",
    )
    loaded = load_prompt("agent_x")

    assert isinstance(loaded, LoadedPrompt)
    assert loaded.name == "agent_x"
    assert loaded.version == "1.4.2"
    assert loaded.text == "You are agent X.\nProduce JSON.\n"
    assert "# version" not in loaded.text
    assert len(loaded.hash) == 64


def test_load_without_frontmatter_defaults_version_to_zero(tmp_path):
    home = Path.home()
    _write(
        home / "alpha-engine-config" / "research" / "prompts" / "legacy.txt",
        "You are a legacy prompt with no version.\n",
    )
    loaded = load_prompt("legacy")

    assert loaded.version == "0.0.0"
    assert loaded.text == "You are a legacy prompt with no version.\n"


def test_frontmatter_skips_leading_blank_lines(tmp_path):
    home = Path.home()
    _write(
        home / "alpha-engine-config" / "research" / "prompts" / "padded.txt",
        "\n\n# version: 2.0.0\nBody after blanks.\n",
    )
    loaded = load_prompt("padded")

    assert loaded.version == "2.0.0"
    assert loaded.text.strip() == "Body after blanks."


def test_format_drop_in_replacement_for_str_format(tmp_path):
    home = Path.home()
    _write(
        home / "alpha-engine-config" / "research" / "prompts" / "fmt.txt",
        "# version: 1.0.0\nHello {name}, regime is {regime}.",
    )
    loaded = load_prompt("fmt")

    rendered = loaded.format(name="Brian", regime="bull")
    assert rendered == "Hello Brian, regime is bull."


def test_hash_is_stable_under_trailing_whitespace_drift(tmp_path):
    home = Path.home()
    _write(
        home / "alpha-engine-config" / "research" / "prompts" / "clean.txt",
        "# version: 1.0.0\nLine A\nLine B\n",
    )
    _write(
        home / "alpha-engine-config" / "research" / "prompts" / "noisy.txt",
        "# version: 1.0.0\nLine A   \nLine B\t\n\n\n",
    )
    clean = load_prompt("clean")
    noisy = load_prompt("noisy")

    assert clean.hash == noisy.hash, (
        "Hash must normalize trailing whitespace + final newlines so "
        "editor settings don't trigger spurious drift alerts."
    )


def test_hash_changes_on_real_content_change(tmp_path):
    home = Path.home()
    _write(
        home / "alpha-engine-config" / "research" / "prompts" / "v1.txt",
        "# version: 1.0.0\nOriginal body.\n",
    )
    _write(
        home / "alpha-engine-config" / "research" / "prompts" / "v2.txt",
        "# version: 1.0.0\nUpdated body.\n",
    )
    v1 = load_prompt("v1")
    v2 = load_prompt("v2")

    assert v1.hash != v2.hash


def test_hard_fail_on_missing_prompt_lists_search_paths(tmp_path):
    with pytest.raises(FileNotFoundError) as excinfo:
        load_prompt("does_not_exist")

    msg = str(excinfo.value)
    assert "does_not_exist" in msg
    assert "alpha-engine-config" in msg
    assert "feedback_no_example_fallback.md" in msg, (
        "Error must reference the feedback rule so future contributors "
        "understand why .example fallback was removed."
    )


def test_no_silent_fallback_to_prompts_example(tmp_path):
    """Even if a prompts.example/<name>.txt sits in the repo, the loader must NOT use it."""
    fake_repo_root = prompt_loader._REPO_ROOT
    _write(
        fake_repo_root / "config" / "prompts.example" / "ghost.txt",
        "# version: 9.9.9\nThis must never load.\n",
    )

    with pytest.raises(FileNotFoundError):
        load_prompt("ghost")


def test_lambda_image_path_resolves(tmp_path):
    """deploy.sh stages the config repo into <repo>/config/prompts/ for the Lambda image."""
    fake_repo_root = prompt_loader._REPO_ROOT
    _write(
        fake_repo_root / "config" / "prompts" / "lambda_only.txt",
        "# version: 1.0.0\nLambda staged prompt.\n",
    )
    loaded = load_prompt("lambda_only")
    assert loaded.version == "1.0.0"
    assert "lambda staged" in loaded.text.lower()


def test_search_order_local_dev_wins_over_lambda_image(tmp_path):
    """When both ~/alpha-engine-config and <repo>/config/prompts/ have a prompt, sibling clone wins."""
    home = Path.home()
    fake_repo_root = prompt_loader._REPO_ROOT
    _write(
        home / "alpha-engine-config" / "research" / "prompts" / "dual.txt",
        "# version: 2.0.0\nFrom sibling clone.\n",
    )
    _write(
        fake_repo_root / "config" / "prompts" / "dual.txt",
        "# version: 1.0.0\nFrom Lambda staging.\n",
    )
    loaded = load_prompt("dual")
    assert loaded.version == "2.0.0", (
        "Local-dev sibling clone must take precedence over the Lambda-staged "
        "fallback so iterating on prompts doesn't require a deploy."
    )


def test_github_workspace_path_resolves(tmp_path, monkeypatch):
    ws = tmp_path / "ci_ws"
    monkeypatch.setenv("GITHUB_WORKSPACE", str(ws))
    _write(
        ws / "alpha-engine-config" / "research" / "prompts" / "ci.txt",
        "# version: 1.1.0\nCI-staged prompt.\n",
    )
    loaded = load_prompt("ci")
    assert loaded.version == "1.1.0"


def test_cache_returns_same_object_on_second_load(tmp_path):
    home = Path.home()
    _write(
        home / "alpha-engine-config" / "research" / "prompts" / "cached.txt",
        "# version: 1.0.0\nCacheable.\n",
    )
    a = load_prompt("cached")
    b = load_prompt("cached")
    assert a is b


def test_clear_cache_drops_cached_prompts(tmp_path):
    home = Path.home()
    target = home / "alpha-engine-config" / "research" / "prompts" / "evict.txt"
    _write(target, "# version: 1.0.0\nFirst body.\n")
    first = load_prompt("evict")

    _write(target, "# version: 2.0.0\nSecond body.\n")
    cached = load_prompt("evict")
    assert cached.text == first.text, "Without clearing cache, file edits are invisible."

    prompt_loader.clear_cache()
    fresh = load_prompt("evict")
    assert fresh.version == "2.0.0"
    assert "Second body" in fresh.text


def test_loaded_prompt_is_frozen_dataclass(tmp_path):
    home = Path.home()
    _write(
        home / "alpha-engine-config" / "research" / "prompts" / "frozen.txt",
        "# version: 1.0.0\nbody\n",
    )
    loaded = load_prompt("frozen")

    with pytest.raises(FrozenInstanceError):
        loaded.version = "9.9.9"  # type: ignore[misc]
