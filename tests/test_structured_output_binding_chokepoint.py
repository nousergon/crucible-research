"""Every structured-output binding in ``agents/`` goes through
``agents.langchain_utils.bind_structured_output``.

WHY THIS IS A STATIC GUARD AND NOT A UNIT TEST. The method a binding uses is
invisible to a mocked test — a fake ``with_structured_output`` accepts any
keyword — so nothing in the suite can tell ``json_schema`` from
``function_calling``. The difference only appears against a live provider, and
by then it is a failed weekly run.

``langchain_openai>=1.0`` changed the ``with_structured_output`` default from
``function_calling`` to ``json_schema``, which sends an OpenAI
``response_format``. Under ``ChatAnthropic`` that default never applied. The
moment these agents resolved through the router to DeepSeek
(alpha-engine-config-I7005 / I7448) every extraction returned::

    litellm.BadRequestError: OpenAIException -
    This response_format type is unavailable now.
    Received Model Group=high-deepseek-v4-pro-max

Measured 2026-08-16 on the canary-replay box, marker
``pr-nousergon-crucible-research-606-f1f856480de2``. Tool calling is supported
by every provider the registry can resolve; ``json_schema`` is an OpenAI-family
capability. Which one this repo depends on is a portability decision
(``model-portability-policy``), so it is made once, in one function.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTS_DIR = REPO_ROOT / "agents"

#: The one module allowed to call the LangChain API directly — it IS the
#: chokepoint.
_CHOKEPOINT = AGENTS_DIR / "langchain_utils.py"


def _agent_modules() -> list[Path]:
    return sorted(p for p in AGENTS_DIR.rglob("*.py") if p != _CHOKEPOINT)


def _direct_binding_calls(path: Path) -> list[int]:
    """Line numbers of ``<anything>.with_structured_output(...)`` calls.

    AST rather than grep so a mention in a docstring or a comment — of which
    this repo has many, describing the very migration that motivated the
    guard — is not a finding.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    hits: list[int] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "with_structured_output"
        ):
            hits.append(node.lineno)
    return hits


@pytest.mark.parametrize("module", _agent_modules(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_agent_module_binds_through_the_chokepoint(module: Path):
    hits = _direct_binding_calls(module)
    assert not hits, (
        f"{module.relative_to(REPO_ROOT)} calls .with_structured_output() directly at "
        f"line(s) {hits}. Use agents.langchain_utils.bind_structured_output(llm, Schema, "
        f"include_raw=...) instead: it pins method='function_calling', which every "
        f"provider the registry can resolve supports. The langchain_openai default is "
        f"'json_schema', which DeepSeek rejects with 'This response_format type is "
        f"unavailable now' — a failure no mocked test can see."
    )


def test_the_chokepoint_pins_function_calling():
    """The guard above is only worth having if the chokepoint still does its job."""
    source = _CHOKEPOINT.read_text()
    tree = ast.parse(source, filename=str(_CHOKEPOINT))

    fn = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "bind_structured_output"
        ),
        None,
    )
    assert fn is not None, "bind_structured_output disappeared from langchain_utils"

    call = next(
        (
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "with_structured_output"
        ),
        None,
    )
    assert call is not None, "bind_structured_output no longer binds a schema"

    method = next(
        (kw.value for kw in call.keywords if kw.arg == "method"), None
    )
    assert isinstance(method, ast.Constant) and method.value == "function_calling", (
        "bind_structured_output must pass method='function_calling' explicitly. "
        "Falling back to the langchain_openai default ('json_schema') reintroduces "
        "the provider-capability dependency this function exists to remove."
    )
