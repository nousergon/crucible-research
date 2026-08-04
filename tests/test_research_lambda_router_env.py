"""The research Lambda's router addressing is codified in deploy.sh.

Brian's ruling 2026-08-03 (alpha-engine-config-I6367): no agent may be
directly linked to OpenRouter. `producers/single_agent.py` addresses the
`high` model group, which needs six variables on the live function. These
tests pin the two properties that would silently break it.
"""

from __future__ import annotations

import pathlib
import re

DEPLOY = pathlib.Path(__file__).parent.parent / "infrastructure" / "deploy.sh"


def _text() -> str:
    return DEPLOY.read_text()


def test_every_router_variable_is_set():
    text = _text()
    for var in (
        "KREPIS_EXEC_CONTEXT",
        "KREPIS_LITELLM_PROXY_URL",
        "KREPIS_ROUTER_CREDENTIAL_SECRET",
        "KREPIS_APPCONFIG_APPLICATION",
        "KREPIS_APPCONFIG_CONFIG_PROFILE",
        "KREPIS_APPCONFIG_ENVIRONMENT",
    ):
        assert f'"{var}"' in text, (
            f"{var} is never set on the function. krepis' AppConfig path is "
            "opt-in on the application id and swallows its own errors, so a "
            "missing one surfaces later as 'LLM_MODEL_REGISTRY.yaml not "
            "found' — naming neither AppConfig nor the cause."
        )


def test_the_helper_is_actually_called_for_the_main_function():
    """A helper nothing invokes is the same as no helper, and this one would
    fail silently: the Lambda keeps its old environment and every routed call
    raises at resolve time."""
    text = _text()
    assert re.search(r'^\s*_apply_router_env "\$FUNCTION_MAIN"', text, re.M), (
        "_apply_router_env is defined but never called for $FUNCTION_MAIN"
    )


def test_the_environment_is_merged_not_replaced():
    """`update-function-configuration --environment` REPLACES the whole
    variable map. The function's other ~20 variables — provider keys,
    RAG_DATABASE_URL, LangSmith config — are codified NOWHERE in this repo and
    exist only on the live function, so writing a fresh map would delete every
    one of them. The read half is what makes this safe."""
    text = _text()
    assert "get-function-configuration" in text and "Environment.Variables" in text, (
        "the current environment is never read back, so the update would "
        "replace ~20 uncodified live variables rather than merge into them"
    )


def test_credential_secret_is_the_lambdas_own():
    """The edge identifies a consumer BY its credential VALUE, and
    krepis.secrets resolves SSM BEFORE os.environ — so naming the shared
    LITELLM_MASTER_KEY here would collapse this Lambda into the director's
    identity at the edge however the environment is set."""
    text = _text()
    assert "ROUTER_CONSUMER_RESEARCH" in text
    assert "KREPIS_ROUTER_CREDENTIAL_SECRET\": \"LITELLM_MASTER_KEY\"" not in text


def test_exec_context_is_lambda():
    """It names WHERE CODE RUNS (model-router-policy R28). The registry
    declares `lambda` on no model entry, deliberately — that is what makes
    this call site fail CLOSED instead of reaching a provider endpoint whose
    traffic is unscanned (alpha-engine-config-I6183)."""
    assert '"KREPIS_EXEC_CONTEXT": "lambda"' in _text()
