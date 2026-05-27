"""Live-LangChain-Anthropic smoke — catches Anthropic API payload-shape
regressions in the LangChain ``ChatAnthropic`` path that mocked unit
tests miss by design.

Every research agent (macro, sector_teams quant/qual/peer_review,
investment_committee/ic_cio) uses ``langchain_anthropic.ChatAnthropic``
to reach the Anthropic API. The unit-test suite ``patch.object(...,
"ChatAnthropic", return_value=fake_llm)`` so tests run offline — but
that means the suite NEVER exercises the real API contract. Payload-
shape drift (model name renames, parameter deprecations, structured-
output schema changes) is invisible to CI until production fires.

The 2026-05-26 morning-signal incident (raw-Anthropic-SDK payload
shape drift via assistant-prefill + server-tool incompatibility) is
the immediate precedent. The LangChain wrapper adds its own
serialization layer on top of the SDK; this smoke exercises BOTH
contracts (LangChain → SDK → API) end-to-end.

Designed to run:

  * In CI on PRs touching ``agents/**/*.py`` — gated on the
    ``ANTHROPIC_API_KEY`` secret. Forks without the secret get a clean
    skip, not a CI failure.
  * Locally via ``.venv/bin/python tests/live_smoke/langchain_anthropic_smoke.py``.

Stays out of pytest's default collection because the file lives under
``tests/live_smoke/`` and the filename doesn't match ``test_*.py``.

Composes with morning-signal #34 (raw-SDK chokepoint) +
alpha-engine-lib #78 (``alpha_engine_lib.anthropic_payload``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make repo importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Force env-only secret resolution — CI does not assume an SSM-capable
# IAM role for this workflow.
os.environ.setdefault("ALPHA_ENGINE_SECRETS_SOURCE", "env")

SMOKE_MODEL = os.environ.get(
    "RESEARCH_LIVE_SMOKE_MODEL", "claude-haiku-4-5"
)
# Generic CI-only message — does NOT use any production prompt
# template (those are gitignored per CLAUDE.md "All agents files with
# prompts MUST be gitignored"). The smoke validates the
# LangChain → SDK → API contract, not a specific agent's prompt.
SMOKE_SYSTEM_PROMPT = (
    "You are running a CI smoke check. Respond with exactly one word."
)
SMOKE_USER_PROMPT = "Reply with 'ok' and nothing else."


def main() -> int:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "langchain_anthropic_smoke: ANTHROPIC_API_KEY not set; skipping. "
            "(Expected on fork PRs without the secret; not a failure.)",
            file=sys.stderr,
        )
        return 0

    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError as exc:
        print(
            f"langchain_anthropic_smoke: required dep missing — {exc}\n"
            "  langchain_anthropic + langchain_core must be installed via "
            "requirements.txt for this smoke to run.",
            file=sys.stderr,
        )
        return 1

    # Mirror the production ChatAnthropic construction (agents/macro_agent.py:
    # uses `anthropic_api_key=` not `api_key=`, max_tokens=, max_retries=).
    # max_tokens=1 caps the response to ~$0.001/run on Haiku.
    print(
        f"langchain_anthropic_smoke: dispatching ChatAnthropic({SMOKE_MODEL!r}, "
        f"max_tokens=1) ...",
        file=sys.stderr,
    )

    try:
        llm = ChatAnthropic(
            model=SMOKE_MODEL,
            anthropic_api_key=api_key,
            max_tokens=1,
            max_retries=0,
        )
        resp = llm.invoke(
            [
                SystemMessage(content=SMOKE_SYSTEM_PROMPT),
                HumanMessage(content=SMOKE_USER_PROMPT),
            ]
        )
    except Exception as exc:  # noqa: BLE001 - smoke surfaces everything
        cls = type(exc).__name__
        print(
            f"langchain_anthropic_smoke: FAILED — {cls}: {exc}\n"
            f"  This is exactly the regression class the smoke is meant to "
            f"catch (mocked tests would have passed). DO NOT MERGE.\n"
            f"  If this is HTTP 400, the LangChain → SDK payload shape "
            f"may have drifted vs the API contract. If HTTP 401/403, the "
            f"ANTHROPIC_API_KEY secret may be stale or wrong-account.",
            file=sys.stderr,
        )
        return 1

    content = getattr(resp, "content", None)
    if not content:
        print(
            f"langchain_anthropic_smoke: FAILED — response carried no "
            f"`content` attribute or content was empty; resp={resp!r}. "
            "LangChain response envelope may have drifted.",
            file=sys.stderr,
        )
        return 1

    print(
        f"langchain_anthropic_smoke: OK — model={SMOKE_MODEL} "
        f"content_len={len(content) if isinstance(content, str) else 'non-str'} "
        f"response_type={type(resp).__name__}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
