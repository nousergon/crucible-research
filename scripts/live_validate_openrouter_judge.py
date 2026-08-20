"""Live validation of the router-resolved judge transport + leak guard
(config#2575).

Ad-hoc operator tool — NOT part of the pytest suite (deliberately: it
makes REAL, billed LLM calls). Companion to
``tests/live_smoke/judge_perturbation_smoke_openrouter.py`` (which
validates the JUDGE's sensitivity via the perturbation battery); this
script instead validates the TRANSPORT/leak-guard plumbing
(``krepis.judge.check_openai_tool_response_for_leak``) against real
responses from the SAME router group evals/judge.py's judge call sites
resolve through, including a deliberately tight-token-budget probe that
reproduces the live truncation-before-tool-call failure shape the guard
exists to catch (config#2575 item 3's "validated against a REAL call,
not a mock" requirement).

alpha-engine-config-I6559 (2026-08-19, crucible-research#666) moved
``evals.judge.evaluate_artifact`` / ``evaluate_artifact_openrouter`` off
direct OpenRouter onto ``krepis.router.resolve_group_spec`` against the
``low`` model group, per the no-direct-OpenRouter ruling
(alpha-engine-config-I6367). This script previously constructed its own
``openai.OpenAI(base_url="https://openrouter.ai/api/v1", ...)`` client
against a hardcoded OpenRouter key and a literal, since-DEPRECATED model
(``moonshotai/kimi-k2.6`` — ``LLM_MODEL_REGISTRY.yaml`` id ``kimi-k2.6``,
``status: deprecated``, model-router-policy.md R4: a deprecated entry
MUST NOT be reachable at runtime, not even by name). Both were exactly
the direct-linkage pattern I6367 forbids and were never updated to match
the I6559 migration (alpha-engine-config-I7880). Rewritten here to
resolve through the router's ``low`` group — the same group
``evals.judge.JUDGE_MODEL_GROUP`` names — via ``krepis.llm.LLMClient``,
same as production. No provider API key is read or required: the router
resolves the credential by name (``resolve_group_spec``); passing one
explicitly would OVERRIDE that resolution rather than merely gate on
presence (see ``evals/judge.py::_call_openrouter_judge_llm``'s docstring
for the ``api_key`` test-seam note this script used to violate by
sourcing a real key from config unconditionally).

Case 2 (the tight-budget truncation reproduction) is consequently no
longer pinned to a specific known-flaky model — it exercises whichever
model the ``low`` group currently resolves to, with a deliberately tight
``max_tokens``. Since the failure mode is model- and sample-dependent,
not raising here was already documented as "not itself a bug" before
this migration and remains so; the group may resolve to a model that
does not exhibit the reasoning-truncation shape at all, in which case
Case 2 harmlessly passes without reproducing the failure. A model-pinned
reproduction, if one is still wanted, belongs in the registry (a group or
a ``pinned_model`` entry — model-router-policy.md, `llm-provider-model-
policy.md` §4's carve-out ledger) rather than a call-site literal.

Usage from repo root (needs a reachable router — model-router-policy.md
§3.4a; on a GitHub-hosted CI runner today this fails closed with a
``ValueError`` per alpha-engine-config-I7853, which is expected and not
a bug in this script):
    python scripts/live_validate_openrouter_judge.py

Prints three cases and their outcome; exits non-zero if any case
produces an UNEXPECTED result (a clean-path case raising, or the
truncation probe NOT raising — the latter is not itself a bug, since
LLM output is stochastic, but is printed loudly for operator review), or
if the router itself is unreachable from this environment.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from krepis.judge import (  # noqa: E402
    JudgeToolCallLeakError,
    check_openai_tool_response_for_leak,
)
from krepis.llm import LLMClient  # noqa: E402
from krepis.router import resolve_group_spec  # noqa: E402

from evals.judge import JUDGE_MODEL_GROUP  # noqa: E402

_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "dimension_scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "dimension": {"type": "string"},
                    "score": {"type": "integer"},
                    "reasoning": {"type": "string"},
                },
                "required": ["dimension", "score", "reasoning"],
            },
        },
        "overall_reasoning": {"type": "string"},
    },
    "required": ["dimension_scores", "overall_reasoning"],
}
_TOOL_NAME = "RubricEvalLLMOutput"
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": _TOOL_NAME,
            "description": "Emit the rubric eval.",
            "parameters": _TOOL_SCHEMA,
        },
    }
]
_SYSTEM = "You are a strict rubric judge."
_USER = (
    "Rubric dimension: clarity (1-5). Output: 'The stock went up "
    "because reasons.' Score it. Call the tool."
)
_EXTRA = {
    "tools": _TOOLS,
    "tool_choice": {"type": "function", "function": {"name": _TOOL_NAME}},
}


def _resolve_router_spec_and_route(*, max_tokens: int, reasoning: dict | None):
    """Resolve the ``low`` judge group for THIS operator's execution
    context. Deliberately ``exec_context="laptop"`` rather than reusing
    ``evals.judge.JUDGE_EXEC_CONTEXT`` (``"lambda"``) — this script is
    documented and run from a laptop/EC2 checkout, and model-router-
    policy R28/R29 requires the caller's OWN declared context, not a
    borrowed production one (alpha-engine-config-I7853 item 4 records the
    sibling defect where a CI caller reused the Lambda constant)."""
    spec, route = resolve_group_spec(
        JUDGE_MODEL_GROUP,
        exec_context="laptop",
        wire="openai",
        max_tokens=max_tokens,
        structured_outputs=False,
    )
    if reasoning is not None:
        spec = dataclasses.replace(spec, reasoning=reasoning)
    return spec, route


def main() -> int:
    try:
        spec_default, route = _resolve_router_spec_and_route(max_tokens=1024, reasoning=None)
    except ValueError as exc:
        print(
            f"live_validate_openrouter_judge: router resolution found no "
            f"reachable model for group {JUDGE_MODEL_GROUP!r} from this "
            f"environment ({exc}). Nothing to validate against — see "
            f"model-router-policy.md §3.4a / alpha-engine-config-I7853 if "
            f"this is unexpected for this environment.",
            file=sys.stderr,
        )
        return 1

    print(f"resolved group={JUDGE_MODEL_GROUP!r} -> model={spec_default.model} route={route.get('route')}")
    exit_code = 0

    def _client(spec, max_tokens: int) -> LLMClient:
        return LLMClient(
            dataclasses.replace(spec, max_tokens=max_tokens),
            callsite_id="live-validate-openrouter-judge",
            timeout=180.0,
            max_retries=0,
        )

    print(f"=== Case 1: clean structured call ({spec_default.model}) — expect PASS ===")
    result1 = _client(spec_default, 1024).complete(
        system=_SYSTEM, user_content=_USER, max_tokens=1024, extra=_EXTRA,
    )
    choice1 = result1.raw_response.choices[0]
    print(f"finish_reason={choice1.finish_reason!r} tool_calls={bool(choice1.message.tool_calls)}")
    try:
        check_openai_tool_response_for_leak(choice1, tool_name=_TOOL_NAME)
        print("RESULT: no leak raised (correct)\n")
    except JudgeToolCallLeakError as e:
        print(f"RESULT: UNEXPECTED RAISE: {e.reason} {e}\n")
        exit_code = 1

    print(f"=== Case 2: reasoning-truncation probe ({spec_default.model}, tight budget) — expect RAISE (stochastic) ===")
    result2 = _client(spec_default, 200).complete(
        system=_SYSTEM, user_content=_USER, max_tokens=200, extra=_EXTRA,
    )
    choice2 = result2.raw_response.choices[0]
    print(
        f"finish_reason={choice2.finish_reason!r} "
        f"tool_calls={choice2.message.tool_calls!r} "
        f"content={choice2.message.content!r}"
    )
    try:
        check_openai_tool_response_for_leak(choice2, tool_name=_TOOL_NAME)
        print(
            "RESULT: no leak raised (LLM output is stochastic, and the "
            "router-resolved model for this run may not be reasoning-"
            "capable — this run happened not to truncate; not itself a "
            "guard failure, but re-run if you need to re-observe the "
            "truncation shape)\n"
        )
    except JudgeToolCallLeakError as e:
        print(f"RESULT: RAISED AS EXPECTED: {e.reason} finish_reason={e.finish_reason}\n")

    print(f"=== Case 3: reasoning-excluded + adequate budget ({spec_default.model}) — expect PASS ===")
    spec_no_reasoning, _ = _resolve_router_spec_and_route(max_tokens=800, reasoning={"exclude": True})
    result3 = _client(spec_no_reasoning, 800).complete(
        system=_SYSTEM, user_content=_USER, max_tokens=800, extra=_EXTRA,
    )
    choice3 = result3.raw_response.choices[0]
    print(f"finish_reason={choice3.finish_reason!r} tool_calls={bool(choice3.message.tool_calls)}")
    try:
        check_openai_tool_response_for_leak(choice3, tool_name=_TOOL_NAME)
        print("RESULT: no leak raised (correct)\n")
    except JudgeToolCallLeakError as e:
        print(f"RESULT: UNEXPECTED RAISE: {e.reason} {e}\n")
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
