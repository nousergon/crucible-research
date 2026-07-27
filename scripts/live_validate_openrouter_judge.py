"""Live validation of the OpenRouter judge tier + leak guard (config#2575).

Ad-hoc operator tool — NOT part of the pytest suite (deliberately: it
makes REAL, billed OpenRouter calls). Companion to
``tests/live_smoke/judge_perturbation_smoke_openrouter.py`` (which
validates the JUDGE's sensitivity via the perturbation battery); this
script instead validates the TRANSPORT/leak-guard plumbing
(``krepis.judge.check_openai_tool_response_for_leak`` +
``evals.judge.evaluate_artifact_openrouter``) against real API responses,
including a deliberately tight-token-budget probe that reproduces the
live truncation-before-tool-call failure shape the guard exists to catch
(config#2575 item 3's "validated against a REAL OpenRouter call, not a
mock" requirement).

Usage from repo root (requires a real OPENROUTER_API_KEY — SSM
``/alpha-engine/OPENROUTER_API_KEY`` or the env var directly):
    python scripts/live_validate_openrouter_judge.py

Prints three cases and their outcome; exits non-zero if any case
produces an UNEXPECTED result (a clean-path case raising, or the
truncation probe NOT raising — the latter is not itself a bug, since
LLM output is stochastic, but is printed loudly for operator review).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from krepis.judge import (  # noqa: E402
    JudgeToolCallLeakError,
    check_openai_tool_response_for_leak,
)

from config import OPENROUTER_API_KEY  # noqa: E402
from evals.judge_models import OPENROUTER_SHADOW  # noqa: E402

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
_TOOL = {
    "type": "function",
    "function": {
        "name": "RubricEvalLLMOutput",
        "description": "Emit the rubric eval.",
        "parameters": _TOOL_SCHEMA,
    },
}
_MESSAGES = [
    {"role": "system", "content": "You are a strict rubric judge."},
    {
        "role": "user",
        "content": (
            "Rubric dimension: clarity (1-5). Output: 'The stock went up "
            "because reasons.' Score it. Call the tool."
        ),
    },
]


def main() -> int:
    if not OPENROUTER_API_KEY:
        print(
            "live_validate_openrouter_judge: no OpenRouter API key resolved "
            "(SSM /alpha-engine/OPENROUTER_API_KEY or OPENROUTER_API_KEY env "
            "var) — nothing to validate against.",
            file=sys.stderr,
        )
        return 1

    from openai import OpenAI

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    exit_code = 0

    print(f"=== Case 1: clean structured call ({OPENROUTER_SHADOW.request_model}) — expect PASS ===")
    resp = client.chat.completions.create(
        model=OPENROUTER_SHADOW.request_model,
        messages=_MESSAGES,
        tools=[_TOOL],
        tool_choice={"type": "function", "function": {"name": "RubricEvalLLMOutput"}},
        max_tokens=1024,
    )
    choice = resp.choices[0]
    print(f"finish_reason={choice.finish_reason!r} tool_calls={bool(choice.message.tool_calls)}")
    try:
        check_openai_tool_response_for_leak(choice, tool_name="RubricEvalLLMOutput")
        print("RESULT: no leak raised (correct)\n")
    except JudgeToolCallLeakError as e:
        print(f"RESULT: UNEXPECTED RAISE: {e.reason} {e}\n")
        exit_code = 1

    print("=== Case 2: reasoning-truncation probe (moonshotai/kimi-k2.6, tight budget) — expect RAISE ===")
    resp2 = client.chat.completions.create(
        model="moonshotai/kimi-k2.6",
        messages=_MESSAGES,
        tools=[_TOOL],
        tool_choice={"type": "function", "function": {"name": "RubricEvalLLMOutput"}},
        max_tokens=200,  # deliberately tight — reproduces the truncation shape
    )
    choice2 = resp2.choices[0]
    print(
        f"finish_reason={choice2.finish_reason!r} "
        f"tool_calls={choice2.message.tool_calls!r} "
        f"content={choice2.message.content!r}"
    )
    try:
        check_openai_tool_response_for_leak(choice2, tool_name="RubricEvalLLMOutput")
        print(
            "RESULT: no leak raised (LLM output is stochastic — this run "
            "happened not to truncate; not itself a guard failure, but "
            "re-run if you need to re-observe the truncation shape)\n"
        )
    except JudgeToolCallLeakError as e:
        print(f"RESULT: RAISED AS EXPECTED: {e.reason} finish_reason={e.finish_reason}\n")

    print("=== Case 3: reasoning-excluded + adequate budget — expect PASS ===")
    resp3 = client.chat.completions.create(
        model="moonshotai/kimi-k2.6",
        messages=_MESSAGES,
        tools=[_TOOL],
        tool_choice={"type": "function", "function": {"name": "RubricEvalLLMOutput"}},
        max_tokens=800,
        extra_body={"reasoning": {"exclude": True}},
    )
    choice3 = resp3.choices[0]
    print(f"finish_reason={choice3.finish_reason!r} tool_calls={bool(choice3.message.tool_calls)}")
    try:
        check_openai_tool_response_for_leak(choice3, tool_name="RubricEvalLLMOutput")
        print("RESULT: no leak raised (correct)\n")
    except JudgeToolCallLeakError as e:
        print(f"RESULT: UNEXPECTED RAISE: {e.reason} {e}\n")
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
