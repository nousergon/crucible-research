"""
Shared JSON extraction utilities for LLM agent responses.

Uses balanced-brace scanning for robust extraction from mixed text/JSON
responses. Handles nested objects, escaped quotes, and malformed outputs.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


def extract_json_object(text: str, hint_key: str | None = None) -> dict | None:
    """
    Extract a JSON object from mixed text. Uses balanced-brace scanning.

    Args:
        text: Raw LLM response that may contain JSON mixed with prose.
        hint_key: Optional key to locate the target object (e.g., '"market_regime"').
                  If provided, finds the object containing this key.
                  If not provided, finds the first top-level object.

    Returns:
        Parsed dict, or None if extraction fails.
    """
    if hint_key:
        start, end = _find_json_block(text, hint_key)
    else:
        start, end = _find_first_block(text, "{", "}")

    if start < 0:
        return None

    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def extract_json_array(text: str, require_key: str | None = None) -> list | None:
    """
    Extract a JSON array from mixed text.

    Falls back to balanced-brace scanning for top-level objects when array
    extraction fails. If `require_key` is set, only objects containing that
    key at the top level are kept — prevents the scanner from returning
    nested sub-objects (e.g., {"reason": "..."} from inside a pick) when
    the outer array is malformed.

    Returns:
        Parsed list, or None if extraction fails.
    """
    start, end = _find_first_block(text, "[", "]")
    if start >= 0:
        try:
            result = json.loads(text[start:end + 1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Fallback: balanced-brace scan for top-level objects.
    # The previous regex r'\{[^{}]+\}' matched innermost objects, which
    # meant nested sub-objects (e.g., {"reason": "strong"}) were returned
    # instead of the outer picks when the array syntax was malformed —
    # silently producing empty picks downstream.
    objects = []
    i = 0
    while i < len(text):
        brace = text.find("{", i)
        if brace == -1:
            break
        obj_start, obj_end = _scan_balanced(text, brace, "{", "}")
        if obj_start < 0:
            break
        try:
            obj = json.loads(text[obj_start:obj_end + 1])
            if isinstance(obj, dict) and (require_key is None or require_key in obj):
                objects.append(obj)
        except json.JSONDecodeError:
            pass
        i = obj_end + 1

    return objects if objects else None


def _find_json_block(text: str, key: str) -> tuple[int, int]:
    """
    Find the JSON object containing `key` using balanced-brace scanning.
    Returns (start, end) inclusive, or (-1, -1) if not found.
    """
    key_pos = text.find(key)
    if key_pos == -1:
        return -1, -1
    brace_pos = text.rfind('{', 0, key_pos)
    if brace_pos == -1:
        return -1, -1
    return _scan_balanced(text, brace_pos, '{', '}')


def _find_first_block(text: str, open_ch: str, close_ch: str) -> tuple[int, int]:
    """Find the first balanced block of the given bracket type."""
    start = text.find(open_ch)
    if start == -1:
        return -1, -1
    return _scan_balanced(text, start, open_ch, close_ch)


def _scan_balanced(text: str, start: int, open_ch: str, close_ch: str) -> tuple[int, int]:
    """Scan for balanced brackets starting at `start`, handling strings and escapes."""
    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return start, i
    return -1, -1
