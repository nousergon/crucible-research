"""Citation checks for Think Tank thesis and pillar outputs.

alpha-engine-config-I12312. Every thesis and pillar input is rendered into the
prompt as a ``<source cid="…">`` block (:mod:`thinktank.sources`). The cited
prompt versions (``thinktank_thesis_cited``, ``thinktank_pillar_cited``) carry
``krepis.llm_trace.CITATION_INSTRUCTION`` and ask the model for two things:

- inline ``[cid]`` / ``[cid, cid]`` markers in its prose, which
  ``krepis.llm_trace`` reads to set ``cited_in_output`` on each source of the
  call's trace row; and
- a structured ``claims`` list (:class:`thinktank.schemas.CitedClaim`), each
  key claim with the cids it relies on.

:func:`check_citations` validates that output against the cids actually
rendered into THAT prompt. It is deterministic and never raises on what the
model wrote: an unknown cid is recorded as invalid (never dropped), a claim
with no valid citation is flagged, and the run carries on.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from krepis.llm_trace import CITATION_INSTRUCTION, RenderedSources

from thinktank.schemas import CheckedClaim, CitationCheck, CitedClaim

logger = logging.getLogger(__name__)

# The inline marker grammar of ``krepis.llm_trace.cited_cids``, verbatim.
# tests/test_thinktank_citations.py pins it to krepis's own pattern, so the
# check here and the trace row's ``cited_in_output`` read the same markers.
CITE_MARKER_RE = re.compile(
    r"\[\s*([A-Za-z0-9_.\-]{1,64}(?:\s*,\s*[A-Za-z0-9_.\-]{1,64})*)\s*\]"
)
_STRIP_MARKER_RE = re.compile(r"[ \t]*" + CITE_MARKER_RE.pattern)


def require_citation_instruction(prompt: Any) -> None:
    """Raise before any spend when a cited prompt lacks the krepis instruction.

    The instruction's wording is what tells the model the marker grammar the
    trace parser reads. A prompt that drifted from it would produce markers
    nothing counts, and every source would read ``cited_in_output: false``
    with no error anywhere.
    """
    text = getattr(prompt, "text", "")
    if CITATION_INSTRUCTION not in text:
        raise ValueError(
            f"prompt {getattr(prompt, 'name', '?')} v{getattr(prompt, 'version', '?')} "
            f"does not carry krepis.llm_trace.CITATION_INSTRUCTION verbatim; its "
            f"citations would not be read. Fix the prompt in alpha-engine-config."
        )


def marker_tokens(text: str) -> list[str]:
    """Every token inside an inline ``[a, b]`` marker, first occurrence first."""
    out: list[str] = []
    for match in CITE_MARKER_RE.finditer(text or ""):
        for token in match.group(1).split(","):
            token = token.strip()
            if token and token not in out:
                out.append(token)
    return out


def strip_citation_markers(value: Any) -> Any:
    """Remove inline ``[cid]`` markers from every string in ``value``.

    For re-rendering a stored output into a later prompt (the prior thesis):
    yesterday's ``[f1]`` named yesterday's first filings chunk, and today's
    ``f1`` can be a different document.
    """
    if isinstance(value, str):
        return _STRIP_MARKER_RE.sub("", value)
    if isinstance(value, Mapping):
        return {k: strip_citation_markers(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [strip_citation_markers(v) for v in value]
    return value


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v)


def check_citations(
    output: Mapping[str, Any],
    claims: Sequence[CitedClaim],
    rendered: RenderedSources | None,
    *,
    prompt_id: str = "",
    prompt_version: str = "",
) -> CitationCheck:
    """Validate an output's citations against the cids rendered into its prompt.

    Args:
        output: The call's structured output as a plain mapping (its
            ``model_dump()``), ``claims`` included or not. Every string in it
            is scanned for inline markers.
        claims: The output's ``claims`` list.
        rendered: The ``RenderedSources`` the prompt was sent with.
        prompt_id, prompt_version: Recorded on the result.

    Never raises on model output and never drops anything the model cited.
    """
    rendered_cids = [s.block.cid for s in (rendered.sources if rendered else ())]
    known = set(rendered_cids)

    cited: list[str] = []
    unresolved: list[str] = []

    def _cite(cid: str) -> None:
        if cid not in cited:
            cited.append(cid)

    for text in _strings(output):
        for token in marker_tokens(text):
            if token in known:
                _cite(token)
            elif token not in unresolved:
                unresolved.append(token)

    checked: list[CheckedClaim] = []
    invalid: list[str] = []
    for claim in claims:
        valid: list[str] = []
        bad: list[str] = []
        for raw in claim.citations:
            cid = raw.strip() if isinstance(raw, str) else raw
            if cid in known:
                if cid not in valid:
                    valid.append(cid)
            elif raw not in bad:
                bad.append(raw)
        for token in marker_tokens(claim.claim):
            if token in known and token not in valid:
                valid.append(token)
        for cid in valid:
            _cite(cid)
        for cid in bad:
            if cid not in invalid:
                invalid.append(cid)
        checked.append(
            CheckedClaim(
                section=claim.section,
                claim=claim.claim,
                citations=valid,
                invalid_citations=bad,
                uncited=not valid,
            )
        )

    ordered_cited = [c for c in rendered_cids if c in cited]
    return CitationCheck(
        prompt_id=prompt_id,
        prompt_version=prompt_version,
        rendered_cids=rendered_cids,
        cited_cids=ordered_cited,
        uncited_sources=[c for c in rendered_cids if c not in cited],
        invalid_cids=invalid,
        unresolved_markers=unresolved,
        claim_count=len(checked),
        uncited_claim_count=sum(1 for c in checked if c.uncited),
        claims=checked,
    )


def log_citation_check(check: CitationCheck, *, ticker: str, call: str) -> None:
    """One INFO line per call; WARNING when something needs a human's eye."""
    summary = (
        "%s %s citations: %d/%d sources cited, %d claim(s), %d uncited, "
        "invalid=%s unresolved=%s"
    )
    args = (
        ticker,
        call,
        len(check.cited_cids),
        len(check.rendered_cids),
        check.claim_count,
        check.uncited_claim_count,
        json.dumps(check.invalid_cids),
        json.dumps(check.unresolved_markers),
    )
    if check.invalid_cids or check.uncited_claim_count or not check.claim_count:
        logger.warning(summary, *args)
    else:
        logger.info(summary, *args)
