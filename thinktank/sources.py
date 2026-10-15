"""Prompt sources — what a Think Tank prompt cites, with stable ids.

Every input a thesis or pillar prompt is built from (a filings chunk, the
universe-board row, the weekly signal, a substrate table row) is rendered
into the prompt as a ``krepis.llm_trace`` source block::

    <source cid="news">
    {"ticker": "QLYS", ...}
    </source>

and handed to the LLM call as ``sources=``, so the call's ``llm_trace.v1``
record carries each source's stable id (chunk id, document id, S3 key and
VersionId), the sha256 of the exact text injected and its character offsets
in the prompt (alpha-engine-config-I12312, audit-2026-10-14 §6.3).

The offsets are computed while rendering, never inferred from the model's
prose: each slot is rendered with :func:`krepis.llm_trace.render_sources`,
the prompt template is formatted, and each rendered section is located in
the result exactly once (the same rule ``krepis.llm_trace.bind_sources``
applies). The template text itself is unchanged; the only thing the model
sees differently is the ``<source cid="…">`` delimiter around each input.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from krepis.llm_trace import RenderedSource, RenderedSources, render_sources

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Slot:
    """A template field whose value is one or more cited sources.

    Args:
        sources: Source mappings (``SourceBlock.from_mapping`` shape). Empty
            means the input was absent.
        empty: What the slot renders as when ``sources`` is empty — the same
            placeholder text the prompt used before sources existed.
        separator: Between consecutive source blocks in this slot.
    """

    sources: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    empty: str = ""
    separator: str = "\n\n"


def s3_object_source(
    cid: str,
    *,
    bucket: str,
    key: str,
    text: str,
    version: str | None = None,
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One S3-backed context input as a ``kind: "s3_object"`` source.

    ``key`` is the object (or, for a partitioned table, the table prefix the
    row was read from); ``version`` is its S3 VersionId when the read
    recorded one, or the row's own as-of stamp for a table row. ``text`` is
    exactly what the prompt injects.
    """
    return {
        "cid": cid,
        "kind": "s3_object",
        "source_id": f"s3://{bucket}/{key}",
        "text": text,
        "uri": f"s3://{bucket}/{key}",
        "version": version,
        "meta": dict(meta) if meta else None,
    }


def format_with_sources(prompt: Any, **fields: Any) -> tuple[str, RenderedSources]:
    """Format ``prompt`` with ``fields``, rendering every :class:`Slot` as
    source blocks. Returns ``(user_content, rendered_sources)``.

    ``rendered_sources.text`` is the whole formatted prompt, so passing the
    pair to ``LLMClient.structured(user_content=..., sources=...)`` binds
    every offset at base 0 — the prompt is sent exactly as returned.

    A slot rendered into the template more than once raises ``ValueError``
    (its offsets would be ambiguous). A slot the template does not use is
    dropped from the trace with a WARNING: a source that was never sent was
    never cited.
    """
    plain: dict[str, Any] = {}
    rendered: dict[str, RenderedSources] = {}
    for name, value in fields.items():
        if isinstance(value, Slot):
            if value.sources:
                section = render_sources(value.sources, separator=value.separator)
                rendered[name] = section
                plain[name] = section.text
            else:
                plain[name] = value.empty
        else:
            plain[name] = value
    user = prompt.format(**plain)

    located: list[RenderedSource] = []
    for name, section in rendered.items():
        count = user.count(section.text)
        if count == 0:
            logger.warning(
                "prompt %s: template has no {%s} field — its %d source(s) "
                "were not sent and are not traced",
                getattr(prompt, "name", "?"), name, len(section.sources),
            )
            continue
        if count > 1:
            raise ValueError(
                f"prompt {getattr(prompt, 'name', '?')}: the {{{name}}} source "
                f"section occurs {count} times in the rendered prompt; its "
                f"citation offsets would be ambiguous"
            )
        base = user.index(section.text)
        located.extend(
            RenderedSource(
                block=s.block,
                content_sha256=s.content_sha256,
                char_start=base + s.char_start,
                char_end=base + s.char_end,
            )
            for s in section.sources
        )
    cids = [s.block.cid for s in located]
    if len(cids) != len(set(cids)):
        raise ValueError(f"duplicate source cid across prompt slots: {sorted(cids)}")
    located.sort(key=lambda s: s.char_start)
    return user, RenderedSources(text=user, sources=tuple(located))
