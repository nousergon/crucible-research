"""Judge evidence contract ``judge_evidence.v1`` (alpha-engine-config-I12312).

Ties every EvalJudge finding to the text it rests on, deterministically.

**Rendering.** Under the contract the judge never sees a JSON blob. Every
element of the judged artifact's input bundle is rendered under an explicit,
stable id, and every leaf of its output under its field path::

    <input id="in:board_row">
    {"pe": 28.4, ...}
    </input>

    <output field="thesis.risks[0]">
    Customer concentration: top-3 buyers are 41% of revenue.
    </output>

Input ids (see :func:`build_judge_inputs`):

* ``in:<key>`` — one top-level key of ``input_data_snapshot``;
* ``in:<key>[<i>]`` — one item of a top-level list (each filings excerpt is
  its own element, so a citation can name the excerpt, not "the filings");
* ``cid:<cid>`` — one ``<source cid="…">`` block found verbatim in the
  artifact's captured producer prompt, when the producer rendered its inputs
  that way (crucible-research#849). Absent today; read when present.

Empty inputs (``None``, ``""``, ``{}``, ``[]``) get no id — they are listed
as absent, so a citation can never rest on nothing.

Output text is rendered RAW (never JSON-escaped), which is what makes a
verbatim quote checkable: the quote the judge copies is the field's own
characters, and ``claim_span`` offsets index that field's string.

**Validation** (:func:`validate_dimension_scores`) is mechanical and never
asks a model: a quote must occur verbatim at its span, the span is
recomputed only when the quote occurs exactly once, and a ``source_ref``
must name an element the judge was actually given. Nothing is silently
repaired — the judge's own span is persisted beside the located one.

**Activation.** A rubric opts in by declaring the contract in its text
(``EVIDENCE CONTRACT: judge_evidence.v1``). It keeps the ``{agent_input}`` /
``{agent_output}`` placeholders, so the same template renders under code
that predates this module (the citations then go unvalidated, never
unrenderable) — the two repos can merge in either order. A rubric without
the declaration renders exactly as before, byte for byte.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from graph.state_schemas import (
    CitationCheck,
    CitedRubricDimensionScore,
    ClaimSpan,
    JudgeEvidence,
    JudgeInputRef,
    ValidatedEvidence,
)

CITATION_CONTRACT = "judge_evidence.v1"

_CONTRACT_RE = re.compile(r"EVIDENCE\s+CONTRACT\s*:\s*(judge_evidence\.v\d+)")

# A ``<source cid="…">`` block as krepis's SourceBlock renderer emits it into
# a producer prompt. Matched on the CAPTURED prompt text only — never on
# model output.
_SOURCE_BLOCK_RE = re.compile(
    r'<source\s+cid="(?P<cid>[^"]+)"[^>]*>\n?(?P<body>.*?)\n?</source>',
    re.DOTALL,
)


def declared_citation_contract(rubric_text: str) -> str | None:
    """The citation contract a rubric declares, or ``None`` (uncited)."""
    m = _CONTRACT_RE.search(rubric_text or "")
    return m.group(1) if m else None


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_empty(value: Any) -> bool:
    return value is None or value == "" or (isinstance(value, (dict, list, tuple)) and not value)


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, default=str, sort_keys=False)


@dataclass(frozen=True)
class JudgeInputElement:
    id: str
    text: str

    def ref(self) -> JudgeInputRef:
        return JudgeInputRef(id=self.id, sha256=_sha256(self.text), chars=len(self.text))


@dataclass(frozen=True)
class JudgeBundle:
    """Everything the judge is shown under the contract, by id/path."""

    inputs: tuple[JudgeInputElement, ...]
    absent_inputs: tuple[str, ...]
    outputs: dict[str, str]

    @property
    def input_ids(self) -> frozenset[str]:
        return frozenset(e.id for e in self.inputs)


def build_judge_inputs(
    input_data_snapshot: Mapping[str, Any] | None,
    *,
    producer_prompt: str | None = None,
) -> tuple[tuple[JudgeInputElement, ...], tuple[str, ...]]:
    """Enumerate the judge's input elements in snapshot order.

    Returns ``(elements, absent_keys)``. Deterministic: the same artifact
    always yields the same ids, texts and order.
    """
    elements: list[JudgeInputElement] = []
    absent: list[str] = []
    for key, value in (input_data_snapshot or {}).items():
        if _is_empty(value):
            absent.append(str(key))
            continue
        if isinstance(value, list):
            for i, item in enumerate(value):
                if _is_empty(item):
                    continue
                elements.append(JudgeInputElement(id=f"in:{key}[{i}]", text=_as_text(item)))
            continue
        elements.append(JudgeInputElement(id=f"in:{key}", text=_as_text(value)))

    seen = {e.id for e in elements}
    for m in _SOURCE_BLOCK_RE.finditer(producer_prompt or ""):
        eid = f"cid:{m.group('cid')}"
        if eid in seen:
            # Same cid twice in one prompt is the producer's ambiguity; the
            # first occurrence is the one offered, never a merge of both.
            continue
        seen.add(eid)
        elements.append(JudgeInputElement(id=eid, text=m.group("body")))
    return tuple(elements), tuple(absent)


def flatten_output(agent_output: Any, prefix: str = "") -> dict[str, str]:
    """Every leaf of ``agent_output`` keyed by field path, as the text the
    judge sees. Strings are verbatim; other scalars are their JSON text.
    Empty containers and ``None`` are omitted (nothing to quote)."""
    out: dict[str, str] = {}

    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for k, v in value.items():
                walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                walk(v, f"{path}[{i}]")
        elif value is None:
            return
        elif isinstance(value, str):
            if value:
                out[path] = value
        else:
            out[path] = json.dumps(value, default=str)

    walk(agent_output, prefix)
    return out


def build_bundle(artifact: Any) -> JudgeBundle:
    """The judge bundle for a ``DecisionArtifact`` (duck-typed)."""
    fpc = getattr(artifact, "full_prompt_context", None)
    producer_prompt = getattr(fpc, "user_prompt", None) if fpc is not None else None
    inputs, absent = build_judge_inputs(
        getattr(artifact, "input_data_snapshot", None) or {},
        producer_prompt=producer_prompt,
    )
    return JudgeBundle(
        inputs=inputs,
        absent_inputs=absent,
        outputs=flatten_output(getattr(artifact, "agent_output", None) or {}),
    )


def render_inputs(bundle: JudgeBundle) -> str:
    parts = [f'<input id="{e.id}">\n{e.text}\n</input>' for e in bundle.inputs]
    if not parts:
        parts.append("(no non-empty input elements)")
    if bundle.absent_inputs:
        parts.append("Absent inputs (empty — no id, cannot be cited): " + ", ".join(bundle.absent_inputs))
    return "\n\n".join(parts)


def render_outputs(bundle: JudgeBundle) -> str:
    parts = [f'<output field="{path}">\n{text}\n</output>' for path, text in bundle.outputs.items()]
    return "\n\n".join(parts) if parts else "(empty output)"


def render_cited_rubric(template: str, bundle: JudgeBundle) -> str:
    """Render a contract-declaring rubric. Same placeholders as the legacy
    renderer, so the template stays renderable by pre-contract code."""
    return template.format(agent_input=render_inputs(bundle), agent_output=render_outputs(bundle))


def judge_prompt_sha256(system: str, user: str) -> str:
    """sha256 over exactly what the judge was sent (system, NUL, user)."""
    return _sha256(f"{system}\x00{user}")


# ── Deterministic validation ────────────────────────────────────────────


def _occurrences(haystack: str, needle: str) -> list[int]:
    idx: list[int] = []
    start = 0
    while True:
        i = haystack.find(needle, start)
        if i < 0:
            return idx
        idx.append(i)
        start = i + 1


def validate_evidence(
    ev: JudgeEvidence,
    *,
    outputs: Mapping[str, str],
    input_ids: Iterable[str],
) -> ValidatedEvidence:
    """Validate one citation. Pure; never raises on model-shaped input."""
    quote = ev.claim_quote or ""
    span = ev.claim_span
    field = span.field if span is not None else ""
    ids = set(input_ids)

    location: str
    located: ClaimSpan | None = None
    reason: str | None = None

    if not quote:
        location, reason = "unlocated", "empty_quote"
    else:
        text = outputs.get(field) if field else None
        if (
            text is not None
            and span is not None
            and span.start is not None
            and span.end is not None
            and 0 <= span.start <= span.end <= len(text)
            and text[span.start : span.end] == quote
        ):
            location = "exact"
            located = ClaimSpan(field=field, start=span.start, end=span.end)
        else:
            in_field = _occurrences(text, quote) if text is not None else []
            if len(in_field) == 1:
                location, reason = "relocated", "span_mismatch"
                located = ClaimSpan(field=field, start=in_field[0], end=in_field[0] + len(quote))
            elif len(in_field) > 1:
                location, reason = "unlocated", "quote_ambiguous"
            else:
                hits = [(p, i) for p, t in outputs.items() for i in _occurrences(t, quote)]
                if len(hits) == 1:
                    p, i = hits[0]
                    location = "relocated"
                    reason = "unknown_field" if text is None else "field_mismatch"
                    located = ClaimSpan(field=p, start=i, end=i + len(quote))
                elif hits:
                    location, reason = "unlocated", "quote_ambiguous"
                else:
                    location, reason = "unlocated", "quote_not_found"

    ref = (ev.source_ref or "").strip()
    if not ref:
        source_status = "missing"
    elif ref in ids:
        source_status = "known"
    else:
        source_status = "unknown"

    valid = location in ("exact", "relocated") and source_status == "known"
    if source_status != "known":
        ref_reason = f"{source_status}_source_ref"
        reason = f"{reason}+{ref_reason}" if reason else ref_reason

    return ValidatedEvidence(
        claim_quote=quote,
        claim_span=span,
        source_ref=ev.source_ref or "",
        location=location,  # type: ignore[arg-type]
        located_span=located,
        source_status=source_status,  # type: ignore[arg-type]
        valid=valid,
        reason=reason,
    )


def _evidence_of(score: Any) -> list[JudgeEvidence]:
    raw = getattr(score, "evidence", None)
    if raw is None and getattr(score, "model_extra", None):
        raw = score.model_extra.get("evidence")
    out: list[JudgeEvidence] = []
    for item in raw or []:
        if isinstance(item, JudgeEvidence):
            out.append(item)
        elif isinstance(item, Mapping):
            out.append(JudgeEvidence.model_validate(item))
        else:
            # A bare string or number where an object belongs: keep it as an
            # unlocatable citation rather than dropping it from the count.
            out.append(JudgeEvidence(claim_quote="", source_ref=str(item)))
    return out


def validate_dimension_scores(
    dimension_scores: Sequence[Any],
    bundle: JudgeBundle,
    *,
    contract: str = CITATION_CONTRACT,
) -> tuple[list[CitedRubricDimensionScore], CitationCheck]:
    """Validate every finding's evidence; return persisted scores + counts."""
    ids = bundle.input_ids
    scored: list[CitedRubricDimensionScore] = []
    check = CitationCheck(
        contract=contract,
        inputs_offered=len(bundle.inputs),
        output_fields_offered=len(bundle.outputs),
    )
    for s in dimension_scores:
        validated = [validate_evidence(e, outputs=bundle.outputs, input_ids=ids) for e in _evidence_of(s)]
        n_valid = sum(1 for v in validated if v.valid)
        base = {k: v for k, v in s.model_dump().items() if k not in ("evidence", "valid_evidence_count", "unsupported")}
        scored.append(
            CitedRubricDimensionScore(
                **base,
                evidence=validated,
                valid_evidence_count=n_valid,
                unsupported=n_valid == 0,
            )
        )
        check.findings_total += 1
        if n_valid == 0:
            check.findings_unsupported += 1
            check.unsupported_dimensions.append(s.dimension)
        for v in validated:
            check.citations_total += 1
            check.citations_valid += int(v.valid)
            check.located_exact += int(v.location == "exact")
            check.located_relocated += int(v.location == "relocated")
            check.unlocated += int(v.location == "unlocated")
            check.unknown_source_ref += int(v.source_status == "unknown")
            check.missing_source_ref += int(v.source_status == "missing")
    check.citations_invalid = check.citations_total - check.citations_valid
    return scored, check
