"""Unit tests for ``scripts/corpus_stats.py`` (distillation SFT-corpus stats).

Locks down:

- Dedup on the model INPUT (identical ``input_messages`` collapse to one pair).
- Schema-generation normalization: v1 (top-level ``agent_id``/``model_name``,
  no ``producer``/``meta``) and v2 (``meta.agent_id`` + ``producer``) both count.
- Teacher segregation: the trigger metric is the DOMINANT teacher's quant count,
  not a blend across teacher versions.
- Trigger metric = deduped ``sector_quant`` (quant-calibrator) pairs vs target.
- Capture freshness: a missing Saturday between first + last capture is surfaced.
- End-to-end over moto S3: reads both prefixes, writes latest.json + dated.
"""

from __future__ import annotations

import json

import boto3
from moto import mock_aws

from scripts import corpus_stats as cs

_BUCKET = "alpha-engine-research"


def _v2(agent_id, model, inp, *, producer="crucible_research", source="live"):
    return {
        "schema_version": 2, "producer": producer, "captured_at": "2026-06-27T05:00:00+00:00",
        "model": model, "call_seq": 0, "input_messages": inp, "output_message": {"x": 1},
        "meta": {"agent_id": agent_id, "source": source, "run_id": "2026-06-26"},
    }


def _v1(agent_id, model, inp):
    return {
        "schema_version": 1, "timestamp": "2026-06-19T05:00:00+00:00",
        "model_name": model, "call_seq": 0, "input_messages": inp,
        "output_message": {"x": 1}, "agent_id": agent_id, "run_id": "2026-06-18",
    }


def _v3(agent_id, model, inp, *, producer="crucible_research", source="live", content_hash=None):
    """A schema-v3 record: source + content_hash live in the standardized
    top-level ``provenance`` block (nousergon_lib.sft #150 / config#1539)."""
    rec = {
        "schema_version": 3, "producer": producer, "captured_at": "2026-06-27T05:00:00+00:00",
        "model": model, "call_seq": 0, "input_messages": inp, "output_text": "ok",
        "meta": {"agent_id": agent_id, "run_id": "2026-06-26"},
        "provenance": {"source": source},
    }
    if content_hash is not None:
        rec["provenance"]["content_hash"] = content_hash
    return rec


def _run(records_by_key, gen_date="2026-07-01"):
    return cs.compute_stats(records_by_key, generated_date=gen_date)


def _lines(*recs):
    return [json.dumps(r) for r in recs]


def test_dedup_on_input():
    dup = _v2("sector_quant:tech", "haiku", [{"role": "user", "content": "AMD"}])
    stats = _run([("2026-06-27/r/a.jsonl", "research", _lines(dup, dict(dup)))])
    assert stats["totals"]["raw_records"] == 2
    assert stats["totals"]["duplicates_dropped"] == 1
    assert stats["totals"]["deduped_pairs"] == 1


def test_v1_and_v2_both_counted_and_producer_inferred():
    recs = _lines(
        _v1("sector_team:health", "haiku", [{"c": "1"}]),
        _v2("sector_qual:tech", "haiku", [{"c": "2"}]),
    )
    stats = _run([("2026-06-19/r/a.jsonl", "research", [recs[0]]),
                  ("2026-06-27/r/b.jsonl", "research", [recs[1]])])
    assert stats["totals"]["deduped_pairs"] == 2
    assert stats["by_producer"]["crucible_research"] == 2
    assert stats["by_schema_version"] == {"1": 1, "2": 1}


def test_v3_provenance_source_is_authoritative_not_defaulted():
    """A v3 record's source is read from provenance.source — a replay-minted
    trace is segregated as `replay`, never silently defaulted to `live`
    (config#1539: replay + live must not blend)."""
    live = _v3("sector_quant:tech", "sonnet-5", [{"c": "AMD"}], source="live")
    replay = _v3("sector_quant:tech", "sonnet-5", [{"c": "NVDA"}], source="replay")
    stats = _run([("2026-06-27/r/a.jsonl", "research", _lines(live, replay))])
    assert stats["by_source"] == {"live": 1, "replay": 1}
    assert stats["by_schema_version"] == {"3": 2}


def test_v3_dedup_keys_off_canonical_provenance_content_hash():
    """Dedup uses the writer's canonical provenance.content_hash when present,
    so two records the lib canonicalized to the SAME hash collapse even if their
    raw input_messages differ (cross-producer canonicalization consistency)."""
    a = _v3("sector_quant:tech", "sonnet-5", [{"c": "raw-A"}], content_hash="deadbeef")
    b = _v3("sector_quant:tech", "sonnet-5", [{"c": "raw-B-different"}], content_hash="deadbeef")
    stats = _run([("2026-06-27/r/x.jsonl", "research", _lines(a, b))])
    assert stats["totals"]["raw_records"] == 2
    assert stats["totals"]["duplicates_dropped"] == 1
    assert stats["totals"]["deduped_pairs"] == 1


def test_trigger_metric_is_dominant_teacher_quant():
    recs = _lines(
        _v2("sector_quant:a", "haiku", [{"c": "a"}]),
        _v2("sector_quant:b", "haiku", [{"c": "b"}]),
        _v2("sector_quant:c", "sonnet", [{"c": "c"}]),   # different teacher
        _v2("sector_qual:d", "haiku", [{"c": "d"}]),      # different task
    )
    stats = _run([("2026-06-27/r/x.jsonl", "research", recs)])
    trg = stats["trigger"]
    assert trg["task"] == "sector_quant"
    assert trg["quant_total_all_teachers"] == 3
    assert trg["dominant_teacher"] == "haiku"
    assert trg["deduped_single_teacher"] == 2   # only the dominant teacher's quant pairs
    assert trg["crossed"] is False


def test_metron_producer_inferred_from_prefix():
    rec = {"schema_version": 2, "captured_at": "t", "input_messages": [{"c": "z"}],
           "meta": {"posture": "live"}}
    stats = _run([("metron/_sft_raw/2026-06-27/p/x.jsonl", "metron", _lines(rec))])
    assert stats["by_producer"].get("metron_advisor") == 1
    assert stats["by_task"].get("advisor") == 1


def test_unparseable_counted_not_crashed():
    stats = _run([("2026-06-27/r/a.jsonl", "research",
                   ["{not json", json.dumps(_v2("sector_quant:a", "haiku", [{"c": "1"}]))])])
    assert stats["totals"]["unparseable"] == 1
    assert stats["totals"]["deduped_pairs"] == 1


def test_missing_saturday_surfaced():
    # capture on Sat 2026-06-13 and Sat 2026-06-27; 2026-06-20 (Sat) is missing.
    recs_a = _lines(_v2("sector_quant:a", "haiku", [{"c": "a"}]))
    recs_b = _lines(_v2("sector_quant:b", "haiku", [{"c": "b"}]))
    stats = _run([("2026-06-13/r/a.jsonl", "research", recs_a),
                  ("2026-06-27/r/b.jsonl", "research", recs_b)])
    assert "2026-06-20" in stats["capture"]["missing_saturdays"]
    assert stats["capture"]["last_captured_date"] == "2026-06-27"
    assert [g["cumulative"] for g in stats["growth"]] == [1, 2]


@mock_aws
def test_end_to_end_s3_read_write():
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket=_BUCKET)
    client.put_object(
        Bucket=_BUCKET,
        Key="decision_artifacts/_sft_raw/2026-06-27/2026-06-26/sector_quant:tech.jsonl",
        Body="\n".join(_lines(
            _v2("sector_quant:tech", "haiku", [{"c": "AMD"}]),
            _v2("sector_quant:fin", "haiku", [{"c": "AFL"}]),
        )).encode(),
    )
    stats = cs.compute_corpus_stats(client, _BUCKET, target_date="2026-07-01")
    assert stats["totals"]["deduped_pairs"] == 2
    assert stats["trigger"]["deduped_single_teacher"] == 2
    # both dated + latest written
    got = client.get_object(Bucket=_BUCKET,
                            Key="decision_artifacts/distillation/corpus_stats/latest.json")
    assert json.loads(got["Body"].read())["totals"]["deduped_pairs"] == 2
    client.get_object(Bucket=_BUCKET,
                      Key="decision_artifacts/distillation/corpus_stats/2026-07-01.json")
