"""Consumer contract test: the fundamentals snapshot body (alpha-engine-config-I11276).

``tests/contracts/fundamentals_snapshot.schema.json`` is a byte-copy of
nousergon-data's ``contracts/fundamentals_snapshot.schema.json`` (D10,
``archive/fundamentals/{date}.json``): a flat object keyed by ticker, each
value a closed row of 14 required numbers. This repo never imports
nousergon-data; the versioned JSON Schema is the coupling.

The reader is ``agents/sector_teams/quant_tools.py::read_fundamentals_from_s3``
(``get_object`` + ``json.loads``), whose result ``get_balance_sheet`` reads.
crucible-dashboard pins only this key's TEMPLATE, because its reader never
opens the body. This one does, so this pin is of the BODY. It fails here, at
review time, when a contract change would silently blank a balance-sheet field:

  1. the pinned schema is a valid JSON Schema;
  2. every field ``get_balance_sheet`` reads is required by the contract;
  3. a contract-conformant snapshot, and the producer's NEUTRAL sentinel row,
     served through the REAL reader on both of its paths (the ``run_date``
     hit and the prefix-scan fallback), validates and reaches
     ``get_balance_sheet`` with every read field intact;
  4. the contract rejects an undeclared field and a missing required one.

Fakes go in through ``monkeypatch``, never ``unittest.mock.patch``: this repo
has a documented full-suite bleed from the latter (see
``tests/test_get_balance_sheet_s3.py``).
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import boto3
import jsonschema
import pandas as pd
import pytest
from botocore.exceptions import ClientError

from agents.sector_teams import quant_tools
from agents.sector_teams.quant_tools import create_quant_tools, read_fundamentals_from_s3

CONTRACT = Path(__file__).parent / "contracts" / "fundamentals_snapshot.schema.json"
RUN_DATE = "2026-10-24"
OLDER = "2026-10-17"

#: snapshot field -> the ``get_balance_sheet`` output key it feeds.
BALANCE_SHEET_READS = {
    "debt_to_equity": "debt_to_equity",
    "current_ratio": "current_ratio",
    "pe_ratio": "pe_ratio",
    "pb_ratio": "price_to_book",
    "revenue_growth_yoy": "revenue_growth",
    "gross_margin": "gross_margins",
}
#: The full subset the reader's docstring promises to return.
READER_DOCUMENTED_FIELDS = {
    "pe_ratio",
    "pb_ratio",
    "debt_to_equity",
    "revenue_growth_yoy",
    "fcf_yield",
    "gross_margin",
    "roe",
    "current_ratio",
}


def _schema() -> dict:
    return json.loads(CONTRACT.read_text())


def _row_schema() -> dict:
    return _schema()["additionalProperties"]


def _conformant_row(seed: float) -> dict:
    return {field: seed + i / 100.0 for i, field in enumerate(_row_schema()["required"])}


def _neutral_row() -> dict:
    """The producer's NEUTRAL sentinel: every declared field at 0.0."""
    return dict.fromkeys(_row_schema()["required"], 0.0)


def _snapshot() -> dict:
    return {"NVDA": _conformant_row(1.0), "AAPL": _neutral_row()}


class _FakeS3:
    def __init__(self, bodies: dict[str, bytes]):
        self.bodies = bodies
        self.got: list[str] = []

    def get_object(self, Bucket, Key):  # noqa: N803 - boto3 keyword names
        self.got.append(Key)
        if Key not in self.bodies:
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": Key}}, "GetObject")
        return {"Body": io.BytesIO(self.bodies[Key])}

    def list_objects_v2(self, Bucket, Prefix, MaxKeys):  # noqa: N803
        return {"Contents": [{"Key": k} for k in self.bodies if k.startswith(Prefix)]}


@pytest.fixture
def fake_s3(monkeypatch):
    def install(bodies: dict[str, bytes]) -> _FakeS3:
        fake = _FakeS3(bodies)
        monkeypatch.setattr(boto3, "client", lambda *a, **k: fake)
        return fake

    return install


def _balance_sheet(fundamentals_data: dict, tickers: list[str]) -> dict:
    tools = create_quant_tools(
        {
            "price_data": {t: pd.DataFrame() for t in tickers},
            "technical_scores": {},
            "factor_profiles": {},
            "fundamentals_data": fundamentals_data,
        }
    )
    tool = next(t for t in tools if t.name == "get_balance_sheet")
    return json.loads(tool.invoke({"tickers": tickers}))


def _assert_reaches_balance_sheet(snapshot: dict, served: dict) -> None:
    jsonschema.validate(instance=served, schema=_schema())
    out = _balance_sheet(served, sorted(snapshot))
    for ticker, row in snapshot.items():
        for field, key in BALANCE_SHEET_READS.items():
            assert out[ticker][key] == row[field], (ticker, field)


def test_pinned_schema_is_valid():
    jsonschema.Draft202012Validator.check_schema(_schema())


def test_the_pin_is_keyed_by_the_producer_and_its_key_template():
    schema = _schema()
    assert schema["$id"] == "nousergon-data/contracts/fundamentals_snapshot.schema.json"
    assert schema["x-key-pattern"] == "archive/fundamentals/{date}.json"


class TestReaderFieldsAreRequired:
    def test_every_balance_sheet_field_is_required(self):
        missing = sorted(set(BALANCE_SHEET_READS) - set(_row_schema()["required"]))
        assert not missing, (
            f"get_balance_sheet reads fields the pinned contract does not require: {missing}. "
            "Each would come back None for every ticker."
        )

    def test_every_documented_reader_field_is_required(self):
        assert READER_DOCUMENTED_FIELDS <= set(_row_schema()["required"])

    def test_the_reader_still_reads_exactly_those_fields(self):
        """Pins the test's own map against the tool source, so a new read
        site in get_balance_sheet cannot slip past item 2 above."""
        import inspect

        src = inspect.getsource(create_quant_tools)
        body = src[src.index("def get_balance_sheet") : src.index("def get_price_performance")]
        read = {seg.split('"')[0] for seg in body.split('f.get("')[1:]}
        assert read == set(BALANCE_SHEET_READS)


class TestRealReaderOverAConformantSnapshot:
    def test_the_run_date_hit_path(self, fake_s3):
        snap = _snapshot()
        fake = fake_s3({f"archive/fundamentals/{RUN_DATE}.json": json.dumps(snap).encode()})
        served = read_fundamentals_from_s3(run_date=RUN_DATE, bucket="b")
        assert fake.got == [f"archive/fundamentals/{RUN_DATE}.json"]
        _assert_reaches_balance_sheet(snap, served)

    def test_the_prefix_scan_fallback_takes_the_newest_snapshot(self, fake_s3):
        snap = _snapshot()
        stale = {"NVDA": _conformant_row(9.0)}
        fake = fake_s3(
            {
                f"archive/fundamentals/{OLDER}.json": json.dumps(stale).encode(),
                f"archive/fundamentals/{RUN_DATE}.json": json.dumps(snap).encode(),
            }
        )
        served = read_fundamentals_from_s3(bucket="b")
        assert fake.got == [f"archive/fundamentals/{RUN_DATE}.json"]
        _assert_reaches_balance_sheet(snap, served)

    def test_a_missed_run_date_falls_back_to_the_scan(self, fake_s3):
        snap = _snapshot()
        fake = fake_s3({f"archive/fundamentals/{OLDER}.json": json.dumps(snap).encode()})
        served = read_fundamentals_from_s3(run_date=RUN_DATE, bucket="b")
        assert fake.got == [
            f"archive/fundamentals/{RUN_DATE}.json",
            f"archive/fundamentals/{OLDER}.json",
        ]
        _assert_reaches_balance_sheet(snap, served)

    def test_the_reader_module_is_the_one_under_test(self):
        assert quant_tools.read_fundamentals_from_s3 is read_fundamentals_from_s3


class TestContractRejectsWhatTheReaderCannotUse:
    def test_an_undeclared_field_is_rejected(self):
        snap = _snapshot()
        snap["NVDA"]["forward_pe"] = 12.0
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=snap, schema=_schema())

    @pytest.mark.parametrize("field", sorted(BALANCE_SHEET_READS))
    def test_a_row_missing_a_read_field_is_rejected(self, field):
        snap = _snapshot()
        del snap["NVDA"][field]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=snap, schema=_schema())

    def test_a_non_numeric_field_is_rejected(self):
        snap = _snapshot()
        snap["NVDA"]["pe_ratio"] = "n/a"
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=snap, schema=_schema())
