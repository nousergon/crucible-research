"""Consumer contract test — the features parquet column contract (alpha-engine-config-I10793).

``tests/contracts/features_parquet.schema.json`` is a pinned copy of
nousergon-data's ``contracts/features_parquet.schema.json``, which nousergon-data
GENERATES from ``features/registry.py::CATALOG``. It declares every column of
``features/{date}/{group}.parquet`` (written by D31 daily and rewritten by D50
from the settled bar): one ``$defs`` row schema per group, ``ticker``/``date``
required, ``additionalProperties: false``. This repo never imports
nousergon-data; the versioned JSON Schema is the coupling.

The reader is ``scoring/factor_scoring.py``: ``_resolve_group_snapshot`` reads
``technical.parquet`` and ``fundamental.parquet``, and ``compute_factor_composites``
ranks the raw columns the composite definitions name. This test fails here, at
review time, when a contract change would silently empty a pillar:

  1. the pinned schema is a valid JSON Schema;
  2. every raw column the champion and challenger composites reference (and
     the inputs of the derived ``sustainable_growth_rate``) is declared in the
     technical or fundamental group; the derived columns are declared in
     neither, so they cannot shadow a parquet column;
  3. the technical columns the reader projects (``_FACTOR_TECHNICAL_COLS``)
     are declared in the technical group;
  4. a contract-conformant snapshot served through the REAL
     ``_resolve_group_snapshot`` validates and yields every composite with all
     of its parquet-backed components counted;
  5. the contract rejects an undeclared column and a missing ticker.

A stale copy is also caught on the producer side: nousergon-data's ``data_gate``
compares this file's validation shape with the producer's on every read.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock

import jsonschema
import pandas as pd
import pytest

from scoring import factor_scoring as fs

CONTRACT = Path(__file__).parent / "contracts" / "features_parquet.schema.json"
DATE = "2026-10-14"
#: Read from the two groups ``_resolve_group_snapshot`` is called for.
READ_GROUPS = ("technical", "fundamental")


def _schema() -> dict:
    return json.loads(CONTRACT.read_text())


def _declared(group: str) -> dict:
    return _schema()["$defs"][group]["properties"]


def _all_definitions() -> dict[str, list[tuple[str, float, bool]]]:
    defs = {f"baseline:{k}": v for k, v in fs._BASELINE_COMPOSITE_DEFS.items()}
    defs.update({f"challenger:{k}": v for k, v in fs.challenger_composite_defs().items()})
    defs.update({f"live:{k}": v for k, v in fs._COMPOSITE_DEFS.items()})
    return defs


def _raw_columns() -> set[str]:
    cols = {c for components in _all_definitions().values() for c, _, _ in components}
    return cols - set(fs._DERIVED_FACTOR_DEFS)


def _conformant_rows(group: str, tickers: list[str]) -> list[dict]:
    rows = []
    for i, ticker in enumerate(tickers):
        row = {}
        for column, prop in _declared(group).items():
            if column == "ticker":
                row[column] = ticker
            elif column == "date":
                row[column] = DATE
            else:
                assert "number" in prop["type"], f"{group}.{column}: unexpected type {prop['type']}"
                row[column] = float(i + 1) / 10.0
        rows.append(row)
    return rows


def _parquet(rows: list[dict]) -> bytes:
    buf = io.BytesIO()
    pd.DataFrame(rows).to_parquet(buf, index=False, engine="pyarrow")
    return buf.getvalue()


def test_pinned_schema_is_valid():
    jsonschema.Draft202012Validator.check_schema(_schema())


class TestReaderColumnsAreDeclared:
    def test_the_groups_the_reader_reads_are_declared_per_ticker(self):
        for group in READ_GROUPS:
            row = _schema()["$defs"][group]
            assert row["required"] == ["ticker", "date"], group

    def test_every_raw_composite_column_is_declared_in_a_read_group(self):
        declared = set(_declared("technical")) | set(_declared("fundamental"))
        missing = sorted(_raw_columns() - declared)
        assert not missing, (
            f"composite definitions reference columns the pinned contract does not declare in "
            f"technical/fundamental: {missing}. Each would rank NaN for every ticker."
        )

    def test_derived_factor_inputs_are_declared(self):
        # sustainable_growth_rate = roe * (1 - payout_ratio)
        assert {"roe", "payout_ratio"} <= set(_declared("fundamental"))

    def test_derived_columns_cannot_collide_with_a_parquet_column(self):
        declared = set(_declared("technical")) | set(_declared("fundamental"))
        assert not (set(fs._DERIVED_FACTOR_DEFS) & declared)

    def test_reader_technical_projection_is_declared(self):
        missing = sorted(set(fs._FACTOR_TECHNICAL_COLS) - set(_declared("technical")))
        assert not missing, missing


class TestRealReaderOverAConformantSnapshot:
    def _s3(self, bodies: dict[str, bytes]) -> MagicMock:
        s3 = MagicMock()

        def _get(Bucket, Key):  # noqa: N803 - boto3 keyword names
            if Key not in bodies:
                raise KeyError(Key)
            return {"Body": io.BytesIO(bodies[Key])}

        s3.get_object.side_effect = _get
        return s3

    def test_resolved_snapshot_validates_and_feeds_every_composite(self):
        tickers = ["AAA", "BBB", "CCC", "DDD"]
        rows = {g: _conformant_rows(g, tickers) for g in READ_GROUPS}
        jsonschema.validate(instance=rows, schema=_schema())
        s3 = self._s3({f"features/{DATE}/{g}.parquet": _parquet(rows[g]) for g in READ_GROUPS})

        frames = {}
        for group in READ_GROUPS:
            df, resolved = fs._resolve_group_snapshot(
                s3, "bucket", group, DATE, max_staleness_td=3, candidate_dates=[DATE],
            )
            assert resolved == DATE
            assert list(df.columns) == list(_declared(group))
            jsonschema.validate(
                instance={group: json.loads(df.to_json(orient="records"))}, schema=_schema(),
            )
            frames[group] = df

        defs = fs._BASELINE_COMPOSITE_DEFS
        out = fs.compute_factor_composites(
            frames["technical"], frames["fundamental"], dict.fromkeys(tickers, "Tech"),
            composite_defs=defs,
        )
        assert sorted(out["ticker"]) == tickers
        for composite, components in defs.items():
            parquet_backed = [c for c, _, _ in components if c != "institutional_accumulation_score"]
            assert out[composite].notna().all(), composite
            assert (out[f"{composite[:-6]}_n"] >= len(parquet_backed)).all(), composite


class TestContractRejectsWhatTheReaderCannotUse:
    def test_an_undeclared_column_is_rejected(self):
        rows = _conformant_rows("technical", ["AAA"])
        rows[0]["not_a_catalog_column_pct"] = 0.1
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance={"technical": rows}, schema=_schema())

    def test_a_row_without_ticker_is_rejected(self):
        rows = _conformant_rows("fundamental", ["AAA"])
        del rows[0]["ticker"]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance={"fundamental": rows}, schema=_schema())
