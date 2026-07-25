"""Producer-side schema conformance for the signals slot contract (M0, config#989).

Complements ``test_signals_producer_contract.py`` (field-set pins on the pure
builder): that test pins *which keys the producer emits*; this one validates
*the assembled artifact* against the versioned Slot R contract in
``alpha_engine_lib.contracts`` (lib >= 0.59.1) — the same schema executor /
predictor / backtester consumer fixtures and external slot implementations
validate against.

The payload comes from the REAL pure builder (``_build_signals_payload``) over
the same synthetic state the producer-contract test drives — not hand-built.
"""

from __future__ import annotations

import pytest

contracts = pytest.importorskip(
    "nousergon_lib.contracts",
    reason="needs alpha-engine-lib[contracts] >= 0.59.1",
)

# Imported after the importorskip guard above.
from graph.research_graph import _build_signals_payload  # noqa: E402
from tests.test_signals_producer_contract import _synthetic_state  # noqa: E402


def _payload() -> dict:
    return _build_signals_payload(_synthetic_state())


class TestProducerOutputConformsToSlotContract:
    def test_built_payload_validates(self):
        contracts.validate("signals", _payload())  # raises ContractViolation on drift

    def test_every_universe_entry_validates_individually(self):
        # per-item errors surface with the ticker-indexed path for debuggability
        payload = _payload()
        assert payload["universe"], "synthetic state must produce universe entries"
        contracts.validate("signals", payload)

    def test_broken_payload_fails_loud(self):
        """Red-fixture demo (config#989 closes-when): dropping a load-bearing
        per-item field must produce a non-empty error list."""
        payload = _payload()
        del payload["universe"][0]["conviction"]
        errors = contracts.conformance_errors("signals", payload)
        assert errors and "conviction" in " ".join(errors)

    def test_missing_top_level_fails_loud(self):
        payload = _payload()
        del payload["sector_modifiers"]
        assert contracts.conformance_errors("signals", payload)


class TestSchemaAndProducerPinAgree:
    """The lib schema's required sets and the repo-local producer pins must not
    drift apart — both express the same contract."""

    def test_schema_required_per_item_matches_local_pin(self):
        from tests.test_signals_producer_contract import _REQUIRED_PER_ITEM

        schema = contracts.load_schema("signals")
        schema_required = set(schema["$defs"]["signal_entry"]["required"])
        assert schema_required == set(_REQUIRED_PER_ITEM), (
            "lib schema and test_signals_producer_contract._REQUIRED_PER_ITEM "
            "disagree — update both sides deliberately, never one"
        )

    def test_schema_required_top_level_subset_of_local_pin(self):
        from tests.test_signals_producer_contract import _REQUIRED_TOP_LEVEL

        schema = contracts.load_schema("signals")
        # local pin may require MORE than the cross-repo contract (producer
        # discipline can exceed the consumer floor), never less
        missing = set(schema["required"]) - set(_REQUIRED_TOP_LEVEL)
        assert not missing, f"lib schema requires fields the producer pin lacks: {sorted(missing)}"
