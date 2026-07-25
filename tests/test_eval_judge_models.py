"""Judge-model registry invariants (L4578(a)).

Locks the pin discipline so a future edit can't silently un-pin a judge,
fake a pin, drift the custom_id tag map, or turn the fail-loud resolve()
into a soft fallback.
"""

from __future__ import annotations

import re

import pytest

from evals import judge_models

_SPECS = (judge_models.HAIKU, judge_models.SONNET)
# Anthropic dated-snapshot suffix, e.g. '-20251001'.
_DATED_SNAPSHOT_RE = re.compile(r"-\d{8}$")


class TestSpecInvariants:
    def test_logical_keys_unique(self):
        keys = [s.logical_key for s in _SPECS]
        assert len(keys) == len(set(keys))

    def test_request_models_unique(self):
        reqs = [s.request_model for s in _SPECS]
        assert len(reqs) == len(set(reqs))

    def test_tags_unique(self):
        tags = [s.tag for s in _SPECS]
        assert len(tags) == len(set(tags))

    def test_pinned_spec_request_differs_from_logical_and_is_dated(self):
        """A pinned spec must actually pin — request_model is a dated
        snapshot distinct from the alias, not the alias relabelled."""
        for spec in _SPECS:
            if spec.pinned:
                assert spec.request_model != spec.logical_key, spec.logical_key
                assert _DATED_SNAPSHOT_RE.search(spec.request_model), (
                    f"{spec.logical_key} claims pinned but request_model "
                    f"{spec.request_model!r} has no dated suffix"
                )

    def test_unpinned_spec_requests_the_alias(self):
        """An unpinned spec must request the alias verbatim — no fake
        date suffix that would 404 (Sonnet 4.6 has no dated snapshot)."""
        for spec in _SPECS:
            if not spec.pinned:
                assert spec.request_model == spec.logical_key, spec.logical_key
                assert not _DATED_SNAPSHOT_RE.search(spec.request_model)

    def test_every_spec_documents_its_pin_decision(self):
        for spec in _SPECS:
            assert spec.pin_note.strip(), spec.logical_key


class TestKnownModels:
    def test_haiku_pinned_to_dated_snapshot(self):
        assert judge_models.HAIKU.logical_key == "claude-haiku-4-5"
        assert judge_models.HAIKU.request_model == "claude-haiku-4-5-20251001"
        assert judge_models.HAIKU.pinned is True

    def test_sonnet_unpinned_alias_only(self):
        # Sonnet 4.6 publishes no dated snapshot — the alias is canonical
        # and a date suffix 404s, so it cannot be pinned.
        assert judge_models.SONNET.logical_key == "claude-sonnet-4-6"
        assert judge_models.SONNET.request_model == "claude-sonnet-4-6"
        assert judge_models.SONNET.pinned is False


class TestResolve:
    def test_resolve_by_logical_key(self):
        assert judge_models.resolve("claude-haiku-4-5") is judge_models.HAIKU

    def test_resolve_by_request_model(self):
        assert (
            judge_models.resolve("claude-haiku-4-5-20251001")
            is judge_models.HAIKU
        )

    def test_resolve_by_tag(self):
        assert judge_models.resolve("s46") is judge_models.SONNET

    def test_resolve_unknown_fails_loud(self):
        # Closed audited set — an unknown id is a bug, not a fallback.
        with pytest.raises(KeyError, match="Unknown judge model"):
            judge_models.resolve("claude-haiku-9-9")

    def test_request_model_for_pins_haiku(self):
        assert (
            judge_models.request_model_for("claude-haiku-4-5")
            == "claude-haiku-4-5-20251001"
        )

    def test_request_model_for_passes_sonnet_alias(self):
        assert (
            judge_models.request_model_for("claude-sonnet-4-6")
            == "claude-sonnet-4-6"
        )


class TestTagMapContinuity:
    def test_tag_values_match_legacy_codec(self):
        # The custom_id codec depended on these exact tags before the
        # registry existed; changing them would orphan historical
        # custom_ids. Lock them via a subset check (not full-dict
        # equality) so a new registry entry (e.g. OPENROUTER_SHADOW,
        # config#2575) doesn't require editing this pin — only a change
        # to an EXISTING tag would fail it.
        assert judge_models.TAG_BY_LOGICAL["claude-haiku-4-5"] == "h45"
        assert judge_models.TAG_BY_LOGICAL["claude-sonnet-4-6"] == "s46"

    def test_judge_module_sources_tag_map_from_registry(self):
        # Single source of truth — judge._JUDGE_MODEL_TAG must BE the
        # registry map, not a divergent copy.
        from evals import judge

        assert judge._JUDGE_MODEL_TAG is judge_models.TAG_BY_LOGICAL
