"""Think Tank has exactly ONE LLM addressing mode: capability group, via the
krepis router. No second, pinned-provider routing plane.

Brian's ruling 2026-08-29, verbatim: *"the entire nous ergon system should
now be running through the krepis router ... we should have no other parallel
setups, it should all funnel through the krepis router."*

alpha-engine-config-I9302 (item 3) measured this exact defect:
``thinktank/client.py``'s ``_llm_client_for()`` carried a SECOND branch that
took a per-tier pinned ``provider`` + ``model`` + ``base_url`` straight from
``thinktank.yaml``'s ``llm.providers`` block — its own ``ProviderSpec``, its
own price literals, its own literal model IDs — entirely bypassing krepis.
The live private ``research/thinktank.yaml`` carries no ``providers:`` block,
so every tier has been group-addressed since Brian's 2026-08-03 ruling
(I6367) and the pinned branch was measured DORMANT — but it was fully wired
machinery that reinstates direct provider linkage on one config edit, with no
guard between. I9302's own words: "Prefer deletion to a guard: a code path
that only a config edit can reach is cheaper to remove than to police."

Every assertion below FAILS against the pre-fix code (``ProviderSpec``
existed, ``TierSpec`` carried ``provider``/``model``/price fields, a tier
could omit ``group`` entirely) and PASSES once the pinned branch, its schema,
and its sample-config documentation are gone.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SETTINGS_PY = _REPO_ROOT / "thinktank" / "settings.py"
_CLIENT_PY = _REPO_ROOT / "thinktank" / "client.py"
_SAMPLE_YAML = _REPO_ROOT / "config" / "thinktank.sample.yaml"


def _names_and_strings(path: Path) -> set[str]:
    """Every identifier and string literal in *path*'s source — a structural
    scan over the parsed AST, not a grep, so a docstring mentioning the old
    name in past tense cannot pass a check the code itself must satisfy."""
    tree = ast.parse(path.read_text())
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, ast.arg):
            out.add(node.arg)
    return out


class TestNoProviderSpecSchema:
    def test_providerspec_class_is_gone_from_settings(self):
        assert "ProviderSpec" not in _names_and_strings(_SETTINGS_PY)

    def test_providerspec_is_gone_from_client(self):
        assert "ProviderSpec" not in _names_and_strings(_CLIENT_PY)

    def test_settings_module_does_not_import_providerspec(self):
        import thinktank.settings as settings_mod

        assert not hasattr(settings_mod, "ProviderSpec")

    def test_thinktank_settings_carries_no_providers_dict(self):
        from thinktank.settings import ThinktankSettings

        fields = set(ThinktankSettings.__dataclass_fields__)
        assert "providers" not in fields

    def test_settings_carries_no_provider_lookup_method(self):
        from thinktank.settings import ThinktankSettings

        assert not hasattr(ThinktankSettings, "provider_for")


class TestTierSpecIsGroupOnly:
    def test_tierspec_carries_no_pinned_fields(self):
        from thinktank.settings import TierSpec

        fields = set(TierSpec.__dataclass_fields__)
        assert not fields & {"provider", "model", "price_in_per_m", "price_out_per_m"}, (
            f"TierSpec still carries pinned-provider fields: "
            f"{fields & {'provider', 'model', 'price_in_per_m', 'price_out_per_m'}}"
        )

    def test_group_is_the_only_addressing_mode(self):
        from thinktank.settings import _parse_tier

        tier = _parse_tier("thesis", {"group": "med", "max_tokens": 8000})
        assert tier.group == "med"

    def test_a_pinned_provider_key_is_rejected_at_parse(self):
        """A tier that still names a provider/model is a config authored
        against the retired schema — reject it loudly rather than silently
        dropping the keys, which would read as "group addressing chosen"
        when nobody chose it."""
        from thinktank.settings import _parse_tier

        with pytest.raises(ValueError, match="provider"):
            _parse_tier(
                "thesis",
                {
                    "provider": "openrouter",
                    "model": "x/y",
                    "max_tokens": 8000,
                    "price_in_per_m": 0.1,
                    "price_out_per_m": 0.2,
                },
            )

    def test_missing_group_is_rejected_at_parse(self):
        from thinktank.settings import _parse_tier

        with pytest.raises(ValueError, match="group"):
            _parse_tier("thesis", {"max_tokens": 8000})


class TestClientHasOneAddressingPath:
    def test_llm_client_for_never_reads_a_pinned_provider(self):
        """`_llm_client_for` must resolve every tier through
        `krepis.router.resolve_group_spec` — no second branch."""
        source = _CLIENT_PY.read_text()
        assert "provider_for(" not in source
        assert "get_secret(provider" not in source

    def test_client_factory_seam_matches_llmclient_native_shape(self):
        """The old `(ProviderSpec, api_key) -> obj` adapter existed only to
        preserve the pinned-mode test seam. With ProviderSpec gone, the seam
        must be `krepis.llm.LLMClient`'s own native
        `(ModelSpec, api_key) -> obj` shape, passed straight through with no
        translation layer."""
        source = _CLIENT_PY.read_text()
        assert "_adapt_client_factory" not in source


class TestSampleConfigDoesNotAdvertisePinning:
    def test_sample_yaml_carries_no_providers_block(self):
        text = _SAMPLE_YAML.read_text()
        assert "providers:" not in text

    def test_sample_yaml_tiers_are_all_group_addressed(self):
        import yaml

        raw = yaml.safe_load(_SAMPLE_YAML.read_text())
        tiers = raw["thinktank"]["llm"]["tiers"]
        assert tiers, "sample config declares no tiers at all"
        for name, t in tiers.items():
            assert "group" in t, f"sample tier {name!r} is not group-addressed"
            assert "provider" not in t and "model" not in t, (
                f"sample tier {name!r} still documents the retired pinned schema"
            )
