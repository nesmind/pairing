"""Unit tests for app/services/installed_projector_catalog.py — pure functions, no I/O, so no monkeypatching
needed here at all."""

from app.services.engine_support_checker import AlwaysSupportedChecker, EngineSupportSet
from app.services.installed_projector_catalog import InstalledProjectorCatalog
from app.services.matricxon_support_checker import MatricxonSupportChecker

_PROJECTOR_TAG = (
    "hf.co/concedo/llama-joycaption-beta-one-hf-llava-mmproj-gguf:llama-joycaption-beta-one-llava-mmproj-model-f16"
)

_NO_SUPPORT = EngineSupportSet({"ollama": AlwaysSupportedChecker(), "matricxon": MatricxonSupportChecker.unloaded()})


def test_display_name_strips_the_hf_co_prefix_the_file_suffix_and_the_gguf_noise():
    assert InstalledProjectorCatalog._display_name(_PROJECTOR_TAG) == "llama-joycaption-beta-one-hf-llava-mmproj"


def test_display_name_leaves_a_repo_with_no_gguf_suffix_alone():
    tag = "hf.co/second-state/Llava-v1.6-Vicuna-7B-GGUF:mmproj"
    assert InstalledProjectorCatalog._display_name(tag) == "Llava-v1.6-Vicuna-7B"


def test_vendor_from_tag_reads_the_real_hf_org_not_a_hardcoded_other():
    assert InstalledProjectorCatalog._vendor_from_tag(_PROJECTOR_TAG) == "concedo"


def test_vendor_from_tag_falls_back_to_other_for_a_non_hf_tag():
    assert InstalledProjectorCatalog._vendor_from_tag("some-plain-ollama-tag") == "Other"


def test_entries_surfaces_a_clip_family_tag_with_no_chat_capability():
    installed_models = [{"name": _PROJECTOR_TAG, "size": 877771808, "details": {"family": "clip"}}]

    entries = InstalledProjectorCatalog.entries(
        installed_models, catalog_tags=set(), hidden_tags=set(), is_admin=True, support_set=_NO_SUPPORT
    )

    assert len(entries) == 1
    entry = entries[0]
    assert entry.tag == _PROJECTOR_TAG
    # No more "(vision projector)" plain-text suffix - that's a real badge now (see renderModelRow's
    # is_projector check in settings.js), so the family string is just the plain display name.
    assert entry.family == "llama-joycaption-beta-one-hf-llava-mmproj"
    assert entry.parameter_size == ""
    # Real repo org ("concedo"), not the hardcoded "Other" this used to always be (see
    # InstalledProjectorCatalog._vendor_from_tag's own docstring) - groups this row under the same vendor
    # heading as its paired chat model in Settings > Model's installed list.
    assert entry.vendor == "concedo"
    assert entry.installed is True
    assert entry.is_projector is True
    assert entry.text_capable is False
    assert entry.vision is False
    assert entry.download_gb == 0.9
    assert entry.is_auto_discovered is True
    # Never shown by the frontend for a projector regardless (see entries' own comment), but should still read
    # as an honest "fine" rather than a real, always-false verdict_for(is_projector=True) result.
    assert entry.matricxon_supported is True


def test_entries_ignores_a_chat_capable_tag():
    installed_models = [{"name": "some-tag", "size": 1_000_000, "details": {"family": "mistral3"}}]

    entries = InstalledProjectorCatalog.entries(
        installed_models, catalog_tags=set(), hidden_tags=set(), is_admin=True, support_set=_NO_SUPPORT
    )

    assert entries == []


def test_entries_skips_a_tag_already_in_the_curated_catalog():
    installed_models = [{"name": _PROJECTOR_TAG, "size": 1_000_000, "details": {"family": "clip"}}]

    entries = InstalledProjectorCatalog.entries(
        installed_models, catalog_tags={_PROJECTOR_TAG}, hidden_tags=set(), is_admin=True, support_set=_NO_SUPPORT
    )

    assert entries == []


def test_entries_hides_a_hidden_tag_from_a_regular_user():
    installed_models = [{"name": _PROJECTOR_TAG, "size": 1_000_000, "details": {"family": "clip"}}]

    regular = InstalledProjectorCatalog.entries(
        installed_models,
        catalog_tags=set(),
        hidden_tags={_PROJECTOR_TAG},
        is_admin=False,
        support_set=_NO_SUPPORT,
    )
    admin = InstalledProjectorCatalog.entries(
        installed_models, catalog_tags=set(), hidden_tags={_PROJECTOR_TAG}, is_admin=True, support_set=_NO_SUPPORT
    )

    assert regular == []
    assert len(admin) == 1
    assert admin[0].hidden is True


def test_entries_reads_matricxons_own_real_ram_estimate():
    installed_models = [{"name": _PROJECTOR_TAG, "size": 1_000_000, "details": {"family": "clip"}}]
    support_set = EngineSupportSet(
        {
            "ollama": AlwaysSupportedChecker(),
            "matricxon": MatricxonSupportChecker(None, {_PROJECTOR_TAG: {"estimated_ram_gb": 1.04}}),
        }
    )

    entries = InstalledProjectorCatalog.entries(
        installed_models,
        catalog_tags=set(),
        hidden_tags=set(),
        is_admin=True,
        support_set=support_set,
    )

    assert entries[0].min_ram_gb_matricxon == 1.04
