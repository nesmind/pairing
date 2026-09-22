"""Unit tests for app/services/installed_projector_catalog.py — pure functions, no I/O, so no monkeypatching
needed here at all."""

from app.services.installed_projector_catalog import InstalledProjectorCatalog
from app.services.matricxon_support_checker import MatricxonSupportChecker

_PROJECTOR_TAG = (
    "hf.co/concedo/llama-joycaption-beta-one-hf-llava-mmproj-gguf:llama-joycaption-beta-one-llava-mmproj-model-f16"
)

_NO_MATRICXON = MatricxonSupportChecker(None, None)


def test_display_name_strips_the_hf_co_prefix_the_file_suffix_and_the_gguf_noise():
    assert InstalledProjectorCatalog._display_name(_PROJECTOR_TAG) == "llama-joycaption-beta-one-hf-llava-mmproj"


def test_display_name_leaves_a_repo_with_no_gguf_suffix_alone():
    tag = "hf.co/second-state/Llava-v1.6-Vicuna-7B-GGUF:mmproj"
    assert InstalledProjectorCatalog._display_name(tag) == "Llava-v1.6-Vicuna-7B"


def test_entries_surfaces_a_clip_family_tag_with_no_chat_capability():
    installed_models = [{"name": _PROJECTOR_TAG, "size": 877771808, "details": {"family": "clip"}}]

    entries = InstalledProjectorCatalog.entries(
        installed_models, catalog_tags=set(), hidden_tags=set(), is_admin=True, checker=_NO_MATRICXON
    )

    assert len(entries) == 1
    entry = entries[0]
    assert entry.tag == _PROJECTOR_TAG
    assert entry.family == "llama-joycaption-beta-one-hf-llava-mmproj (vision projector)"
    assert entry.installed is True
    assert entry.is_projector is True
    assert entry.text_capable is False
    assert entry.vision is False
    assert entry.download_gb == 0.9
    assert entry.is_auto_discovered is True


def test_entries_ignores_a_chat_capable_tag():
    installed_models = [{"name": "some-tag", "size": 1_000_000, "details": {"family": "mistral3"}}]

    entries = InstalledProjectorCatalog.entries(
        installed_models, catalog_tags=set(), hidden_tags=set(), is_admin=True, checker=_NO_MATRICXON
    )

    assert entries == []


def test_entries_skips_a_tag_already_in_the_curated_catalog():
    installed_models = [{"name": _PROJECTOR_TAG, "size": 1_000_000, "details": {"family": "clip"}}]

    entries = InstalledProjectorCatalog.entries(
        installed_models, catalog_tags={_PROJECTOR_TAG}, hidden_tags=set(), is_admin=True, checker=_NO_MATRICXON
    )

    assert entries == []


def test_entries_hides_a_hidden_tag_from_a_regular_user():
    installed_models = [{"name": _PROJECTOR_TAG, "size": 1_000_000, "details": {"family": "clip"}}]

    regular = InstalledProjectorCatalog.entries(
        installed_models,
        catalog_tags=set(),
        hidden_tags={_PROJECTOR_TAG},
        is_admin=False,
        checker=_NO_MATRICXON,
    )
    admin = InstalledProjectorCatalog.entries(
        installed_models, catalog_tags=set(), hidden_tags={_PROJECTOR_TAG}, is_admin=True, checker=_NO_MATRICXON
    )

    assert regular == []
    assert len(admin) == 1
    assert admin[0].hidden is True


def test_entries_reads_matricxons_own_real_ram_estimate():
    installed_models = [{"name": _PROJECTOR_TAG, "size": 1_000_000, "details": {"family": "clip"}}]
    checker = MatricxonSupportChecker(None, {_PROJECTOR_TAG: {"estimated_ram_gb": 1.04}})

    entries = InstalledProjectorCatalog.entries(
        installed_models,
        catalog_tags=set(),
        hidden_tags=set(),
        is_admin=True,
        checker=checker,
    )

    assert entries[0].min_ram_gb_matricxon == 1.04
