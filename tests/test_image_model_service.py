"""Unit tests for app/services/image_model_service.py + the image-model routing in the extended catalog and
app/routers/image_model_admin.py. Hugging Face is faked (httpx.MockTransport / stubbed repo_files); the models
folder is a tmp_path."""

from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException

from app.routers import image_model_admin
from app.schemas import ImageModelRequest, SdCppConfig
from app.services import image_engine_service, image_model_service
from app.services.extended_model_catalog_service import ExtendedModelCatalog
from app.services.huggingface_client import HuggingFaceCatalogSearch
from app.services.image_model_service import ImageModelStore

_REAL_CLIENT = httpx.AsyncClient
_TAG = "hf.co/Green-Sky/SD-Turbo-GGUF:sd_turbo-f16-q8_0"
_REPO = "Green-Sky/SD-Turbo-GGUF"
_FILE = "sd_turbo-f16-q8_0.gguf"


@pytest.fixture(autouse=True)
def models_root(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()

    async def fake_resolve(_db):
        return root

    monkeypatch.setattr(image_model_service, "resolve_models_dir", fake_resolve)
    return root


async def _register(db, tag=_TAG):
    catalog = ExtendedModelCatalog(db)
    await catalog._save([{"tag": tag, "kind": "image", "family": "SD-Turbo", "download_gb": 2.0}])


def _fake_hf(monkeypatch, body=b"weights", status=200):
    def handler(_request):
        return httpx.Response(status, content=body, headers={"content-length": str(len(body))})

    monkeypatch.setattr(
        image_model_service.httpx,
        "AsyncClient",
        lambda **kw: _REAL_CLIENT(transport=httpx.MockTransport(handler), **kw),
    )


async def _download(store, tag=_TAG):
    repo_id, filename, dest = await store.prepare_pull(tag)
    return [e async for e in store.download(repo_id, filename, dest, None)]


@pytest.mark.asyncio
async def test_tag_path_mapping_round_trips(db, models_root):
    store = await ImageModelStore.open(db)
    path = store.path_for(_TAG)
    assert path == models_root / _REPO / _FILE
    assert store.tag_for(path) == _TAG
    assert store.tag_for(models_root / "loose.gguf") is None


@pytest.mark.parametrize(
    "tag", ["llama3:latest", "hf.co/a/b", "hf.co/../etc:passwd", "hf.co/a/b:../../x", "hf.co/a/b:/abs"]
)
def test_parse_tag_rejects_non_hf_and_path_tricks(tag):
    with pytest.raises(ValueError):
        ImageModelStore.parse_tag(tag)


@pytest.mark.asyncio
async def test_list_files_includes_tagged_and_loose_models_of_known_types(db, models_root):
    (models_root / _REPO).mkdir(parents=True)
    (models_root / _REPO / _FILE).write_bytes(b"x" * 10)
    (models_root / "loose.safetensors").write_bytes(b"y")
    (models_root / "notes.txt").write_text("ignore me")
    files = {f.name: f for f in (await ImageModelStore.open(db)).list_files()}
    assert set(files) == {f"{_REPO}/{_FILE}", "loose.safetensors"}
    assert files[f"{_REPO}/{_FILE}"].tag == _TAG and files["loose.safetensors"].tag is None


@pytest.mark.asyncio
async def test_pull_downloads_into_the_store_and_sets_the_engine_model_when_unset(db, models_root, monkeypatch):
    await _register(db)
    _fake_hf(monkeypatch, b"weights")

    events = await _download(await ImageModelStore.open(db))

    assert events[-1] == {"done": True}
    assert any(e.get("completed") == 7 and e.get("total") == 7 for e in events)
    dest = models_root / _REPO / _FILE
    assert dest.read_bytes() == b"weights" and not dest.with_name(dest.name + ".part").exists()
    assert (await image_engine_service.get_sdcpp_config(db)).model_path == str(dest)


@pytest.mark.asyncio
async def test_pull_keeps_an_already_chosen_engine_model(db, monkeypatch):
    await _register(db)
    await image_engine_service.set_sdcpp_config(db, SdCppConfig(model_path="/other.gguf"))
    _fake_hf(monkeypatch)
    await _download(await ImageModelStore.open(db))
    assert (await image_engine_service.get_sdcpp_config(db)).model_path == "/other.gguf"


@pytest.mark.asyncio
async def test_pull_reports_gated_and_http_failures_without_leaving_files(db, models_root, monkeypatch):
    await _register(db)
    _fake_hf(monkeypatch, status=401)
    assert "gated" in (await _download(await ImageModelStore.open(db)))[-1]["error"]
    _fake_hf(monkeypatch, status=404)
    assert "Could not download" in (await _download(await ImageModelStore.open(db)))[-1]["error"]
    assert not list((models_root / _REPO).glob("*"))


@pytest.mark.asyncio
async def test_pull_refuses_unregistered_tags_and_a_remote_engine(db):
    store = await ImageModelStore.open(db)
    with pytest.raises(ValueError, match="Unknown"):
        await store.prepare_pull("hf.co/someone/Else-GGUF:model")
    await _register(db)
    await image_engine_service.set_sdcpp_config(db, SdCppConfig(mode="remote", remote_hosts=["http://h:1"]))
    with pytest.raises(ValueError, match="Local mode"):
        await store.prepare_pull(_TAG)


@pytest.mark.asyncio
async def test_delete_removes_the_file_prunes_folders_and_clears_the_engine_model(db, models_root, monkeypatch):
    await _register(db)
    _fake_hf(monkeypatch)
    await _download(await ImageModelStore.open(db))

    await (await ImageModelStore.open(db)).delete(_TAG)

    assert not (models_root / "Green-Sky").exists() and models_root.exists()
    assert (await image_engine_service.get_sdcpp_config(db)).model_path is None


@pytest.mark.asyncio
async def test_delete_is_refused_for_a_remote_engine(db):
    await image_engine_service.set_sdcpp_config(db, SdCppConfig(mode="remote", remote_hosts=["http://h:1"]))
    with pytest.raises(ValueError, match="Local mode"):
        await (await ImageModelStore.open(db)).delete(_TAG)


@pytest.mark.asyncio
async def test_the_curated_default_is_pullable_without_any_admin_registration(db):
    repo_id, filename, dest = await (await ImageModelStore.open(db)).prepare_pull(_TAG)
    assert (repo_id, filename) == (_REPO, _FILE) and dest.name == _FILE


@pytest.mark.asyncio
async def test_catalog_lists_the_curated_default_first_not_installed(db):
    [entry] = await (await ImageModelStore.open(db)).catalog_entries()
    assert entry.tag == _TAG and entry.is_image and not entry.installed and not entry.removable
    assert entry.family == "SD-Turbo" and entry.download_gb == 2.02


@pytest.mark.asyncio
async def test_catalog_marks_an_installed_model_and_ignores_untagged_files(db, models_root):
    (models_root / _REPO).mkdir(parents=True)
    (models_root / _REPO / _FILE).write_bytes(b"x" * 10)
    (models_root / "loose.gguf").write_bytes(b"x")
    [entry] = await (await ImageModelStore.open(db)).catalog_entries()
    assert entry.installed and entry.tag == _TAG


@pytest.mark.asyncio
async def test_catalog_adds_admin_added_and_otherwise_installed_models_after_the_curated_ones(db, models_root):
    added = "hf.co/second-state/stable-diffusion-v1-5-GGUF:sd-v1-5-Q8_0"
    await _register(db, added)
    other = models_root / "org/Other-GGUF/other.gguf"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"x")

    entries = await (await ImageModelStore.open(db)).catalog_entries(is_admin=True)

    assert [e.tag for e in entries] == [_TAG, added, "hf.co/org/Other-GGUF:other"]
    by_tag = {e.tag: e for e in entries}
    assert by_tag[added].removable and not by_tag[added].installed  # an admin may drop a not-installed addition
    assert not by_tag["hf.co/org/Other-GGUF:other"].removable and by_tag["hf.co/org/Other-GGUF:other"].installed
    assert not (await (await ImageModelStore.open(db)).catalog_entries())[1].removable  # not for a non-admin


def test_is_image_repo_detection():
    detect = HuggingFaceCatalogSearch.is_image_repo
    assert detect({"pipeline_tag": "text-to-image"})
    assert detect({"pipeline_tag": None, "tags": ["gguf", "sd.cpp"]})
    assert not detect({"pipeline_tag": "image-text-to-text", "tags": ["gguf"]})  # a vision *chat* model
    assert not detect({})


@pytest.mark.asyncio
async def test_adding_an_image_repo_file_stores_an_image_entry_without_probing_it(db, monkeypatch):
    repo = {
        "repo_id": _REPO,
        "is_image": True,
        "parameter_size": None,
        "files": [{"filename": _FILE, "download_gb": 2.0}],
    }

    async def fake_repo_files(_repo_id, _proxy):
        return repo

    async def fake_list_models():
        return []

    monkeypatch.setattr(HuggingFaceCatalogSearch, "repo_files", staticmethod(fake_repo_files))
    from app.services import extended_model_catalog_service as svc

    monkeypatch.setattr(svc, "list_models", fake_list_models)

    entry = await ExtendedModelCatalog(db).add(_REPO, _FILE, None)

    assert entry["kind"] == "image" and entry["tag"] == _TAG and entry["family"] == "SD-Turbo"


@pytest.mark.asyncio
async def test_image_entries_never_appear_in_the_chat_browse_more_list(db, user, monkeypatch):
    from app.services import extended_model_catalog_service as svc

    async def fake_list_models():
        return []

    monkeypatch.setattr(svc, "list_models", fake_list_models)
    catalog = ExtendedModelCatalog(db)
    await catalog._save(
        [
            {"tag": _TAG, "kind": "image", "family": "SD-Turbo", "download_gb": 2.0},
            {"tag": "hf.co/org/Chat-GGUF:chat", "family": "Chat", "download_gb": 1.0, "architecture": "llama"},
        ]
    )
    assert [e.tag for e in await catalog.build(user)] == ["hf.co/org/Chat-GGUF:chat"]


@pytest.mark.asyncio
async def test_clear_keeps_image_entries(db):
    await _register(db)
    await ExtendedModelCatalog(db).clear()
    assert [e["tag"] for e in await ExtendedModelCatalog(db).list()] == [_TAG]


@pytest.mark.asyncio
async def test_router_list_pull_and_delete(db, models_root, monkeypatch):
    await _register(db)
    _fake_hf(monkeypatch)

    catalog = await image_model_admin.get_diffusion_model_catalog(db=db, user=SimpleNamespace(role="admin"))
    assert catalog.local is True and [e.tag for e in catalog.entries] == [_TAG]
    listing = await image_model_admin.list_image_models(db=db, _admin=None)
    assert listing.models == [] and listing.local is True

    response = await image_model_admin.pull_image_model(ImageModelRequest(tag=_TAG), db=db, _admin=None)
    frames = [chunk async for chunk in response.body_iterator]
    assert '"done": true' in frames[-1]
    assert len((await image_model_admin.list_image_models(db=db, _admin=None)).models) == 1

    assert (await image_model_admin.delete_image_model(ImageModelRequest(tag=_TAG), db=db, _admin=None)).ok is True
    assert (await image_model_admin.list_image_models(db=db, _admin=None)).models == []
    assert await ExtendedModelCatalog(db).list() == []


@pytest.mark.asyncio
async def test_router_pull_maps_validation_errors_to_400(db):
    with pytest.raises(HTTPException) as exc_info:
        await image_model_admin.pull_image_model(ImageModelRequest(tag="llama3:latest"), db=db, _admin=None)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_models_dir_defaults_to_a_stable_diffusion_sibling_of_the_chat_engine_folders(db, monkeypatch):
    from app.schemas import MatricxonServerConfig, OllamaServerConfig
    from app.services import settings_service

    monkeypatch.undo()  # this test exercises the real resolver, not the autouse stub
    assert await image_model_service.resolve_models_dir(db) == image_model_service.sdcpp_installer.models_dir()

    await settings_service.set_ollama_server_config(db, OllamaServerConfig(models_path="/data/models/ollama"))
    assert await image_model_service.resolve_models_dir(db) == Path("/data/models/diffusion")

    await settings_service.set_matricxon_server_config(db, MatricxonServerConfig(models_path="/mnt/models/matricxon"))
    assert await image_model_service.resolve_models_dir(db) == Path("/mnt/models/diffusion")  # Matricxon wins

    await settings_service.set_matricxon_server_config(db, MatricxonServerConfig(models_path="/mnt/custom-dir"))
    await settings_service.set_ollama_server_config(db, OllamaServerConfig(models_path="/data/other"))
    assert await image_model_service.resolve_models_dir(db) == image_model_service.sdcpp_installer.models_dir()

    await image_engine_service.set_sdcpp_config(db, SdCppConfig(models_path="/explicit/place"))
    assert await image_model_service.resolve_models_dir(db) == Path("/explicit/place")


@pytest.mark.asyncio
async def test_router_catalog_reports_a_remote_engine_as_not_local(db):
    await image_engine_service.set_sdcpp_config(db, SdCppConfig(mode="remote", remote_hosts=["http://h:1"]))
    catalog = await image_model_admin.get_diffusion_model_catalog(db=db, user=SimpleNamespace(role="user"))
    assert catalog.local is False
