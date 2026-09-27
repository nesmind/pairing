"""Unit tests for app/services/inference_client.py — the engine-agnostic dispatch layer every generic call site
(chat, RAG, titles, model management) goes through instead of importing app.services.ollama_client or
app.services.matricxon_client directly. Confirms dispatch actually follows app.services.engine_service.
current_engine(), and that both engines' own exception types get normalized into one InferenceError."""

import pytest

from app.services import (
    engine_service,
    inference_client,
    matricxon_admin,
    matricxon_client,
    ollama_admin,
    ollama_client,
)


@pytest.fixture(autouse=True)
def reset_cache():
    engine_service._cached_engine = engine_service.DEFAULT_ENGINE
    yield
    engine_service._cached_engine = engine_service.DEFAULT_ENGINE


@pytest.mark.asyncio
async def test_list_models_dispatches_to_ollama_when_active(monkeypatch):
    engine_service._cached_engine = "ollama"

    async def fake_list_models():
        return [{"name": "from-ollama"}]

    monkeypatch.setattr(ollama_client, "list_models", fake_list_models)
    monkeypatch.setattr(matricxon_client, "list_models", lambda: pytest.fail("must not call matricxon"))

    assert await inference_client.list_models() == [{"name": "from-ollama"}]


@pytest.mark.asyncio
async def test_list_models_dispatches_to_matricxon_when_active(monkeypatch):
    engine_service._cached_engine = "matricxon"

    async def fake_list_models():
        return [{"name": "from-matricxon"}]

    monkeypatch.setattr(matricxon_client, "list_models", fake_list_models)
    assert await inference_client.list_models() == [{"name": "from-matricxon"}]


@pytest.mark.asyncio
async def test_list_models_wraps_ollama_error_as_inference_error(monkeypatch):
    engine_service._cached_engine = "ollama"

    async def fake_list_models():
        raise ollama_client.OllamaError("down")

    monkeypatch.setattr(ollama_client, "list_models", fake_list_models)

    with pytest.raises(inference_client.InferenceError):
        await inference_client.list_models()


@pytest.mark.asyncio
async def test_list_models_wraps_matricxon_error_as_inference_error(monkeypatch):
    engine_service._cached_engine = "matricxon"

    async def fake_list_models():
        raise matricxon_client.MatricxonError("down")

    monkeypatch.setattr(matricxon_client, "list_models", fake_list_models)
    with pytest.raises(inference_client.InferenceError):
        await inference_client.list_models()


@pytest.mark.asyncio
async def test_chat_stream_dispatches_by_active_engine(monkeypatch):
    engine_service._cached_engine = "ollama"

    async def fake_ollama_stream(_model, _messages, _params):
        yield "from-ollama"

    monkeypatch.setattr(ollama_client, "chat_stream", fake_ollama_stream)
    result = [c async for c in inference_client.chat_stream("m", [], {})]
    assert result == ["from-ollama"]


@pytest.mark.asyncio
async def test_pull_model_stream_dispatches_to_the_active_engines_admin_service(monkeypatch):
    engine_service._cached_engine = "ollama"

    async def fake_pull(tag):
        yield {"status": f"pulling {tag} via ollama"}

    monkeypatch.setattr(ollama_admin, "pull_model_stream", fake_pull)
    result = [p async for p in inference_client.pull_model_stream("hf.co/some/repo:some-file")]
    assert result == [{"status": "pulling hf.co/some/repo:some-file via ollama"}]


@pytest.mark.asyncio
async def test_pull_model_stream_rejects_a_non_hf_tag_for_ollama(monkeypatch):
    """The real gap this closes: Ollama (unlike Matricxon, which has no access to Ollama's own registry protocol
    at all) will happily pull a bare library name like "llama3" — a tag Matricxon could never resolve, so it
    would silently become unusable the moment an admin switched the active engine."""
    engine_service._cached_engine = "ollama"
    monkeypatch.setattr(ollama_admin, "pull_model_stream", lambda _tag: pytest.fail("must not reach the engine"))

    with pytest.raises(inference_client.InferenceError):
        async for _ in inference_client.pull_model_stream("llama3"):
            pass


@pytest.mark.asyncio
async def test_pull_model_stream_rejects_a_non_hf_tag_for_matricxon(monkeypatch):
    engine_service._cached_engine = "matricxon"
    monkeypatch.setattr(matricxon_admin, "pull_model_stream", lambda _tag: pytest.fail("must not reach the engine"))

    with pytest.raises(inference_client.InferenceError):
        async for _ in inference_client.pull_model_stream("some-bare-tag"):
            pass


@pytest.mark.asyncio
async def test_delete_model_dispatches_to_matricxon_admin_when_active(monkeypatch):
    engine_service._cached_engine = "matricxon"
    calls = []

    async def fake_delete(tag):
        calls.append(tag)

    monkeypatch.setattr(matricxon_admin, "delete_model", fake_delete)
    await inference_client.delete_model("some-tag")
    assert calls == ["some-tag"]


def test_active_capabilities_reflects_the_active_engine():
    engine_service._cached_engine = "ollama"
    assert inference_client.active_capabilities().support_checking is False

    engine_service._cached_engine = "matricxon"
    assert inference_client.active_capabilities().support_checking is True


@pytest.mark.asyncio
async def test_embed_raises_inference_error_when_the_active_engine_cannot_embed(monkeypatch):
    """A future engine without real embedding support (see InferenceEngine.embed's own default) must fail the
    same clean way an unreachable engine already does — one InferenceError, not an AttributeError or a silent
    no-op — so a caller only ever needs one except clause regardless of *why* the call couldn't go through."""
    from app.services.engines.base import EngineCapabilities, InferenceEngine
    from app.services.engines.registry import EngineRegistry

    class _NoEmbeddingsEngine(InferenceEngine):
        name = "ollama"
        display_name = "No-Embeddings Test Engine"
        capabilities = EngineCapabilities(
            embeddings=False,
            model_management=False,
            stop_model=False,
            local_process=False,
            host_pool=False,
            support_checking=False,
            format_introspection=False,
        )
        error_types = ()

        async def list_models(self):
            return []

        def chat_stream(self, model, messages, params):
            async def _gen():
                return
                yield

            return _gen()

        async def chat_once(self, model, messages, params=None):
            return ""

        async def check_health(self):
            return {}

    fake_registry = EngineRegistry([_NoEmbeddingsEngine()])
    monkeypatch.setattr(inference_client, "registry", fake_registry)
    engine_service._cached_engine = "ollama"

    with pytest.raises(inference_client.InferenceError):
        await inference_client.embed("some text", "some-model")


@pytest.mark.asyncio
async def test_stop_model_does_not_wrap_errors(monkeypatch):
    # stop_model is cleanup/best-effort on both engines (see ollama_client.stop_model/matricxon_client.stop_model's
    # own docstrings — they already swallow failures internally), so the dispatcher passes it through untouched.
    engine_service._cached_engine = "ollama"
    called = []

    async def fake_stop(model):
        called.append(model)

    monkeypatch.setattr(ollama_client, "stop_model", fake_stop)
    await inference_client.stop_model("some-model")
    assert called == ["some-model"]
