"""Unit tests for app/services/chat_attachment_service.py: attachment
validation/classification/storage (including the multi-file per-type
caps), the text-extraction fold-in, and the image-attachment application
reply_generation_service ends up using (see tests/test_chat_service.py
for how build_reply_stream wires all three together)."""

import base64
import io

import pytest
from fastapi import HTTPException, UploadFile

from app.config import MAX_ATTACHMENT_DOCUMENTS, MAX_ATTACHMENT_IMAGES, MAX_ATTACHMENT_MB
from app.services import chat_attachment_service, model_catalog_service
from app.services.chat_attachment_service import AttachmentInfo


def _upload(filename: str, data: bytes) -> UploadFile:
    return UploadFile(io.BytesIO(data), filename=filename)


@pytest.fixture(autouse=True)
def _isolated_attachments_dir(tmp_path, monkeypatch):
    """save_attachments() writes through app.config.ATTACHMENTS_DIR
    directly (imported by name into chat_attachment_service, same
    module-split import-binding gotcha as elsewhere in this codebase) —
    without this, every test below would write real files into the
    project's own attachments/ folder instead of a throwaway tmp_path."""
    monkeypatch.setattr(chat_attachment_service, "ATTACHMENTS_DIR", tmp_path)


@pytest.mark.asyncio
async def test_save_attachments_classifies_an_image_extension_as_image():
    [info] = await chat_attachment_service.save_attachments([_upload("photo.PNG", b"fake-bytes")], "conv-1")
    assert info.type == "image"
    assert info.filename == "photo.PNG"
    assert info.path.read_bytes() == b"fake-bytes"


@pytest.mark.asyncio
async def test_save_attachments_classifies_a_document_extension_as_text():
    [info] = await chat_attachment_service.save_attachments([_upload("notes.pdf", b"%PDF-fake")], "conv-1")
    assert info.type == "text"


@pytest.mark.asyncio
async def test_save_attachments_rejects_an_unsupported_extension():
    with pytest.raises(HTTPException) as exc_info:
        await chat_attachment_service.save_attachments([_upload("virus.exe", b"x")], "conv-1")
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_save_attachments_rejects_a_file_over_the_size_limit():
    oversized = b"x" * (MAX_ATTACHMENT_MB * 1024 * 1024 + 1)
    with pytest.raises(HTTPException) as exc_info:
        await chat_attachment_service.save_attachments([_upload("big.txt", oversized)], "conv-1")
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_save_attachments_writes_two_same_named_files_without_colliding():
    """Two attachments in the same conversation with the same original
    filename must not overwrite each other — see save_attachments' own
    comment on the random-token prefix."""
    first, second = await chat_attachment_service.save_attachments(
        [_upload("same.txt", b"first"), _upload("same2.txt", b"second")], "conv-1"
    )
    assert first.path != second.path
    assert first.path.read_bytes() == b"first"
    assert second.path.read_bytes() == b"second"


@pytest.mark.asyncio
async def test_save_attachments_accepts_up_to_the_document_and_image_limits_combined():
    uploads = [_upload(f"doc{i}.txt", b"x") for i in range(MAX_ATTACHMENT_DOCUMENTS)] + [
        _upload("photo.png", b"img") for _ in range(MAX_ATTACHMENT_IMAGES)
    ]
    infos = await chat_attachment_service.save_attachments(uploads, "conv-1")
    assert len(infos) == MAX_ATTACHMENT_DOCUMENTS + MAX_ATTACHMENT_IMAGES


@pytest.mark.asyncio
async def test_save_attachments_rejects_too_many_documents_without_writing_any():
    uploads = [_upload(f"doc{i}.txt", b"x") for i in range(MAX_ATTACHMENT_DOCUMENTS + 1)]
    with pytest.raises(HTTPException) as exc_info:
        await chat_attachment_service.save_attachments(uploads, "conv-1")
    assert exc_info.value.status_code == 400
    assert str(MAX_ATTACHMENT_DOCUMENTS) in exc_info.value.detail
    assert list((chat_attachment_service.ATTACHMENTS_DIR / "conv-1").glob("*")) == []


@pytest.mark.asyncio
async def test_save_attachments_rejects_too_many_images_without_writing_any():
    uploads = [_upload(f"photo{i}.png", b"x") for i in range(MAX_ATTACHMENT_IMAGES + 1)]
    with pytest.raises(HTTPException) as exc_info:
        await chat_attachment_service.save_attachments(uploads, "conv-1")
    assert exc_info.value.status_code == 400
    assert str(MAX_ATTACHMENT_IMAGES) in exc_info.value.detail
    assert list((chat_attachment_service.ATTACHMENTS_DIR / "conv-1").glob("*")) == []


def test_extract_attachment_text_reuses_document_extract(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello from disk")
    assert chat_attachment_service.extract_attachment_text(path) == "hello from disk"


def test_fold_text_attachments_into_prompt_is_a_noop_for_an_image(tmp_path):
    path = tmp_path / "photo.png"
    path.write_bytes(b"fake")
    attachment = AttachmentInfo(path=path, filename="photo.png", type="image")
    assert chat_attachment_service.fold_text_attachments_into_prompt("base", [attachment]) == "base"


def test_fold_text_attachments_into_prompt_appends_every_files_content(tmp_path):
    path1 = tmp_path / "notes.txt"
    path1.write_text("the secret is 42")
    path2 = tmp_path / "more.txt"
    path2.write_text("and the sky is blue")
    attachments = [
        AttachmentInfo(path=path1, filename="notes.txt", type="text"),
        AttachmentInfo(path=path2, filename="more.txt", type="text"),
    ]
    prompt = chat_attachment_service.fold_text_attachments_into_prompt("base", attachments)
    assert "base" in prompt
    assert "notes.txt" in prompt
    assert "the secret is 42" in prompt
    assert "more.txt" in prompt
    assert "and the sky is blue" in prompt


@pytest.mark.asyncio
async def test_apply_image_attachment_is_a_noop_for_text(db, tmp_path, monkeypatch):
    def fail_if_called(_db):
        raise AssertionError("get_default_vision_model must not be called for a text attachment")

    monkeypatch.setattr(model_catalog_service, "get_default_vision_model", fail_if_called)

    path = tmp_path / "notes.txt"
    path.write_text("hi")
    attachment = AttachmentInfo(path=path, filename="notes.txt", type="text")
    ollama_messages = [{"role": "user", "content": "hi"}]

    result = await chat_attachment_service.apply_image_attachment(db, [attachment], ollama_messages)

    assert result is None
    assert ollama_messages == [{"role": "user", "content": "hi"}]  # untouched


@pytest.mark.asyncio
async def test_apply_image_attachment_returns_none_when_no_vision_model_installed(db, tmp_path, monkeypatch):
    async def fake_get_default_vision_model(_db):
        return None

    monkeypatch.setattr(model_catalog_service, "get_default_vision_model", fake_get_default_vision_model)

    path = tmp_path / "photo.png"
    path.write_bytes(b"fake-image-bytes")
    attachment = AttachmentInfo(path=path, filename="photo.png", type="image")
    ollama_messages = [{"role": "user", "content": "look at this"}]

    result = await chat_attachment_service.apply_image_attachment(db, [attachment], ollama_messages)

    assert result is None
    assert "images" not in ollama_messages[-1]


@pytest.mark.asyncio
async def test_apply_image_attachment_attaches_base64_bytes_to_the_last_message(db, tmp_path, monkeypatch):
    async def fake_get_default_vision_model(_db):
        return "llava:latest"

    monkeypatch.setattr(model_catalog_service, "get_default_vision_model", fake_get_default_vision_model)

    path = tmp_path / "photo.png"
    path.write_bytes(b"fake-image-bytes")
    doc_path = tmp_path / "notes.txt"
    doc_path.write_text("hi")
    attachments = [
        AttachmentInfo(path=doc_path, filename="notes.txt", type="text"),
        AttachmentInfo(path=path, filename="photo.png", type="image"),
    ]
    ollama_messages = [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "look at this"},
    ]

    result = await chat_attachment_service.apply_image_attachment(db, attachments, ollama_messages)

    assert result == "llava:latest"
    # Only the last entry (the user's own turn) is touched.
    assert ollama_messages[0] == {"role": "system", "content": "be nice"}
    assert ollama_messages[-1]["role"] == "user"
    assert ollama_messages[-1]["content"] == "look at this"
    assert base64.b64decode(ollama_messages[-1]["images"][0]) == b"fake-image-bytes"
