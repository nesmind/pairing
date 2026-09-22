"""Unit tests for app/services/document_extract.py's pure text-chunking
logic — no filesystem, no Ollama, no database."""

from app.services.document_extract import chunk_text, hash_bytes


def test_chunk_text_empty_input_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\t  ") == []


def test_chunk_text_short_text_is_a_single_chunk():
    assert chunk_text("hello world", size=100, overlap=10) == ["hello world"]


def test_chunk_text_splits_long_text_with_overlap():
    text = "abcdefghij" * 10  # 100 chars
    chunks = chunk_text(text, size=30, overlap=5)

    assert len(chunks) > 1
    # Every chunk after the first repeats the previous chunk's last
    # `overlap` characters at its own start — that's the whole point of
    # overlap (an idea split across a boundary still appears whole
    # somewhere).
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev[-5:] == nxt[:5]
    # Reassembling without the overlap should reproduce the original.
    assert chunks[0] + "".join(c[5:] for c in chunks[1:]) == text


def test_chunk_text_strips_surrounding_whitespace():
    assert chunk_text("  hello  ", size=100, overlap=10) == ["hello"]


def test_hash_bytes_is_deterministic_and_sensitive_to_content():
    a = hash_bytes(b"same content")
    b = hash_bytes(b"same content")
    c = hash_bytes(b"different content")
    assert a == b
    assert a != c
    assert len(a) == 64  # sha256 hex digest length
