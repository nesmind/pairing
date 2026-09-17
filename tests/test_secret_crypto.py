"""Unit tests for app/services/secret_crypto.py."""

from app.services import secret_crypto


def test_encrypt_then_decrypt_round_trips():
    ciphertext = secret_crypto.encrypt("s3cret")
    assert ciphertext != "s3cret"
    assert secret_crypto.decrypt(ciphertext) == "s3cret"


def test_encrypt_output_is_not_plaintext_and_looks_like_a_fernet_token():
    ciphertext = secret_crypto.encrypt("hunter2")
    assert "hunter2" not in ciphertext


def test_decrypt_of_garbage_returns_none_instead_of_raising():
    assert secret_crypto.decrypt("not-a-real-fernet-token") is None
