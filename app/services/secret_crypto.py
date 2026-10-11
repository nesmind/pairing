"""Symmetric encryption for secrets stored in the app's own database (not .env, where a secret can't be
protected this way). Used for HttpProxyConfig's password: the key (derived from SECRET_KEY) lives in .env and
the ciphertext in app_settings, two storage locations, giving defense in depth (e.g. DB read access without
filesystem access).

Not used for the MySQL password (db_config_url): it sits inside DATABASE_URL in .env, the same file as
SECRET_KEY, and encrypting with a key beside the value protects nothing. Real protection needs the key
elsewhere (OS keyring, mounted secret), out of scope.

Not used for login passwords: those are one-way bcrypt hashes (auth_service), stronger than reversible
encryption for values only verified, never resubmitted.
"""

import base64
import hashlib
import logging
from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken

from app.config import SECRET_KEY

logger = logging.getLogger("llama_chat")


@lru_cache(maxsize=1)
def _fernet() -> Fernet:
    """Fernet needs a 32-byte url-safe-base64 key; SECRET_KEY is arbitrary-length (random hex by default), hashed
    to 32 bytes. Cached: SECRET_KEY is fixed per process.

    If SECRET_KEY isn't pinned in .env it's regenerated every restart (see app.config's warning), and anything
    encrypted under the old key becomes undecryptable: the same accepted tradeoff as session cookies.
    """
    key_bytes = hashlib.sha256(SECRET_KEY.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(key_bytes))


def encrypt(plaintext: str) -> str:
    return _fernet().encrypt(plaintext.encode("utf-8")).decode("utf-8")


def decrypt(ciphertext: str) -> str | None:
    """None (not an exception) for a token that won't decrypt under the current key: SECRET_KEY may have changed
    (see _fernet), and a corrupt or foreign value should read as unset, not crash the config load.
    """
    try:
        return _fernet().decrypt(ciphertext.encode("utf-8")).decode("utf-8")
    except InvalidToken:
        logger.warning("Could not decrypt a stored secret (SECRET_KEY changed since it was saved?) — treating as unset")
        return None
