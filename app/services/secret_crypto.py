"""
Symmetric encryption for secrets this app stores in its own database (as
opposed to .env — see the note below on why an .env-stored secret can't
be meaningfully protected this same way). Currently used for
app.schemas.http_proxy_config.HttpProxyConfig's own password field: the
key (derived from app.config.SECRET_KEY) lives in .env, while the
ciphertext lives in the app_settings table — two genuinely separate
storage locations/access levels, so this provides real defense in depth
(e.g. against someone with DB read access but not filesystem access).

Deliberately NOT used for the MySQL database password (see
app.services.db_config_url) — that password lives *inside* DATABASE_URL,
itself stored in .env, the same file SECRET_KEY comes from. Encrypting
a value with a key stored right next to it in the same file protects
against nothing; anyone who can read one can read the other. Real
protection there would need the key to come from somewhere else
entirely (an OS keyring, a mounted secret, ...) — real infrastructure
work outside this module's scope, not something to fake with a
same-file key.

Never used for user login passwords: those are one-way bcrypt hashes
(see app.services.auth_service), which is strictly stronger than
reversible encryption for anything only ever *verified*, never
resubmitted elsewhere — encrypting an already-hashed value would add
nothing.
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
    """Fernet needs an exact 32-byte url-safe-base64 key; SECRET_KEY is
    an arbitrary-length string (a random hex token by default, or
    whatever an admin sets) — hashed down to exactly 32 bytes first.
    Cached (not re-derived per call) since SECRET_KEY is fixed for this
    process's lifetime, same reasoning as any other derive-once value.

    If SECRET_KEY isn't pinned in .env, it's regenerated randomly every
    restart (see app.config's own warning) — anything encrypted under
    the old key becomes undecryptable, the exact same accepted tradeoff
    that already applies to session cookies for the same reason."""
    key_bytes = hashlib.sha256(SECRET_KEY.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(key_bytes))


def encrypt(plaintext: str) -> str:
    return _fernet().encrypt(plaintext.encode("utf-8")).decode("utf-8")


def decrypt(ciphertext: str) -> str | None:
    """None (rather than raising) on a token that won't decrypt under
    the current key — SECRET_KEY having changed since this was encrypted
    is a real, expected possibility (see _fernet's own docstring), and a
    corrupt/foreign value should degrade to "treat as unset", not crash
    whatever was loading this config."""
    try:
        return _fernet().decrypt(ciphertext.encode("utf-8")).decode("utf-8")
    except InvalidToken:
        logger.warning("Could not decrypt a stored secret (SECRET_KEY changed since it was saved?) — treating as unset")
        return None
