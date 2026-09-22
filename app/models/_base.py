"""
Shared helpers every ORM model file in this package uses: id generation,
the fixed id-column length, and a UTC timestamp helper. Split out of the
old single models.py so each domain file only imports what it needs.
"""

import uuid
from datetime import UTC, datetime


def new_id() -> str:
    """Generates a short, URL-safe unique id used as a primary key.
    UUIDs (instead of auto-increment integers) mean ids are safe to
    expose in URLs and never collide even if databases are later merged."""
    return uuid.uuid4().hex


# Every id this app generates is exactly this long (see new_id above).
# SQLite treats a bare `String` as unbounded TEXT and doesn't care, but
# MySQL's VARCHAR requires an explicit length on every text column — an
# unbounded one fails at CREATE TABLE time, not silently. Giving id/enum
# columns their real length here (rather than a generic default like 255
# everywhere) keeps a future MySQL install's indexes small and correct
# without changing anything about how SQLite already behaves.
ID_LEN = 32


def utcnow() -> datetime:
    """Timestamp helper so every row uses the same UTC clock, regardless
    of the server's local timezone — avoids subtly wrong "sort by date"
    behavior if the app is later moved to a server in another timezone."""
    return datetime.now(UTC)
