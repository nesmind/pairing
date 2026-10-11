"""Shared helpers for ORM models: id generation, id-column length and a UTC timestamp."""

import uuid
from datetime import UTC, datetime


def new_id() -> str:
    """Short, URL-safe unique primary key (safe to expose, no collisions across merged databases)."""
    return uuid.uuid4().hex


# Length of every generated id. MySQL needs an explicit VARCHAR length (SQLite ignores it);
# exact lengths keep indexes small.
ID_LEN = 32


def utcnow() -> datetime:
    """UTC timestamp, independent of the server's timezone."""
    return datetime.now(UTC)
