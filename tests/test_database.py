"""Unit tests for app/database.py's schema_is_current/init_db gating on
app.config.AUTO_MIGRATE — see that flag's own docstring for why it
exists: multiple app instances racing to migrate the same database
concurrently on cold start can fail outright (confirmed live, not
theoretical), so a multi-instance deployment needs to be able to turn
automatic migration off and rely on a separate, one-time
`alembic upgrade head` step instead."""

import sqlite3

import pytest

from app import database


def test_schema_is_current_false_for_an_empty_database(tmp_path):
    db_path = tmp_path / "empty.db"
    sqlite3.connect(str(db_path)).close()
    assert database.schema_is_current(f"sqlite:///{db_path}") is False


def test_schema_is_current_true_after_running_migrations(tmp_path):
    db_path = tmp_path / "migrated.db"
    url = f"sqlite:///{db_path}"
    database._run_migrations(url)
    assert database.schema_is_current(url) is True


def test_schema_is_current_false_when_stamped_at_an_older_revision(tmp_path):
    """A database that's real and Alembic-managed, just not on the
    newest revision yet (e.g. an app instance updated to newer code
    before the migration step ran) must not be mistaken for current."""
    from alembic.config import Config

    from alembic import command

    db_path = tmp_path / "stale.db"
    url = f"sqlite:///{db_path}"

    from app.config import BASE_DIR

    cfg = Config(str(BASE_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BASE_DIR / "alembic"))
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, database._BASELINE_REVISION)

    assert database.schema_is_current(url) is False


@pytest.mark.asyncio
async def test_init_db_auto_migrates_when_auto_migrate_is_on(tmp_path, monkeypatch):
    db_path = tmp_path / "fresh.db"
    url = f"sqlite:///{db_path}"
    monkeypatch.setattr(database, "AUTO_MIGRATE", True)
    monkeypatch.setattr(database, "DATABASE_URL", url)

    await database.init_db()

    assert database.schema_is_current(url) is True


@pytest.mark.asyncio
async def test_init_db_refuses_to_start_when_auto_migrate_is_off_and_schema_is_stale(tmp_path, monkeypatch):
    db_path = tmp_path / "not_migrated.db"
    url = f"sqlite:///{db_path}"
    sqlite3.connect(str(db_path)).close()  # exists, but no schema at all
    monkeypatch.setattr(database, "AUTO_MIGRATE", False)
    monkeypatch.setattr(database, "DATABASE_URL", url)

    with pytest.raises(database.SchemaNotCurrentError, match="alembic upgrade head"):
        await database.init_db()


@pytest.mark.asyncio
async def test_init_db_starts_cleanly_when_auto_migrate_is_off_and_schema_is_already_current(tmp_path, monkeypatch):
    db_path = tmp_path / "pre_migrated.db"
    url = f"sqlite:///{db_path}"
    database._run_migrations(url)  # simulates the separate "migrate once" step

    monkeypatch.setattr(database, "AUTO_MIGRATE", False)
    monkeypatch.setattr(database, "DATABASE_URL", url)

    calls = []
    monkeypatch.setattr(database, "_run_migrations", lambda *a, **k: calls.append((a, k)))

    await database.init_db()

    assert calls == []  # never attempted a migration -- AUTO_MIGRATE was off
