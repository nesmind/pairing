"""Unit tests for app/services/db_schema_check.py: classifying a
candidate database's existing tables (if any) before build_schema or
save_and_apply ever runs DDL against it — see that module's own
docstring for why a raw connection succeeding isn't enough on its own."""

import sqlite3

import pytest

from app.services.db_schema_check import IncompatibleSchemaError, classify_schema, connection_result_with_schema_status


def test_classify_schema_empty_database(tmp_path):
    db_path = tmp_path / "empty.db"
    sqlite3.connect(str(db_path)).close()
    assert classify_schema(f"sqlite:///{db_path}") == "empty"


def test_classify_schema_recognizes_a_real_pAIring_schema(tmp_path):
    from app.database import _run_migrations

    db_path = tmp_path / "real.db"
    url = f"sqlite:///{db_path}"
    _run_migrations(url)
    assert classify_schema(url) == "compatible"


def test_classify_schema_recognizes_a_legacy_pre_alembic_schema(tmp_path):
    """A pAIring schema that predates Alembic entirely (no
    alembic_version table, but real pAIring tables) must still classify
    as compatible — this is exactly the case app.database._run_migrations
    itself already knows how to stamp-and-upgrade rather than treat as
    foreign."""
    from app.database import _run_migrations

    db_path = tmp_path / "legacy.db"
    url = f"sqlite:///{db_path}"
    _run_migrations(url)  # creates the schema, including alembic_version
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("DROP TABLE alembic_version")
        conn.commit()
    finally:
        conn.close()
    assert classify_schema(url) == "compatible"


def test_classify_schema_rejects_a_database_with_unrelated_tables(tmp_path):
    db_path = tmp_path / "foreign.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE wordpress_posts (id INTEGER PRIMARY KEY, title TEXT)")
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(IncompatibleSchemaError, match="wordpress_posts"):
        classify_schema(f"sqlite:///{db_path}")


def test_classify_schema_rejects_a_pAIring_schema_with_one_extra_foreign_table(tmp_path):
    """Even mostly-compatible isn't good enough — a single table pAIring
    doesn't recognize alongside its own real tables still means this
    database is shared with something else, which is exactly the
    situation this check exists to catch."""
    from app.database import _run_migrations

    db_path = tmp_path / "mixed.db"
    url = f"sqlite:///{db_path}"
    _run_migrations(url)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE some_other_apps_table (id INTEGER PRIMARY KEY)")
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(IncompatibleSchemaError, match="some_other_apps_table"):
        classify_schema(url)


def test_connection_result_with_schema_status_reports_empty():
    result = connection_result_with_schema_status("sqlite:///:memory:", "Connected successfully.")
    assert result.ok is True
    assert "Empty database" in result.message


def test_connection_result_with_schema_status_reports_incompatible(tmp_path):
    db_path = tmp_path / "foreign.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE other_app_table (id INTEGER PRIMARY KEY)")
        conn.commit()
    finally:
        conn.close()

    result = connection_result_with_schema_status(f"sqlite:///{db_path}", "Connected successfully.")
    assert result.ok is False
    assert "other_app_table" in result.message
    assert result.message.startswith("Connected successfully.")
