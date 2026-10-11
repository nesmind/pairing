"""Tests for services/db_schema_check.py: classifying a candidate database's existing tables before build_schema or
save_and_apply run DDL (a successful raw connection isn't enough; see the module).
"""

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
    """A pre-Alembic pAIring schema (real tables, no alembic_version) must classify as compatible: the case
    app.database._run_migrations stamps and upgrades.
    """
    from app.database import _run_migrations

    db_path = tmp_path / "legacy.db"
    url = f"sqlite:///{db_path}"
    _run_migrations(url)
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
    """Mostly compatible isn't enough: one unrecognized table beside pAIring's own means the database is shared with
    something else."""
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
