"""Schema-compatibility check for Settings > System > Database (split from db_config_service.py for the
file-size rule). Read-only: lists a candidate database's existing tables, no DDL.

A connection can succeed against a database with unrelated tables (another app, a stale export, a typo'd
name). build_schema would then hand it to Alembic, whose pre-Alembic heuristic
(app.database._run_migrations) takes any tables for an old pAIring install and just *stamps* it migrated
without creating pAIring's tables: the app starts, then 500s on every request, with an unexplained admin
lockout. This read-only check catches that early.
"""

from sqlalchemy import create_engine, inspect

from app.config import APP_NAME
from app.database import Base
from app.schemas import DbActionResult


class IncompatibleSchemaError(Exception):
    """Raised by classify_schema when the database has a table pAIring doesn't recognize; it blocks every write action
    rather than warn."""


def classify_schema(url: str) -> str:
    """ "empty" (no tables; safe to build) or "compatible" (every table is one of pAIring's models or
    alembic_version: current, older or legacy pre-Alembic). Raises IncompatibleSchemaError otherwise.
    """
    from app import models  # noqa: F401 -- register every table before reading Base.metadata

    engine = create_engine(url)
    try:
        existing_tables = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()

    if not existing_tables:
        return "empty"

    known_tables = set(Base.metadata.tables.keys()) | {"alembic_version"}
    foreign_tables = existing_tables - known_tables
    if foreign_tables:
        names = ", ".join(sorted(foreign_tables)[:5])
        raise IncompatibleSchemaError(
            f"This database already has table(s) {APP_NAME} doesn't recognize ({names}) — "
            f"point at an empty database, or one already running {APP_NAME}."
        )
    return "compatible"


def connection_result_with_schema_status(url: str, connected_message: str) -> DbActionResult:
    """Used by db_config_service's _test_sqlite/_test_mysql after a successful connection: connectivity alone
    isn't "OK", so this adds the schema verdict before build_schema/save.
    """
    try:
        status = classify_schema(url)
    except IncompatibleSchemaError as exc:
        return DbActionResult(ok=False, message=f"{connected_message} {exc}")
    suffix = (
        f"Empty database — ready for a fresh {APP_NAME} schema."
        if status == "empty"
        else f"Existing {APP_NAME} schema detected — compatible."
    )
    return DbActionResult(ok=True, message=f"{connected_message} {suffix}")
