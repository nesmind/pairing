"""
Schema-compatibility check for Settings > System > Database (see
app/services/db_config_service.py, which this is split out of purely to
stay under CLAUDE.md's file-size rule). Read-only: lists whatever tables
already exist at a candidate database, without ever running DDL.

A raw connection can succeed against a database that already has
unrelated tables in it — someone else's app, a stale export, a typo'd
database name that happens to exist. Without this check, build_schema
would hand that straight to Alembic, whose own "is this a pre-Alembic
pAIring install?" heuristic (see app.database._run_migrations) would
wrongly assume any pre-existing tables are exactly that and just *stamp*
the database as already migrated — without ever creating pAIring's own
tables. The result: an app that starts up fine and then 500s on every
request, an admin lockout with no obvious cause. Catching this early,
while it's still just a read-only "list the tables" check, is the whole
point of this module.
"""

from sqlalchemy import create_engine, inspect

from app.config import APP_NAME
from app.database import Base
from app.schemas import DbActionResult


class IncompatibleSchemaError(Exception):
    """Raised by classify_schema when a candidate database already has
    at least one table pAIring doesn't recognize — see this module's own
    docstring for why that has to block every write action rather than
    just being a warning."""


def classify_schema(url: str) -> str:
    """Returns "empty" (no tables — safe to build a fresh schema) or
    "compatible" (every existing table name is one pAIring's own models
    define, or alembic_version — safe whether that's a fully-current
    install, an older one just missing newer tables, or a legacy
    pre-Alembic one). Raises IncompatibleSchemaError otherwise."""
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
    """Shared by db_config_service's _test_sqlite/_test_mysql once a raw
    connection has already succeeded: connectivity alone isn't enough to
    call a candidate database "OK" here — see this module's own
    docstring. This adds the schema-compatibility verdict on top, before
    the admin ever gets to build_schema/save with it."""
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
