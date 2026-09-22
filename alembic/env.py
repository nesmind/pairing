"""
Alembic's entry point. Wired to this project's own config/models instead
of the usual generated boilerplate: the database URL comes from
app.config.DATABASE_URL (the single place that already resolves it from
the environment) and the "what should the schema look like" target comes
from app.models' metadata, so `alembic revision --autogenerate` compares
the real database against the real ORM models rather than a copy of
either.
"""

from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context
from app import models  # noqa: F401 -- registers every table on Base.metadata
from app.config import DATABASE_URL
from app.database import Base

config = context.config
if config.config_file_name is not None:
    # disable_existing_loggers=False: this runs embedded, inside the app's own
    # long-running process (app.database.init_db, on every startup when
    # AUTO_MIGRATE is on), not as the standalone `alembic upgrade` CLI command
    # its default of True assumes. Left at True, this silently disabled every
    # logger the app had already registered — "llama_chat", uvicorn's own
    # "uvicorn.error"/"uvicorn.access" — for the rest of that process's life,
    # the instant a migration ran, with no error or warning anywhere. Confirmed
    # live: app.log went completely silent right after alembic's own setup
    # lines despite real traffic being served correctly the whole time.
    fileConfig(config.config_file_name, disable_existing_loggers=False)

# Only fall back to the app's own live DATABASE_URL if the caller hasn't
# already set one on this Config object. The plain CLI (`alembic upgrade
# head`) never does — alembic.ini deliberately has no sqlalchemy.url of
# its own (see its comment) — so this is what supplies it there. But
# app.database._run_migrations pre-sets sqlalchemy.url itself when
# building the schema on a *candidate* database (see
# app.services.db_config_service.build_schema); unconditionally
# overwriting that here would silently run every migration against the
# live app database instead of whatever target was actually asked for.
if not config.get_main_option("sqlalchemy.url"):
    config.set_main_option("sqlalchemy.url", DATABASE_URL)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emits the SQL a migration would run without a live DB connection
    (`alembic upgrade head --sql`) — not used by this app itself, but
    kept so that path still works for anyone who wants to review/hand-
    apply the SQL instead of letting Alembic connect directly."""
    context.configure(
        url=DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
