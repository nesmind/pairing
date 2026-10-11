"""Alembic entry point, wired to the project's config: URL from app.config.DATABASE_URL, target
metadata from app.models, so autogenerate compares the real DB with the real models.
"""

from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context
from app import models  # noqa: F401 -- registers every table on Base.metadata
from app.config import DATABASE_URL
from app.database import Base

config = context.config
if config.config_file_name is not None:
    # disable_existing_loggers=False: this runs embedded in the app process (app.database.init_db).
    # The default True silently disabled every registered logger (app, uvicorn) after a migration.
    fileConfig(config.config_file_name, disable_existing_loggers=False)

# Fall back to DATABASE_URL only if the caller set no URL. The plain CLI never does (alembic.ini
# has none); app.database._run_migrations pre-sets one for candidate databases
# (db_config_service.build_schema), which must not be overwritten.
if not config.get_main_option("sqlalchemy.url"):
    config.set_main_option("sqlalchemy.url", DATABASE_URL)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emits SQL without a live DB (`alembic upgrade head --sql`); kept for reviewing or hand-applying SQL."""
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
