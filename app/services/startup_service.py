"""The work behind app/main.py's startup/shutdown handlers, split from main.py for the file-size rule; it is
business logic (seeding, backfills, knowledge sync, the instances supervisor), not routing.
"""

import logging

from sqlalchemy import select, text

from app.config import AUTO_MIGRATE, IS_PRIMARY
from app.database import AsyncSessionLocal, init_db
from app.models import User
from app.services import (
    cluster_membership,
    comfyui_pool,
    connector_config_cache,
    engine_service,
    image_engine_service,
    instance_pool,
    instance_service,
    matricxon_pool,
    matricxon_process,
    matricxon_ps_poller,
    matricxon_telemetry,
    ollama_pool,
    ollama_ps_poller,
    ollama_telemetry,
    retention_poller,
    sdcpp_pool,
    settings_service,
    system_metrics_poller,
)
from app.services.auth_service import seed_default_users
from app.services.conversation_service import mark_interrupted_messages_as_errored
from app.services.document_sync import sync_all_on_startup
from app.services.image_generation_service import mark_interrupted_jobs_as_errored
from app.services.inference_client import InferenceError, list_models
from app.services.note_service import seed_default_notes

logger = logging.getLogger("llama_chat")


async def run_startup_tasks() -> None:
    """Runs once at server start: ensures the schema is current (or with AUTO_MIGRATE off, just verifies it; init_db),
    seeds the starter admin, logs a clear warning if the active engine (engine_service) is unreachable (instead
    of a confusing error later) and syncs KNOWLEDGE_DIR so files dropped while down are picked up.
    """
    logger.info(
        "AUTO_MIGRATE=%s — %s",
        AUTO_MIGRATE,
        "schema will be migrated automatically if needed"
        if AUTO_MIGRATE
        else "schema must already be current (run `.venv/bin/alembic upgrade head` separately if not)",
    )
    await init_db()

    # A fresh AsyncSessionLocal, not get_db: this runs outside any request, so we open and close the session
    # ourselves.
    async with AsyncSessionLocal() as db:
        admin_user = await seed_default_users(db)
        # Conversations from before multi-user support have no owner (app/database.py migration): attach them to admin
        # rather than leave them invisible (deleting history would be a bad surprise). `channel_id IS NULL` is
        # load-bearing: a channel's shared conversation (models/conversation.py) also has owner_id NULL by design, and
        # without it this backfill stole every channel's conversation for admin on each restart (see the repair below).
        result = await db.execute(
            text("UPDATE conversations SET owner_id = :admin_id WHERE owner_id IS NULL AND channel_id IS NULL"),
            {"admin_id": admin_user.id},
        )
        if result.rowcount:
            logger.info("Assigned %d pre-existing conversation(s) to admin.", result.rowcount)

        # Self-healing repair on every startup (cheap: a no-op UPDATE of zero rows once fixed): a channel conversation
        # must never have owner_id (mutually exclusive with channel_id, models/conversation.py), but the backfill above
        # did that each restart before it excluded channel_id. Stray rows get corrected rather than staying broken
        # (invisible-but-undeletable in the owner's list).
        result = await db.execute(
            text("UPDATE conversations SET owner_id = NULL WHERE owner_id IS NOT NULL AND channel_id IS NOT NULL"),
        )
        if result.rowcount:
            logger.info("Repaired %d channel conversation(s) with a stray owner_id.", result.rowcount)
        await db.commit()

        # Backfill: every user gets their 3 protected persona/rules/skill notes if missing (seed accounts on a fresh
        # install and accounts predating the feature). seed_default_notes is idempotent, so no one-time migration flag.
        for user in (await db.execute(select(User))).scalars().all():
            await seed_default_notes(db, user)

        await db.commit()

        # Primes the engine/ComfyUI host pools (Settings > External servers) from what's saved. Unlike the
        # instance-count/proxy-mode priming below, this runs on *every* instance: each routes its own chat/image
        # requests
        # through these pools (ollama_pool/comfyui_pool) and needs its cache warm before the connectivity check after
        # this block.
        ollama_pool.refresh_from_config(await settings_service.get_ollama_server_config(db))
        comfyui_pool.refresh_from_config(await settings_service.get_comfyui_config(db))
        sdcpp_pool.refresh_from_config(await image_engine_service.get_sdcpp_config(db))
        matricxon_config = await settings_service.get_matricxon_server_config(db)
        matricxon_pool.refresh_from_config(matricxon_config)
        # Which engine feeds inference_client's dispatch (engine_service.load_cache_from_db): every instance needs its
        # own in-process cache, like the pool refreshes above.
        await engine_service.load_cache_from_db(db)
        # Every Connector's admin state: same reasoning, and it must run before the reachability check below in case a
        # Connector-backed engine is active (services.connectors).
        await connector_config_cache.load_cache_from_db(db)

        # Every instance, not just the primary, like the pool refresh: each instruments its own outbound
        # Ollama/Matricxon
        # calls (ollama_client/matricxon_client) and needs its own OpenTelemetry SDK, one per engine (see
        # matricxon_telemetry for the separate TracerProvider). Must precede the reachability check below, which
        # returns early on a cold start without the engine and would skip this permanently.
        ollama_telemetry.init_ollama_telemetry()
        matricxon_telemetry.init_matricxon_telemetry()

        # Local-instances supervisor (Settings > System): primary only, so a spawned sibling never spawns siblings
        # (instance_service.py). Priming instance_pool's proxy-mode cache here (same gate) lets instance_proxy.py decide
        # per request without a DB round trip (instance_pool.get_cached_proxy_mode).
        if IS_PRIMARY:
            await instance_service.reconcile_on_startup(db)
            instance_pool.set_cached_proxy_mode(await settings_service.get_proxy_mode(db))

            # Re-syncs Matricxon's.env to the last-saved config on every startup, not just on save (see
            # matricxon_process.write_env_file: Matricxon must get the right parameters however it is started: this app,
            # bare scripts/start.sh, a reboot, systemd). Closes a gap: a save that landed before
            # write_env_file could resolve the project directory silently no-oped and left.env stale with nothing to
            # retry;
            # a restart is when the directory is most likely resolvable.
            matricxon_process.write_env_file(matricxon_config)

            # Loaded-model state (GET /api/ps) of Ollama/Matricxon, this machine's CPU/RAM/disk, and the retention sweep
            # that bounds all of these (plus telemetry_events) are host-level, not per-instance: starting them on every
            # instance would duplicate work (like the supervisor above). They must also run before the reachability
            # check
            # below (same early-return reason as init_ollama_telemetry); each tolerates Ollama/Matricxon/syscalls
            # being
            # unavailable via its own per-iteration try/except.
            ollama_ps_poller.start_ollama_ps_poller()
            matricxon_ps_poller.start_matricxon_ps_poller()
            system_metrics_poller.start_system_metrics_poller()
            retention_poller.start_retention_poller()
            cluster_membership.start_membership_sync()

            # A message left "streaming" (personal or channel) was a generation task in an earlier process; an
            # asyncio.Task can't survive a restart, so it would show "typing" forever. Single-instance restart only (see
            # mark_interrupted_messages_as_errored).
            interrupted = await mark_interrupted_messages_as_errored(db)
            if interrupted:
                logger.info(
                    "Marked %d interrupted repl%s as errored after restart.",
                    interrupted,
                    "y" if interrupted == 1 else "ies",
                )

            # Same restart gap as the reply sweep, for jobs left "queued"/"running" (mark_interrupted_jobs_as_errored).
            # ComfyUI itself is NOT stopped (it survives app restarts, see comfyui_service.py); only this app's
            # bookkeeping
            # of a job it can no longer track is corrected.
            interrupted_jobs = await mark_interrupted_jobs_as_errored(db)
            if interrupted_jobs:
                logger.info(
                    "Marked %d interrupted image generation job%s as errored after restart.",
                    interrupted_jobs,
                    "" if interrupted_jobs == 1 else "s",
                )

    active_engine = engine_service.current_engine()
    try:
        models = await list_models()
        logger.info("Connected to %s — %d model(s) available.", active_engine, len(models))
    except InferenceError as exc:
        logger.warning(
            "Could not reach the active engine (%s) at startup (%s). The UI will still load, "
            "but chat requests will fail until it's running.",
            active_engine,
            exc,
        )
        return

    async with AsyncSessionLocal() as db:
        # sync_all_on_startup degrades gracefully (logged as an "error" in its stats, not raised) if no embedding model
        # is available yet.
        stats = await sync_all_on_startup(db)
        logger.info(
            "Knowledge base synced: %d added, %d updated, %d removed, %d unchanged.",
            stats["added"],
            stats["updated"],
            stats["removed"],
            stats["skipped"],
        )
        for error in stats["errors"]:
            logger.warning("Knowledge base: %s", error)


async def run_shutdown_tasks() -> None:
    """Cleans up siblings this primary spawned (matters most on plain scripts/stop.sh: no cgroup, unlike a
    systemd stop/reboot; see instance_service.py). Also flushes telemetry spans still queued in either engine's
    OTel SDK before exit (on a worker thread; see ollama_telemetry.shutdown_ollama_telemetry).
    """
    if IS_PRIMARY:
        await instance_service.terminate_all_siblings()
    await ollama_telemetry.shutdown_ollama_telemetry()
    await matricxon_telemetry.shutdown_matricxon_telemetry()
