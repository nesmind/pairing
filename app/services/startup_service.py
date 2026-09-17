"""
The actual work behind app/main.py's startup/shutdown event handlers —
split out purely to keep main.py under CLAUDE.md's file-size rule; this
is business logic (seeding, backfills, knowledge sync, the local-
instances supervisor), not routing, so it belongs here regardless.
"""

import logging

from sqlalchemy import select, text

from app.config import AUTO_MIGRATE, IS_PRIMARY
from app.database import AsyncSessionLocal, init_db
from app.models import User
from app.services import (
    comfyui_pool,
    instance_pool,
    instance_service,
    ollama_pool,
    ollama_ps_poller,
    ollama_telemetry,
    retention_poller,
    settings_service,
    system_metrics_poller,
)
from app.services.auth_service import seed_default_users
from app.services.conversation_service import mark_interrupted_messages_as_errored
from app.services.document_sync import sync_all_on_startup
from app.services.image_generation_service import mark_interrupted_jobs_as_errored
from app.services.note_service import seed_default_notes
from app.services.ollama_client import OllamaError, list_models

logger = logging.getLogger("llama_chat")


async def run_startup_tasks() -> None:
    """Runs once when the server process starts: makes sure the
    database schema is up to date (or, with app.config.AUTO_MIGRATE off,
    just verifies it already is — see init_db), seeds the starter admin
    account, logs a clear warning (instead of a confusing error deep in
    a request later) if Ollama isn't reachable yet, and syncs the
    knowledge-base folder (app.config.KNOWLEDGE_DIR) so any files
    dropped in while the server was down are picked up immediately."""
    logger.info(
        "AUTO_MIGRATE=%s — %s",
        AUTO_MIGRATE,
        "schema will be migrated automatically if needed"
        if AUTO_MIGRATE
        else "schema must already be current (run `.venv/bin/alembic upgrade head` separately if not)",
    )
    await init_db()

    # A fresh AsyncSessionLocal() rather than the get_db() dependency:
    # this runs outside of any HTTP request, so there's no request-
    # scoped session to reuse — we open one, use it, and close it
    # ourselves (the `async with` below).
    async with AsyncSessionLocal() as db:
        admin_user = await seed_default_users(db)
        # Conversations created before multi-user support existed have
        # no owner (see the migration in app/database.py) — attach them
        # to admin rather than leaving them permanently invisible to
        # everyone, since deleting someone's existing chat history as a
        # side effect of an unrelated feature would be a bad surprise.
        # `channel_id IS NULL` is load-bearing here, not just belt-and-
        # suspenders: a channel's shared conversation (see
        # app/models/conversation.py) also has owner_id NULL *by design*
        # — without this exclusion, this backfill would silently steal
        # every channel's conversation for admin on every single
        # restart, which is exactly what happened before this exclusion
        # was added (see repair pass right below for undoing that).
        result = await db.execute(
            text("UPDATE conversations SET owner_id = :admin_id WHERE owner_id IS NULL AND channel_id IS NULL"),
            {"admin_id": admin_user.id},
        )
        if result.rowcount:
            logger.info("Assigned %d pre-existing conversation(s) to admin.", result.rowcount)

        # Self-healing repair, unconditional on every startup (cheap: a
        # no-op UPDATE touching zero rows once nothing is broken): a
        # channel's shared conversation must never have owner_id set —
        # channel_id/owner_id are mutually exclusive (see
        # app/models/conversation.py) — but the backfill above did
        # exactly that on every restart before it excluded channel_id,
        # so any row still carrying that stray owner_id (from before
        # this fix existed) gets corrected here rather than staying
        # silently broken (invisible-but-undeletable in its owner's
        # personal chat list) until someone notices and asks about it.
        result = await db.execute(
            text("UPDATE conversations SET owner_id = NULL WHERE owner_id IS NOT NULL AND channel_id IS NOT NULL"),
        )
        if result.rowcount:
            logger.info("Repaired %d channel conversation(s) with a stray owner_id.", result.rowcount)
        await db.commit()

        # Backfill: every user (not just admin/ran above) gets their 3
        # protected persona/rules/skill notes if they don't already have
        # them — covers both the two seed accounts on a fresh install and
        # any account that existed before this feature shipped.
        # seed_default_notes is idempotent, so this is safe to run on
        # every single startup rather than needing its own one-time
        # migration flag.
        for user in (await db.execute(select(User))).scalars().all():
            await seed_default_notes(db, user)

        await db.commit()

        # Primes the live Ollama/ComfyUI host pools (Settings > External
        # servers) from whatever's currently saved — unlike the
        # instance-count/proxy-mode priming right below, this runs on
        # *every* instance, not just the primary: each instance
        # independently routes its own chat/image-generation requests
        # through these pools (see app.services.ollama_pool/comfyui_pool),
        # so every one of them needs its own cache warm before the
        # Ollama connectivity check right after this block runs.
        ollama_pool.refresh_from_config(await settings_service.get_ollama_server_config(db))
        comfyui_pool.refresh_from_config(await settings_service.get_comfyui_config(db))

        # Same "every instance, not just primary" reasoning as the pool refresh right above — each instance
        # instruments its own outbound Ollama calls independently (see app.services.ollama_client), so each needs
        # its own OpenTelemetry SDK wired up. Must happen before the Ollama-reachability check below: that check
        # `return`s early on a cold start without Ollama reachable yet, which would otherwise skip this permanently.
        ollama_telemetry.init_ollama_telemetry()

        # Local-instances supervisor (Settings > System) — primary only,
        # so a sibling this same call might spawn never tries to spawn
        # siblings of its own. See app/services/instance_service.py.
        # Priming instance_pool's proxy-mode cache here too (same
        # primary-only gate) is what lets app/services/instance_proxy.py
        # decide per-request without a DB round trip — see
        # instance_pool.get_cached_proxy_mode's own docstring.
        if IS_PRIMARY:
            await instance_service.reconcile_on_startup(db)
            instance_pool.set_cached_proxy_mode(await settings_service.get_proxy_mode(db))

            # Ollama's currently-loaded-model state (GET /api/ps), this machine's own CPU/RAM/disk usage, and the
            # retention sweep that keeps both (plus telemetry_events) from growing unbounded over a long uptime
            # are all host-level, not per-instance — starting any of them from every instance would just
            # duplicate work for no benefit, same reasoning as the supervisor right above. Must also run before
            # the reachability check below for the same early-return reason as init_ollama_telemetry() — all
            # three already tolerate Ollama/the underlying syscalls being unavailable via their own per-iteration
            # try/except, so it's safe to start them even before that check has run.
            ollama_ps_poller.start_ollama_ps_poller()
            system_metrics_poller.start_system_metrics_poller()
            retention_poller.start_retention_poller()

            # A message left "streaming" (personal chat or channel alike)
            # is a generation task that was running in some earlier
            # process — an asyncio.Task can't survive a restart, so
            # without this it would stay stuck showing a permanent
            # "typing" state forever. See mark_interrupted_messages_as_errored's
            # own docstring for why this is scoped to a single-instance
            # restart only.
            interrupted = await mark_interrupted_messages_as_errored(db)
            if interrupted:
                logger.info(
                    "Marked %d interrupted repl%s as errored after restart.",
                    interrupted,
                    "y" if interrupted == 1 else "ies",
                )

            # Same restart-survival gap as the reply sweep above, for a
            # generation job left "queued"/"running" by a prior process
            # — see mark_interrupted_jobs_as_errored's own docstring.
            # ComfyUI itself is deliberately NOT stopped here (see
            # app/services/comfyui_service.py's own docstring on why it
            # survives an app restart) — only this app's own bookkeeping
            # of a job it can no longer track is corrected.
            interrupted_jobs = await mark_interrupted_jobs_as_errored(db)
            if interrupted_jobs:
                logger.info(
                    "Marked %d interrupted image generation job%s as errored after restart.",
                    interrupted_jobs,
                    "" if interrupted_jobs == 1 else "s",
                )

    try:
        models = await list_models()
        logger.info("Connected to Ollama — %d model(s) available.", len(models))
    except OllamaError as exc:
        logger.warning(
            "Could not reach Ollama at startup (%s). The UI will still load, "
            "but chat requests will fail until Ollama is running.",
            exc,
        )
        return

    async with AsyncSessionLocal() as db:
        # sync_all_on_startup degrades gracefully (logged as an "error"
        # in its stats, not raised) if no embedding model is available
        # yet.
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
    """Cleans up any siblings this primary spawned — most load-bearing
    on the plain scripts/stop.sh path (no cgroup to do this for us
    there, unlike a real systemd stop/reboot — see
    app/services/instance_service.py's own docstring). Also flushes any
    telemetry spans still queued in this instance's own OTel SDK before
    the process exits — see ollama_telemetry.shutdown_ollama_telemetry's
    own docstring for why this must run on a worker thread, not inline."""
    if IS_PRIMARY:
        await instance_service.terminate_all_siblings()
    await ollama_telemetry.shutdown_ollama_telemetry()
