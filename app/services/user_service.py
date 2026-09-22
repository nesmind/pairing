"""
Account management for the Settings > Users tab (admin-only): creating,
editing, and permanently deleting accounts, plus the "would this leave
zero working admins" guard shared by the last two.
"""

from fastapi import HTTPException
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import KNOWLEDGE_DIR
from app.models import AppSetting, ChannelMember, Conversation, Document, Note, User
from app.schemas import UserCreate, UserUpdate
from app.services import engine_service
from app.services.auth_service import hash_password
from app.services.model_catalog_service import get_default_model_for_new_users, installed_chat_models
from app.services.note_service import seed_default_notes
from app.services.settings_service import default_model_key


async def active_admin_count(db: AsyncSession, exclude_username: str | None = None) -> int:
    """How many admins would still be able to log in, optionally not
    counting one specific username — used to refuse an edit that would
    leave the system with zero working admin accounts (see
    update_user/delete_user below)."""
    stmt = select(func.count(User.id)).where(User.role == "admin", User.status == "active")
    if exclude_username:
        stmt = stmt.where(User.username != exclude_username)
    return (await db.execute(stmt)).scalar_one()


async def create_user(db: AsyncSession, body: UserCreate) -> User:
    """Creates a new account. Admin-only (enforced by the router's
    Depends(require_admin)): there's no public self-registration in this
    app by design, since it's meant for a private, invite-only
    deployment.

    Refuses to create anyone while no chat-capable model is installed —
    there'd be nothing to give them as their own default model (see
    get_default_model_for_new_users), and a brand-new account that can't
    chat at all until an admin remembers to fix its model later is a
    worse experience than just blocking creation with a clear reason."""
    installed = await installed_chat_models()
    if not installed:
        raise HTTPException(
            status_code=400,
            detail="No models installed yet — install at least one chat model before adding a new user.",
        )
    existing = (await db.execute(select(User).where(User.username == body.username))).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(status_code=400, detail=f'Username "{body.username}" is already taken.')

    user = User(
        username=body.username,
        password_hash=hash_password(body.password),
        first_name=body.first_name or None,
        last_name=body.last_name or None,
        role=body.role,
        status=body.status,
    )
    db.add(user)
    await db.flush()  # assigns user.id so the rows below can reference it

    default_model = await get_default_model_for_new_users(db, installed)
    key = default_model_key(engine_service.current_engine())
    db.add(AppSetting(owner_id=user.id, key=key, value={"model": default_model}))
    await seed_default_notes(db, user)

    await db.commit()
    await db.refresh(user)
    return user


async def update_user(db: AsyncSession, username: str, body: UserUpdate) -> User:
    """Edits an existing account: change its password, promote/demote
    between admin and user, and/or disable or re-enable it. Every field
    is optional, so a save only touches what was actually changed.

    A disabled account's session stops working on its very next request
    (see app.services.auth_service.get_current_user), not just on its
    next login attempt. Refuses any change that would leave zero active
    admins, so a mis-click can't lock every admin out of their own
    System tab."""
    user = (await db.execute(select(User).where(User.username == username))).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found.")

    would_lose_admin = user.role == "admin" and (
        (body.role is not None and body.role != "admin") or (body.status is not None and body.status != "active")
    )
    if would_lose_admin and await active_admin_count(db, exclude_username=username) == 0:
        raise HTTPException(status_code=400, detail="Can't remove the last active admin.")

    if body.password:
        user.password_hash = hash_password(body.password)
    # "or None" (not a bare assignment): lets the Settings UI clear an
    # existing name by saving an empty field — an empty string and
    # "wasn't touched" would otherwise be indistinguishable once stored,
    # so this normalizes an empty value down to NULL rather than leaving
    # a blank string sitting in the column.
    if body.first_name is not None:
        user.first_name = body.first_name or None
    if body.last_name is not None:
        user.last_name = body.last_name or None
    if body.role is not None:
        user.role = body.role
    if body.status is not None:
        user.status = body.status

    await db.commit()
    await db.refresh(user)
    return user


async def delete_user(db: AsyncSession, username: str) -> None:
    """Permanently deletes an account and everything it owns — its
    conversations (and their messages), private documents (and their
    files on disk), and notes. There's no undo; disabling (see
    update_user above) is the reversible alternative when the goal is
    just revoking access, not losing the account's data. Refuses to
    leave the system with zero active admins, same guard as
    update_user — and additionally refuses to delete the "admin" seed
    account specifically (see
    app.services.auth_service.seed_default_users), regardless of how
    many other admins exist: it's the one account guaranteed to always
    be there, so there's always a way back in."""
    user = (await db.execute(select(User).where(User.username == username))).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found.")

    if username == "admin":
        raise HTTPException(status_code=400, detail='The "admin" account can\'t be deleted.')

    if user.role == "admin" and await active_admin_count(db, exclude_username=username) == 0:
        raise HTTPException(status_code=400, detail="Can't remove the last active admin.")

    # Deleted one row at a time via the ORM (never a bulk query.delete())
    # so each relationship's cascade actually fires — a conversation's
    # messages and note_pins, a document's chunks, a note's pins. A bulk
    # delete bypasses ORM-level cascades entirely and would leave those
    # orphaned.
    conversations = (await db.execute(select(Conversation).where(Conversation.owner_id == user.id))).scalars().all()
    for conversation in conversations:
        await db.delete(conversation)

    documents = (await db.execute(select(Document).where(Document.owner_id == user.id))).scalars().all()
    for document in documents:
        # Same order as the single-document delete in
        # app/services/document_upload.py: remove the file first, so a
        # crash between the two steps leaves at worst an orphaned file
        # (self-healed by the next sync) rather than a DB row pointing
        # at nothing.
        (KNOWLEDGE_DIR / document.filename).unlink(missing_ok=True)
        await db.delete(document)

    notes = (await db.execute(select(Note).where(Note.owner_id == user.id))).scalars().all()
    for note in notes:
        await db.delete(note)

    # No cascade relationships to worry about for either of these — a bulk delete is fine. Channel membership
    # rows specifically: leaving one behind (confirmed live, not just theoretical) makes
    # channel_service.to_channel_out crash on that channel from then on — it reads member.user.username
    # unconditionally, and a membership row pointing at a deleted user has no User to read that from anymore.
    await db.execute(delete(AppSetting).where(AppSetting.owner_id == user.id))
    await db.execute(delete(ChannelMember).where(ChannelMember.user_id == user.id))

    await db.delete(user)
    await db.commit()
