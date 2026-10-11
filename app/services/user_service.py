"""Account management for Settings > Users (admin-only): create, edit and permanently delete accounts, plus
the "would this leave zero working admins" guard shared by the last two.
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
    """How many admins could still log in, optionally excluding one username; used to refuse an edit leaving zero
    working admins (update_user/delete_user).
    """
    stmt = select(func.count(User.id)).where(User.role == "admin", User.status == "active")
    if exclude_username:
        stmt = stmt.where(User.username != exclude_username)
    return (await db.execute(stmt)).scalar_one()


async def create_user(db: AsyncSession, body: UserCreate) -> User:
    """Creates an account. Admin-only (router's require_admin); there is no self-registration (private, invite-only).

    Refuses while no chat-capable model is installed: there'd be no default model to give them
    (get_default_model_for_new_users), and an account that can't chat until an admin fixes its model is worse
    than blocking creation with a reason.
    """
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
    await db.flush()  # assigns user.id for the rows below

    default_model = await get_default_model_for_new_users(db, installed)
    key = default_model_key(engine_service.current_engine())
    db.add(AppSetting(owner_id=user.id, key=key, value={"model": default_model}))
    await seed_default_notes(db, user)

    await db.commit()
    await db.refresh(user)
    return user


async def update_user(db: AsyncSession, username: str, body: UserUpdate) -> User:
    """Edits an account: password, admin/user role, disabled or enabled. All fields optional; a save touches only
    what changed.

    A disabled account's session stops on its next request (auth_service.get_current_user), not just its next
    login. Refuses any change that would leave zero active admins, so a mis-click can't lock out everyone.
    """
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
    # "or None": lets the Settings UI clear a name by saving an empty field; "" and "untouched" would be
    # indistinguishable once stored, so empty is normalized to NULL.
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
    """Permanently deletes an account and everything it owns: conversations (and messages), private documents (and
    files) and notes. No undo; disabling (update_user) is the reversible way to revoke access. Refuses to leave
    zero active admins (as update_user) and to delete the "admin" seed account (auth_service.seed_default_users)
    whatever other admins exist: the one account always present, so there's always a way back in.
    """
    user = (await db.execute(select(User).where(User.username == username))).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found.")

    if username == "admin":
        raise HTTPException(status_code=400, detail='The "admin" account can\'t be deleted.')

    if user.role == "admin" and await active_admin_count(db, exclude_username=username) == 0:
        raise HTTPException(status_code=400, detail="Can't remove the last active admin.")

    # Deleted row by row via the ORM, never a bulk query.delete, so cascades fire (a conversation's messages and
    # note_pins, a document's chunks, a note's pins); a bulk delete bypasses them and orphans rows.
    conversations = (await db.execute(select(Conversation).where(Conversation.owner_id == user.id))).scalars().all()
    for conversation in conversations:
        await db.delete(conversation)

    documents = (await db.execute(select(Document).where(Document.owner_id == user.id))).scalars().all()
    for document in documents:
        # Same order as document_upload.py's single-document delete: file first, so a crash leaves at worst an orphaned
        # file (healed by the next sync) rather than a DB row pointing at nothing.
        (KNOWLEDGE_DIR / document.filename).unlink(missing_ok=True)
        await db.delete(document)

    notes = (await db.execute(select(Note).where(Note.owner_id == user.id))).scalars().all()
    for note in notes:
        await db.delete(note)

    # No cascades here: a bulk delete is fine. A leftover channel-membership row for a deleted user crashed
    # channel_service.to_channel_out for that channel (it reads member.user.username unconditionally).
    await db.execute(delete(AppSetting).where(AppSetting.owner_id == user.id))
    await db.execute(delete(ChannelMember).where(ChannelMember.user_id == user.id))

    await db.delete(user)
    await db.commit()
