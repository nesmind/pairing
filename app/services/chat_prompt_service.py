"""
Builds one reply's system prompt (persona/rule notes, then RAG context,
then a message's own ad-hoc attachments, in that layering order) and the RAG
source-filename list shown under a reply. Split out of
app.services.chat_service.build_reply_stream to keep that file under
CLAUDE.md's file-size rule — this is exactly the "what goes into the
prompt" half of that function, with the history-trimming/generation-
dispatch half left behind there.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Conversation, User
from app.services import chat_attachment_service
from app.services.chat_attachment_service import AttachmentInfo
from app.services.document_retrieval import build_augmented_system_prompt, retrieve_context
from app.services.inference_client import InferenceError
from app.services.note_service import build_pinned_system_prompt, resolve_conversation_notes


async def build_system_prompt(
    db: AsyncSession,
    conversation: Conversation,
    user: User,
    content: str,
    params: dict,
    attachments: list[AttachmentInfo],
) -> tuple[str, list[str]]:
    """Returns (system_prompt, source_filenames) for this one turn.

    Layered in this order: explicit persona/rules/skill note pins (Notes
    page or the chat page's icons), falling back to the user's default
    note per slot (see app/services/note_service.py); then RAG context
    (every chat automatically searches the shared knowledge base — no
    per-conversation on/off switch, only the rag_top_k "how many chunks"
    setting; degrades silently if the knowledge base is empty or the
    embedding model is unreachable, rather than failing the whole
    request); then, last, any of a message's own ad-hoc attachments that
    are `type="text"` (an image attachment is handled separately — see
    chat_attachment_service.apply_image_attachment, called by
    chat_service after `history` is built from this function's result).
    A channel conversation passes user_id=None to retrieve_context so RAG
    only searches the global knowledge base, never any one member's
    private uploads (see that function's own docstring)."""
    system_prompt = ""
    notes_by_type = await resolve_conversation_notes(db, conversation, user.id)
    if any(notes_by_type.values()):
        system_prompt = build_pinned_system_prompt(notes_by_type)

    source_filenames: list[str] = []
    rag_scope_user_id = user.id if conversation.channel_id is None else None
    try:
        retrieved = await retrieve_context(db, content, rag_scope_user_id, top_k=params.get("rag_top_k", 4))
    except InferenceError:
        retrieved = []
    if retrieved:
        system_prompt = build_augmented_system_prompt(system_prompt, retrieved)
        # De-duplicated, in first-seen (best-match-first) order — a
        # document can contribute more than one chunk, but it should
        # only be listed once in the "Sources" note.
        seen = set()
        for chunk in retrieved:
            if chunk["filename"] not in seen:
                seen.add(chunk["filename"])
                source_filenames.append(chunk["filename"])

    if attachments:
        system_prompt = chat_attachment_service.fold_text_attachments_into_prompt(system_prompt, attachments)

    return system_prompt, source_filenames
