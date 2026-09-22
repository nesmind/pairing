"""Channel + ChannelMember models — a Channel is a named group chat: its
members share exactly one Conversation (see Conversation.channel_id in
app/models/conversation.py) instead of each having their own. See
app/services/channel_service.py for creation/membership logic and
app/services/conversation_service.py for how a channel's shared
conversation is gated (any member can read/chat; only an admin or a
member flagged is_manager can change its model/persona/rules/skill)."""

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import relationship

from app.database import Base
from app.models._base import ID_LEN, new_id, utcnow


class Channel(Base):
    """A named group chat. Membership (who's in it, and which members can
    manage its model/persona/rules/skill) lives on ChannelMember below;
    the actual chat thread is the one Conversation this channel owns."""

    __tablename__ = "channels"

    id = Column(String(ID_LEN), primary_key=True, default=new_id)
    name = Column(String(255), nullable=False, unique=True)
    created_at = Column(DateTime, default=utcnow)

    # uselist=False: exactly one shared Conversation per channel.
    # cascade="all, delete-orphan" so deleting a channel takes its chat
    # thread (and, transitively, that conversation's messages/note_pins —
    # see Conversation's own cascades) with it.
    conversation = relationship(
        "Conversation",
        back_populates="channel",
        uselist=False,
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    # lazy="selectin" — see app/models/conversation.py's
    # Conversation.messages for why the app defaults accessed
    # relationships to this under the async session;
    # channel_service.can_manage_channel_conversation reads
    # conversation.channel.members directly off an already-loaded
    # Conversation without an extra query.
    members = relationship(
        "ChannelMember",
        back_populates="channel",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class ChannelMember(Base):
    """One user's membership in one channel. is_manager grants that
    member the right to change the channel conversation's model/persona/
    rules/skill (see channel_service.can_manage_channel_conversation) —
    only users holding the global "channel_manager" role are allowed to
    be flagged this way (enforced in channel_service, not here, since a
    role change on User shouldn't retroactively edit existing
    memberships)."""

    __tablename__ = "channel_members"
    __table_args__ = (UniqueConstraint("channel_id", "user_id", name="uq_channel_member_channel_user"),)

    id = Column(String(ID_LEN), primary_key=True, default=new_id)
    channel_id = Column(String(ID_LEN), ForeignKey("channels.id"), nullable=False)
    user_id = Column(String(ID_LEN), ForeignKey("users.id"), nullable=False)
    is_manager = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=utcnow)

    channel = relationship("Channel", back_populates="members", lazy="selectin")
    # lazy="selectin" — channel_service.to_channel_out reads
    # member.user.username directly.
    user = relationship("User", lazy="selectin")
