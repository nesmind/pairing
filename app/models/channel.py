"""Channel and ChannelMember: a named group chat whose members share one Conversation
(Conversation.channel_id). Logic: app/services/channel_service.py; gating: conversation_service.py
(any member reads/chats; admins and is_manager members change model/persona/rules/skill).
"""

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import relationship

from app.database import Base
from app.models._base import ID_LEN, new_id, utcnow


class Channel(Base):
    """A named group chat; membership is on ChannelMember, the thread is its one Conversation."""

    __tablename__ = "channels"

    id = Column(String(ID_LEN), primary_key=True, default=new_id)
    name = Column(String(255), nullable=False, unique=True)
    created_at = Column(DateTime, default=utcnow)

    # uselist=False: one shared Conversation per channel. delete-orphan cascade removes its thread and messages.
    conversation = relationship(
        "Conversation",
        back_populates="channel",
        uselist=False,
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    # lazy="selectin" (see Conversation.messages); channel_service.can_manage_channel_conversation reads members without
    # extra queries.
    members = relationship(
        "ChannelMember",
        back_populates="channel",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class ChannelMember(Base):
    """One user's membership in a channel. is_manager allows changing the conversation's model/persona/rules/skill;
    only users with the "channel_manager" role may be flagged (enforced in channel_service).
    """

    __tablename__ = "channel_members"
    __table_args__ = (UniqueConstraint("channel_id", "user_id", name="uq_channel_member_channel_user"),)

    id = Column(String(ID_LEN), primary_key=True, default=new_id)
    channel_id = Column(String(ID_LEN), ForeignKey("channels.id"), nullable=False)
    user_id = Column(String(ID_LEN), ForeignKey("users.id"), nullable=False)
    is_manager = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=utcnow)

    channel = relationship("Channel", back_populates="members", lazy="selectin")
    # lazy="selectin": channel_service.to_channel_out reads member.user.username.
    user = relationship("User", lazy="selectin")
