"""ImageGenerationJob model (lifecycle: app/services/image_generation_service.py; engine calls: sdcpp_client.py)."""

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text

from app.database import Base
from app.models._base import ID_LEN, new_id, utcnow


class ImageGenerationJob(Base):
    """One text-to-image request: one row, one image (no batching in v1). No User relationship: queries
    filter by owner_id; use `db.get(User, job.owner_id)` if needed. A relationship without lazy="selectin"
    read outside an awaited context crashes (see app/routers/chat.py get_attachment).
    """

    __tablename__ = "image_generation_jobs"

    id = Column(String(ID_LEN), primary_key=True, default=new_id)
    owner_id = Column(String(ID_LEN), ForeignKey("users.id"), nullable=False)
    # "queued" -> "running" (submitted, polling /history) -> "complete" or "error". Only
    # image_generation_service._run_job writes transitions.
    status = Column(String(20), nullable=False, default="queued")
    prompt = Column(Text, nullable=False)
    negative_prompt = Column(Text, nullable=True)
    # Image model file used (from sdcpp_client.list_checkpoints); not an FK, since the engine owns the files.
    checkpoint = Column(String(255), nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    steps = Column(Integer, nullable=False)
    cfg = Column(Float, nullable=False)
    seed = Column(Integer, nullable=False)
    # "text_to_image" or "image_to_image"; strength (0-1, distance from the source) only for the latter.
    # The source image is not stored (app.services.image_init).
    mode = Column(String(20), nullable=False, default="text_to_image", server_default="text_to_image")
    strength = Column(Float, nullable=True)
    # The engine's own job id, for polling it. Null until submit succeeds.
    engine_job_id = Column(String(64), nullable=True)
    # stable-diffusion.cpp only: log file size at job start, so progress reads only this job's lines
    # (app.services.sdcpp_progress). Null for other engines.
    log_offset = Column(Integer, nullable=True)
    # Relative to IMAGES_DIR (like ATTACHMENTS_DIR); set when status becomes "complete".
    image_path = Column(String(500), nullable=True)
    image_filename = Column(String(255), nullable=True)
    # Reason shown instead of an image; only for status="error".
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    @property
    def url(self) -> str | None:
        """For ImageGenerationJobOut.url; None until the image exists (app/routers/image_generation.py
        GET.../jobs/{id}/image)."""
        return f"/api/image-generation/jobs/{self.id}/image" if self.image_path else None
