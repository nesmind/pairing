"""ImageGenerationJob model — see app/services/image_generation_service.py
for the actual generation lifecycle this describes, and
app/services/comfyui_client.py for the ComfyUI HTTP calls it makes."""

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text

from app.database import Base
from app.models._base import ID_LEN, new_id, utcnow


class ImageGenerationJob(Base):
    """One text-to-image request — one row per generation, one image per
    row (no batching in v1). No relationship to User: every query already
    filters by owner_id directly, so there's nothing that ever needs to
    walk from this row back to the full User object; if a future read
    site does, it should look one up via `db.get(User, job.owner_id)`
    directly rather than adding a lazy relationship here — see
    app/routers/chat.py's get_attachment for why (a relationship without
    lazy="selectin", read outside an awaited context, is a real crash
    this codebase already hit once)."""

    __tablename__ = "image_generation_jobs"

    id = Column(String(ID_LEN), primary_key=True, default=new_id)
    owner_id = Column(String(ID_LEN), ForeignKey("users.id"), nullable=False)
    # "queued" (row created, background task not yet started) ->
    # "running" (submitted to ComfyUI, polling /history) -> "complete" or
    # "error". See image_generation_service._run_job for the only writer
    # of every transition.
    status = Column(String(20), nullable=False, default="queued")
    prompt = Column(Text, nullable=False)
    negative_prompt = Column(Text, nullable=True)
    # Which ComfyUI checkpoint this job was submitted with — an ordinary
    # string (whatever comfyui_client.list_checkpoints() returned at
    # request time), not an FK: ComfyUI owns the actual checkpoint files,
    # this app just remembers which name was used for display/re-run.
    checkpoint = Column(String(255), nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    steps = Column(Integer, nullable=False)
    cfg = Column(Float, nullable=False)
    seed = Column(Integer, nullable=False)
    # ComfyUI's own /prompt response id — needed to poll GET /history/{id}
    # for this specific job. Null until the submit call succeeds.
    comfy_prompt_id = Column(String(64), nullable=True)
    # Relative to app.config.IMAGES_DIR (never absolute — see
    # ATTACHMENTS_DIR's identical convention), set only once status
    # becomes "complete".
    image_path = Column(String(500), nullable=True)
    image_filename = Column(String(255), nullable=True)
    # Set only on status="error" — the reason, shown to the user instead
    # of an image.
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    @property
    def url(self) -> str | None:
        """Powers ImageGenerationJobOut.url — None until the image
        actually exists on disk (see app/routers/image_generation.py's
        GET .../jobs/{id}/image, the only reader of image_path)."""
        return f"/api/image-generation/jobs/{self.id}/image" if self.image_path else None
