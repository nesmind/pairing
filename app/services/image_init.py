"""Image-to-image source pictures, kept only in memory between the request and its background job (both run in the
same process): a user's photo is never written to disk or the database, and is gone once the job has started."""

import base64

from app.models import ImageGenerationJob

_pending: dict[str, bytes] = {}


def stash(job_id: str, b64_png: str) -> None:
    _pending[job_id] = base64.b64decode(b64_png)


def discard(job_id: str) -> None:
    _pending.pop(job_id, None)


def init_params(job: ImageGenerationJob, engine: str) -> dict:
    """Extra engine params for `job` ({} for text-to-image). ValueError (-> the job's error) if the source image is
    unusable. Always consumes the stashed image."""
    png = _pending.pop(job.id, None)
    if job.mode != "image_to_image":
        return {}
    if engine != "sdcpp":
        raise ValueError("Image-to-image needs the stable-diffusion.cpp engine.")
    if png is None:
        raise ValueError("The source image is no longer available (the server restarted) - submit it again.")
    return {"init_image": base64.b64encode(png).decode(), "strength": job.strength}
