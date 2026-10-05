"""Thin async client for stable-diffusion.cpp's `sd-server`: generation through its native async job API
(/sdcpp/v1/img_gen + /jobs/{id}, which — unlike the blocking A1111 txt2img — can be cancelled mid-run), and the
model list through the A1111-compatible /sdapi/v1/sd-models."""

import base64

import httpx

from app.services import sdcpp_pool

_REQUEST_TIMEOUT = httpx.Timeout(30.0, connect=5.0)
_LIST_TIMEOUT = httpx.Timeout(10.0, connect=5.0)


class SdCppError(Exception):
    """sd-server unreachable or returned an error — becomes a job's error_message / a clean HTTP error."""


async def submit_job(params: dict) -> tuple[str, str]:
    """Submits one txt2img job via the native async API (/sdcpp/v1/img_gen), pool-routed with failover (safe: nothing
    exists yet). Returns (host, job_id) — a job only exists on the host that accepted it, so every later call about
    it (get_job, cancel) must use that exact host. The checkpoint isn't sent: sd-server serves exactly the model it
    was launched with."""
    payload = {
        "prompt": params["prompt"],
        "negative_prompt": params.get("negative_prompt") or "",
        "width": params["width"],
        "height": params["height"],
        "seed": params["seed"],
        "batch_count": 1,
        "sample_params": {"sample_steps": params["steps"], "guidance": {"txt_cfg": params["cfg"]}},
    }
    if params.get("init_image"):  # image-to-image: a base64 PNG sized width x height, plus how far to move from it
        payload["init_image"] = params["init_image"]
        payload["strength"] = params["strength"]

    async def _call(host: str) -> tuple[str, str]:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
            resp = await client.post(f"{host}/sdcpp/v1/img_gen", json=payload)
            resp.raise_for_status()
            return host, resp.json()["id"]

    try:
        return await sdcpp_pool.call_with_failover(_call)
    except httpx.HTTPError as exc:
        raise SdCppError(f"Could not submit the job to stable-diffusion.cpp: {exc}") from exc


async def get_job(host: str, job_id: str) -> dict:
    """The job's current state ({"status": queued|generating|completed|failed|cancelled, ...}) on `host`."""
    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
        try:
            resp = await client.get(f"{host}/sdcpp/v1/jobs/{job_id}")
            if resp.status_code in (404, 410):
                raise SdCppError("The engine no longer knows this job (was stable-diffusion.cpp restarted?).")
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise SdCppError(f"Could not reach stable-diffusion.cpp at {host}: {exc}") from exc
    return resp.json()


def image_from_job(job: dict) -> bytes:
    """The PNG bytes of a completed job."""
    try:
        return base64.b64decode(job["result"]["images"][0]["b64_json"])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise SdCppError("Unexpected job result shape from stable-diffusion.cpp.") from exc


async def cancel_job(host: str, job_id: str) -> None:
    """Best-effort: stops a queued or generating job. A job that already finished (409/410) is fine."""
    try:
        async with httpx.AsyncClient(timeout=_LIST_TIMEOUT) as client:
            await client.post(f"{host}/sdcpp/v1/jobs/{job_id}/cancel")
    except httpx.HTTPError:
        pass


async def list_checkpoints() -> list[str]:
    """The model(s) the server has loaded (one, by design) — for the Images page's picker."""

    async def _call(host: str) -> list:
        async with httpx.AsyncClient(timeout=_LIST_TIMEOUT) as client:
            resp = await client.get(f"{host}/sdapi/v1/sd-models")
            resp.raise_for_status()
            return resp.json()

    try:
        data = await sdcpp_pool.call_with_failover(_call)
    except httpx.HTTPError as exc:
        raise SdCppError(f"Could not reach stable-diffusion.cpp: {exc}") from exc
    try:
        return [entry["model_name"] for entry in data]
    except (KeyError, TypeError) as exc:
        raise SdCppError("Unexpected /sd-models response shape from stable-diffusion.cpp.") from exc
