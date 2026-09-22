"""
Thin async wrapper around ComfyUI's local HTTP API — mirrors
app/services/ollama_client.py's own shape/boundary (no FastAPI/DB
knowledge here, just "how to talk to ComfyUI").

ComfyUI's API is node-graph based, not a plain "one request in, one
response out" endpoint like Ollama's /api/chat: a generation is
submitted as a full "workflow" — a JSON graph of nodes wired together
(checkpoint loader -> positive/negative text encode -> empty latent ->
sampler -> VAE decode -> save image) — via POST /prompt, which returns
immediately with a prompt_id; the actual result is fetched later via
GET /history/{prompt_id} once ComfyUI finishes it. _WORKFLOW_TEMPLATE
below is a standard SD1.5-style txt2img graph — a reasonable default,
not a universal one (see this feature's own plan notes: a different
checkpoint/install may need a different graph shape).

Unlike Ollama's chat_stream, a ComfyUI job is *stateful per host*: a
prompt_id returned by submit_job only ever exists on whichever host
actually received it — get_history/fetch_image_bytes for that job must
keep using that exact same host, never re-pick from the pool the way
Ollama's per-request routing can. submit_job (host not yet chosen) and
list_checkpoints (host-agnostic, any configured host's own catalog) are
the only two calls that go through app.services.comfyui_pool's
failover; everything else takes `host` as an explicit, already-decided
parameter (see app.services.image_generation_service, the only caller,
which threads the one host a job was submitted to through its own
lifetime).
"""

import copy
import uuid

import httpx

from app.config import APP_NAME
from app.services import comfyui_pool

_REQUEST_TIMEOUT = httpx.Timeout(30.0, connect=5.0)

# Node ids are plain, human-picked labels (matching ComfyUI's own
# exported-workflow numbering conventions) — not meaningful beyond
# wiring the graph together and letting _run_job below know which node's
# output ("9") is the final image.
_WORKFLOW_TEMPLATE = {
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 0,
            "steps": 20,
            "cfg": 7.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
            "model": ["4", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0],
        },
    },
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ""}},
    "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["4", 1]}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["4", 1]}},
    "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
    "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": APP_NAME, "images": ["8", 0]}},
}
# The node whose output is the finished image — image_generation_service
# reads history["outputs"][SAVE_IMAGE_NODE_ID]["images"] once complete.
SAVE_IMAGE_NODE_ID = "9"


class ComfyUIError(Exception):
    """Raised when ComfyUI is unreachable or returns an error, so callers
    can turn it into a clean HTTP error (or a job's own error_message)
    instead of a raw stack trace — mirrors ollama_client.OllamaError."""


def _build_workflow(params: dict) -> dict:
    """Returns a fresh copy of _WORKFLOW_TEMPLATE with `params` (prompt,
    negative_prompt, checkpoint, width, height, steps, cfg, seed)
    substituted in — never mutates the shared module-level template in
    place, since concurrent jobs building off the same dict object would
    otherwise corrupt each other."""
    workflow = copy.deepcopy(_WORKFLOW_TEMPLATE)
    workflow["4"]["inputs"]["ckpt_name"] = params["checkpoint"]
    workflow["5"]["inputs"]["width"] = params["width"]
    workflow["5"]["inputs"]["height"] = params["height"]
    workflow["6"]["inputs"]["text"] = params["prompt"]
    workflow["7"]["inputs"]["text"] = params.get("negative_prompt") or ""
    workflow["3"]["inputs"]["steps"] = params["steps"]
    workflow["3"]["inputs"]["cfg"] = params["cfg"]
    workflow["3"]["inputs"]["seed"] = params["seed"]
    return workflow


async def submit_job(params: dict) -> tuple[str, str]:
    """Submits a txt2img workflow built from `params`, routed across the
    pool with failover (safe here — nothing has been created yet, so a
    retry on a different host duplicates no state). Returns (host,
    prompt_id) — the caller must keep using this exact `host` for every
    later call about this same job (see this module's own docstring)."""
    payload = {"prompt": _build_workflow(params), "client_id": uuid.uuid4().hex}

    async def _call(host: str) -> tuple[str, str]:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
            resp = await client.post(f"{host}/prompt", json=payload)
            resp.raise_for_status()
            return host, resp.json()["prompt_id"]

    try:
        return await comfyui_pool.call_with_failover(_call)
    except httpx.HTTPError as exc:
        raise ComfyUIError(f"Could not submit job to ComfyUI: {exc}") from exc


async def get_history(host: str, prompt_id: str) -> dict | None:
    """Returns {"outputs": {...}} once `prompt_id` has finished, or None
    while it's still queued/running (an absent or empty entry — ComfyUI
    only adds a prompt_id to /history once it's done). Always the exact
    `host` this job was submitted to — see this module's own docstring
    for why this never fails over to a different host."""
    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
        try:
            resp = await client.get(f"{host}/history/{prompt_id}")
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ComfyUIError(f"Could not reach ComfyUI at {host}: {exc}") from exc
    entry = resp.json().get(prompt_id)
    return entry if entry else None


async def fetch_image_bytes(host: str, filename: str, subfolder: str, image_type: str) -> bytes:
    """Downloads one output image's raw bytes, as referenced by a
    get_history() result's outputs[...]["images"][i] entry — same exact
    `host` that job was submitted to, for the same reason get_history is."""
    params = {"filename": filename, "subfolder": subfolder, "type": image_type}
    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
        try:
            resp = await client.get(f"{host}/view", params=params)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ComfyUIError(f"Could not fetch image from ComfyUI at {host}: {exc}") from exc
    return resp.content


async def list_checkpoints() -> list[str]:
    """Returns every checkpoint ComfyUI currently has installed, for the
    generation page's model picker. Host-agnostic (any configured host's
    own catalog) — pool-routed with failover, same "not guaranteed
    uniform across hosts" caveat app.services.ollama_admin's own model
    pull/delete already documents for the identical multi-host
    situation."""

    async def _call(host: str) -> dict:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
            resp = await client.get(f"{host}/object_info/CheckpointLoaderSimple")
            resp.raise_for_status()
            return resp.json()

    try:
        data = await comfyui_pool.call_with_failover(_call)
    except httpx.HTTPError as exc:
        raise ComfyUIError(f"Could not reach ComfyUI: {exc}") from exc
    try:
        return data["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
    except (KeyError, IndexError) as exc:
        raise ComfyUIError("Unexpected /object_info response shape from ComfyUI.") from exc
