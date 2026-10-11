"""Thin async wrapper around ComfyUI's local HTTP API; like ollama_client.py, no FastAPI/DB knowledge.

ComfyUI is node-graph based: a generation is a "workflow" JSON graph (checkpoint loader -> text encodes ->
empty latent -> sampler -> VAE decode -> save image) POSTed to /prompt, which returns a prompt_id at once;
the result comes later from GET /history/{prompt_id}. _WORKFLOW_TEMPLATE is a standard SD1.5-style
txt2img graph: a default, not universal (other checkpoints may need another shape).

A job is stateful per host: its prompt_id exists only on the host that received it, so get_history /
fetch_image_bytes must reuse that host, unlike Ollama's per-request routing. Only submit_job and
list_checkpoints (host-agnostic) use comfyui_pool's failover; the rest take an explicit `host`
(image_generation_service, the only caller, threads it through).
"""

import copy
import uuid

import httpx

from app.config import APP_NAME
from app.services import comfyui_pool

_REQUEST_TIMEOUT = httpx.Timeout(30.0, connect=5.0)

# Node ids are plain labels (ComfyUI export numbering), only used to wire the graph and to tell _run_job
# which node ("9") holds the final image.
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
# The node whose output is the finished image (image_generation_service reads
# history["outputs"][SAVE_IMAGE_NODE_ID]["images"]).
SAVE_IMAGE_NODE_ID = "9"


class ComfyUIError(Exception):
    """Raised when ComfyUI is unreachable or errors, for a clean HTTP error or job error_message (like
    ollama_client.OllamaError)."""


def _build_workflow(params: dict) -> dict:
    """A fresh copy of _WORKFLOW_TEMPLATE with `params` (prompt, negative_prompt, checkpoint, width, height,
    steps, cfg, seed) filled in; never mutates the shared template (concurrent jobs would corrupt it).
    """
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
    """Submits a txt2img workflow from `params`, with pool failover (safe: nothing exists yet). Returns
    (host, prompt_id); reuse this `host` for the job (see module docstring).
    """
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


async def cancel_job(host: str, prompt_id: str) -> None:
    """Best-effort: drops a queued prompt and interrupts what is executing on `host`."""
    try:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
            await client.post(f"{host}/queue", json={"delete": [prompt_id]})
            await client.post(f"{host}/interrupt")
    except httpx.HTTPError:
        pass


async def get_history(host: str, prompt_id: str) -> dict | None:
    """{"outputs": {...}} once `prompt_id` finished, else None (ComfyUI adds it to /history only when done).
    Always the submitting `host`, never failed over.
    """
    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
        try:
            resp = await client.get(f"{host}/history/{prompt_id}")
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ComfyUIError(f"Could not reach ComfyUI at {host}: {exc}") from exc
    entry = resp.json().get(prompt_id)
    return entry if entry else None


async def fetch_image_bytes(host: str, filename: str, subfolder: str, image_type: str) -> bytes:
    """Downloads one output image's bytes, from a get_history outputs[...]["images"][i] entry, on the job's host."""
    params = {"filename": filename, "subfolder": subfolder, "type": image_type}
    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
        try:
            resp = await client.get(f"{host}/view", params=params)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ComfyUIError(f"Could not fetch image from ComfyUI at {host}: {exc}") from exc
    return resp.content


async def list_checkpoints() -> list[str]:
    """Every checkpoint ComfyUI has, for the generation page's picker. Host-agnostic and pool-routed with
    failover; hosts may differ (as in ollama_admin's multi-host pull/delete).
    """

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
