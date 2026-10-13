"""Matricxon settings Save without a restart: most values change live through Matricxon's
PUT /api/runtime-settings (../matricxon/app/routers/runtime_settings_router.py); only the ones fixed
at process start still restart it (matricxon_process.apply_local_config)."""

import httpx

from app.schemas import MatricxonServerConfig
from app.services import matricxon_http
from app.services.matricxon_pool import LOCAL_MATRICXON_HOST

# Read once when Matricxon starts: changing one needs a restart. Label: what the form calls it.
RESTART_FIELDS = {
    "project_dir": "Project directory",
    "models_path": "Models directory",
    "prompt_cache_path": "Saved chat caches directory",
    "advertise_url": "Advertised address",
    "cluster_role": "Cluster mode",
    "device": "Compute device",
    "torch_threads": "CPU threads",
}

# Changed live. pAIring field -> Matricxon runtime-settings field.
LIVE_FIELDS = {
    "max_loaded_models": "max_loaded_models",
    "keep_alive_minutes": "default_keep_alive_seconds",
    "memory_safety_margin": "memory_safety_margin",
    "prompt_cache_slots": "prompt_cache_slots",
    "prompt_cache_budget_mb": "prompt_cache_budget_mb",
    "max_decode_batch": "max_decode_batch",
    "log_level": "log_level",
    "enable_quantized_native_compute": "enable_quantized_native_compute",
    "gemv_backend": "gemv_backend",
    "gpu_weight_mode": "gpu_weight_mode",
}

_TIMEOUT_SECONDS = 60.0  # switching kernels can build the native library


class LiveApplyUnsupportedError(RuntimeError):
    """This Matricxon has no runtime-settings route (an older version): restart instead."""


class MatricxonSettingsPlan:
    """What a Save from `old` to `new` needs: a restart, a live update, or nothing."""

    def __init__(self, old: MatricxonServerConfig, new: MatricxonServerConfig) -> None:
        self._old = old
        self._new = new

    def _changed(self, fields: dict[str, str]) -> list[str]:
        return [f for f in fields if getattr(self._old, f) != getattr(self._new, f)]

    @property
    def restart_fields(self) -> list[str]:
        changed = self._changed(RESTART_FIELDS)
        # Cluster mode: live changes would reach this node only, so keep the restart path.
        if self._new.cluster_role != "off" and self._changed(LIVE_FIELDS):
            changed.append("cluster_role")
        return list(dict.fromkeys(changed))

    @property
    def restart_labels(self) -> list[str]:
        return [RESTART_FIELDS[f] for f in self.restart_fields]

    @property
    def restart_required(self) -> bool:
        return bool(self.restart_fields)

    def live_payload(self) -> dict[str, object]:
        """Changed live values in Matricxon's names; None asks for Matricxon's default."""
        payload: dict[str, object] = {}
        for field in self._changed(LIVE_FIELDS):
            value = getattr(self._new, field)
            if field == "keep_alive_minutes" and value is not None:
                value *= 60
            payload[LIVE_FIELDS[field]] = value
        return payload


async def push_live(payload: dict[str, object]) -> dict[str, list[str]]:
    """Sends the values to the local Matricxon; returns what it applied now and at next load."""
    try:
        async with matricxon_http.client(timeout=_TIMEOUT_SECONDS) as client:
            response = await client.put(f"{LOCAL_MATRICXON_HOST}/api/runtime-settings", json=payload)
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Could not reach Matricxon: {exc}") from exc
    if response.status_code in (404, 405):
        raise LiveApplyUnsupportedError("This Matricxon version cannot change settings live.")
    if response.status_code >= 400:
        raise RuntimeError(f"Matricxon refused the new settings ({response.status_code}): {response.text[:200]}")
    data = response.json()
    return {"applied_now": data.get("applied_now", []), "applied_next_load": data.get("applied_next_load", [])}
