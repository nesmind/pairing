"""Extra files a split image model needs beside its diffusion weights (text encoder, VAE), fetched with the
model and passed to sd-server automatically (no typing --llm/--vae). They live in a `_companions` folder next
to the model. Add a recipe to RECIPES for another model family.
"""

import shutil
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from app.services.hf_download import stream_file

FOLDER = "_companions"


@dataclass(frozen=True)
class Companion:
    flag: str
    repo: str
    filename: str
    label: str
    size_gb: float

    @property
    def local_name(self) -> str:
        return self.filename.rsplit("/", 1)[-1]

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/main/{self.filename}"


_Z_IMAGE = (
    Companion(
        "--llm",
        "unsloth/Qwen3-4B-Instruct-2507-GGUF",
        "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        "Qwen3-4B text encoder",
        2.5,
    ),
    Companion("--vae", "Comfy-Org/z_image_turbo", "split_files/vae/ae.safetensors", "Z-Image VAE", 0.34),
)
RECIPES: tuple[tuple[tuple[str, ...], tuple[Companion, ...]], ...] = ((("z_image", "z-image"), _Z_IMAGE),)


class ImageCompanions:
    @staticmethod
    def dir_for(model_path: Path) -> Path:
        return model_path.parent / FOLDER

    @staticmethod
    def for_model(model_path: Path) -> tuple[Companion, ...]:
        name = model_path.name.lower()
        return next((files for keys, files in RECIPES if any(k in name for k in keys)), ())

    @classmethod
    def missing(cls, model_path: Path) -> list[Companion]:
        return [c for c in cls.for_model(model_path) if not (cls.dir_for(model_path) / c.local_name).is_file()]

    @classmethod
    def args(cls, model_path: Path) -> dict[str, str]:
        """{"--llm": path, "--vae": path} for companions already on disk."""
        paths = {c.flag: cls.dir_for(model_path) / c.local_name for c in cls.for_model(model_path)}
        return {flag: str(path) for flag, path in paths.items() if path.is_file()}

    @classmethod
    async def download(cls, model_path: Path, flags: list[str], proxy_url: str | None) -> AsyncIterator[dict]:
        """Downloads the missing companions among `flags` (the user's choice) as progress events; the last is
        {"error":...} on failure."""
        for c in cls.missing(model_path):
            if c.flag not in flags:
                continue
            async for event in stream_file(c.url, cls.dir_for(model_path) / c.local_name, proxy_url, c.label):
                yield event
                if "error" in event:
                    return

    @classmethod
    def remove(cls, model_path: Path) -> None:
        shutil.rmtree(cls.dir_for(model_path), ignore_errors=True)
