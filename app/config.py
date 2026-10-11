"""Central configuration, read from environment variables with local-dev defaults.

Nothing else should read `os.environ` directly. On a real deployment, copy `.env.example` to `.env` and edit that.
"""

import logging
import os
import secrets
from pathlib import Path

from dotenv import load_dotenv

# Load `.env` into the environment; fills only unset values (no-op under Docker/systemd).
load_dotenv()

# Project root (parent of `app/`); keeps paths absolute whatever the launch directory.
BASE_DIR = Path(__file__).resolve().parent.parent

# Default model for a conversation that names none (see settings_service.get_default_model).
# Falls back to the smallest app/model_catalog.py entry.
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "hf.co/google/gemma-4-E2B-it-qat-q4_0-gguf:gemma-4-E2B_q4_0-it")

# Embedding model for RAG (app/rag.py); separate from the chat model, pulled from Settings > Knowledge.
# Options are in app/model_catalog.py's EMBEDDING_CATALOG. Default nomic-embed-text; all-MiniLM-L6-v2 is lighter.
# Switching requires re-ingesting documents.
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "hf.co/nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0")

# The embedding model's trained context window. Left unset, the engine's larger default gets clamped
# with a warning and can cause errors on some inputs (see app/services/inference_client.py:embed).
# A 1000-character chunk is about 150-250 tokens, well under it.
EMBEDDING_NUM_CTX = int(os.environ.get("EMBEDDING_NUM_CTX", "512"))

# Web server
# Host/port this app listens on (not the ML engine's).
APP_HOST = os.environ.get("APP_HOST", "0.0.0.0")
APP_PORT = int(os.environ.get("APP_PORT", "8000"))

# Set by app.services.instance_service on sibling processes ("local instances"); 0 on the primary.
# Only the primary spawns/reconciles siblings.
INSTANCE_INDEX = int(os.environ.get("PAIRING_INSTANCE_INDEX", "0"))
IS_PRIMARY = INSTANCE_INDEX == 0

# This build's version (a constant); shown in Settings > System and near the logo.
APP_VERSION = "0.1"

# The display name (a constant): single source for UI/API occurrences of "pAIring". Templates read it
# via `templates.env.globals["app_name"]`, plain JS via base.html's `window.APP_NAME`.
# Not used for internal identifiers (e.g. the "pAIring-server" process title).
APP_NAME = "pAIring"

# Signs the session cookie; anyone knowing it can forge sessions, so set a real secret in .env.
# The random fallback avoids a shared key but logs everyone out on each restart.
SECRET_KEY = os.environ.get("SECRET_KEY")
if not SECRET_KEY:
    SECRET_KEY = secrets.token_hex(32)
    logging.getLogger("llama_chat").warning(
        "SECRET_KEY not set in .env — using a random key that changes every "
        "restart (logging everyone out each time). Set SECRET_KEY before "
        "exposing this app beyond your own machine.",
    )

# Marks the session cookie Secure. Off by default so plain-HTTP local setups work; set
# SESSION_COOKIE_SECURE=true under HTTPS, or the cookie is sent in cleartext.
SESSION_COOKIE_SECURE = os.environ.get("SESSION_COOKIE_SECURE", "false").strip().lower() == "true"
if not SESSION_COOKIE_SECURE:
    logging.getLogger("llama_chat").warning(
        "SESSION_COOKIE_SECURE not set in .env — session cookie sent over plain HTTP. Set "
        "SESSION_COOKIE_SECURE=true once this app is served over HTTPS.",
    )

# Storage
# SQLite file holding conversations, messages, settings and RAG documents; moving servers means copying the folder.
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
# DATABASE_URL derives from DATA_DIR only when unset; an explicit one wins. To relocate an instance,
# override both.
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{DATA_DIR / 'pAIring.db'}")

# Auto-migrate the schema on every startup. With several instances on one database, set false on
# all and run `alembic upgrade head` once first: concurrent cold-start migrations can fail.
# See ROADMAP.md "Scaling to more users".
AUTO_MIGRATE = os.environ.get("AUTO_MIGRATE", "true").strip().lower() != "false"

# Folder where the admin drops RAG files (.txt/.md/.pdf): the knowledge base's source of truth,
# synced by app/rag.py on startup and on "Rescan". Top-level so it is easy to find.
KNOWLEDGE_DIR = Path(os.environ.get("KNOWLEDGE_DIR", BASE_DIR / "knowledge"))
KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

# Per-message attachments (images for vision models; text/PDF/docx folded into the prompt, see
# app/services/chat_attachment_service.py). Separate from KNOWLEDGE_DIR (RAG).
ATTACHMENTS_DIR = Path(os.environ.get("ATTACHMENTS_DIR", BASE_DIR / "attachments"))
ATTACHMENTS_DIR.mkdir(parents=True, exist_ok=True)

# Fixed for now (unlike RAG's max_file_mb, see settings_service.get_rag_limits).
MAX_ATTACHMENT_MB = 10

# Per-message caps, combinable (chat_attachment_service.save_attachments): documents and images.
MAX_ATTACHMENT_DOCUMENTS = 5
MAX_ATTACHMENT_IMAGES = 1

# Profile pictures from Settings > Account (app/services/avatar_service.py); only the filename is stored.
AVATARS_DIR = Path(os.environ.get("AVATARS_DIR", BASE_DIR / "avatars"))
AVATARS_DIR.mkdir(parents=True, exist_ok=True)
MAX_AVATAR_MB = 2

# Image generation (ComfyUI)
# ComfyUI's HTTP API address (app/services/comfyui_client.py); a deployment fact. How to launch it
# is an admin AppSetting (settings_service.get_comfyui_config).
COMFYUI_HOST = os.environ.get("COMFYUI_HOST", "http://localhost:8188")
# stable-diffusion.cpp's `sd-server` (default image engine, app/services/sdcpp_client.py); its port derives from this
# URL.
SDCPP_HOST = os.environ.get("SDCPP_HOST", "http://localhost:8189")

# Generated images folder; only relative paths are stored in the DB.
IMAGES_DIR = Path(os.environ.get("IMAGES_DIR", BASE_DIR / "generated_images"))
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Local-mode auto-install (Settings > External servers)
# Tested tags, never "latest": bump them on purpose after verifying. Read only by the installers in app/services/.
OLLAMA_GITHUB_REPO = "ollama/ollama"
OLLAMA_PINNED_VERSION = "v0.33.3"
COMFYUI_GITHUB_REPO = "comfyanonymous/ComfyUI"
COMFYUI_PINNED_VERSION = "v0.35.1"
SDCPP_GITHUB_REPO = "leejet/stable-diffusion.cpp"
SDCPP_PINNED_VERSION = "master-929-3f8527a"
MATRICXON_GITHUB_REPO = "nesmind/matricxon"
# Exception: Matricxon is developed alongside this app, so installs default to its newest main branch.
# A tag or branch can be set in Settings (MatricxonServerConfig.install_version).
MATRICXON_DEFAULT_VERSION = "main"

# Where auto-installed engines go; outside BASE_DIR so git status never sees them.
EXTERNAL_DIR = Path(os.environ.get("EXTERNAL_DIR", BASE_DIR / "external"))
EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

# Generation defaults
# Defaults for the Settings page's adjustable LLM params. New conversations start with them;
# per-conversation overrides are sent to the engine (app/services/inference_client.py, app/routers/settings.py).
DEFAULT_GENERATION_PARAMS = {
    # Token-choice randomness: 0 = greedy, higher = more creative. Engine default 0.8.
    "temperature": 0.8,
    # Nucleus sampling: only tokens within cumulative probability top_p.
    "top_p": 0.9,
    # Only the top_k most likely tokens.
    "top_k": 40,
    # Penalizes recently seen tokens; 1.0 = off.
    "repeat_penalty": 1.1,
    # Context window in tokens; kept conservative since it applies to every model.
    "num_ctx": 4096,
    # Max tokens per reply; -1 = no limit.
    "num_predict": 1024,
    # Random seed; -1 = new each time.
    "seed": -1,
    # Document chunks retrieved per question from the shared knowledge base (no per-chat switch).
    # No effect while the base is empty or no embedding model exists.
    "rag_top_k": 4,
    # Persona/rules/skill slots this conversation turned off (app/notes.py); empty = user defaults.
    "disabled_default_notes": [],
}

# Fallback where a conversation's rag_top_k is unavailable (app.rag.retrieve_context).
RAG_TOP_K = DEFAULT_GENERATION_PARAMS["rag_top_k"]

# Chunk size in characters (not tokens) and overlap between chunks. Kept well under EMBEDDING_NUM_CTX
# (512): non-Latin scripts tokenize ~4x worse (measured for Hebrew). app.rag.ingest_document_stream skips
# chunks that still overflow.
RAG_CHUNK_SIZE = 500
RAG_CHUNK_OVERLAP = 75

# Theme catalogs live in app/theme_config.py (split out for the file-size cap).
