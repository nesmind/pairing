"""
Central configuration for the whole app.

Every setting the app needs to know about a specific server (where Ollama lives, which port to listen on, where to
put its database file, ...) is read here from environment variables, with a sensible default for local development.
Nothing else in the codebase should read `os.environ` directly — this keeps "things that change when you move to a
new server" in exactly one place, which is what makes the project portable.

To change a value on a real deployment, copy `.env.example` to `.env` and
edit it there instead of editing this file.
"""

import logging
import os
import secrets
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a `.env` file (if one exists) into the process
# environment. This is a no-op in production setups that already inject
# real environment variables (e.g. Docker, systemd) — it only fills in
# values that aren't already set.
load_dotenv()

# Absolute path to the project's root folder (the parent of the `app/`
# package this file lives in). Used to build absolute paths below so the
# app works no matter what directory it's *launched* from.
BASE_DIR = Path(__file__).resolve().parent.parent

# Which model Ollama should use by default when a conversation doesn't specify one (e.g. a brand-new chat, or the
# starter admin account's very first message before any per-user/system default has been saved — see
# app/services/settings_service.get_default_model). Falls back to the smallest entry in app/model_catalog.py so a
# fresh install's first chat at least names a real, currently-offered model rather than a stale/removed one.
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "hf.co/google/gemma-4-E2B-it-qat-q4_0-gguf:gemma-4-E2B_q4_0-it")

# Model used to turn text into embedding vectors for the RAG feature (see app/rag.py) — a *separate*, smaller
# model from the chat model, specialized for similarity search, not text generation. Both options this app
# knows about live in app/model_catalog.py's EMBEDDING_CATALOG, pullable from Settings > Knowledge like a chat
# model, same UX as DEFAULT_MODEL above. Defaults to nomic-embed-text (~137M params) for retrieval quality; the
# smaller/faster all-MiniLM-L6-v2 (~23M params) is offered there too as a lighter alternative for slow/CPU-only
# hardware. Switching later requires re-ingesting existing documents — embeddings from different models aren't
# compatible with each other.
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "hf.co/nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0")

# This model's own trained context window (per its GGUF metadata) — left unspecified, Ollama's own default for an
# embedding call can be larger than what a given model was actually trained with, which it then has to silently
# clamp back down at request time (logged as a "requested context size too large for model" warning). Passing this
# explicitly avoids that mismatch/clamp path entirely, which in practice has been the difference between a clean
# embedding and an internal server error for some inputs — see app/ollama_client.py:embed. RAG_CHUNK_SIZE below
# (1000 characters) comfortably fits within this for ordinary text — a 1000-character English chunk tokenizes to
# roughly 150-250 tokens, well under the limit.
EMBEDDING_NUM_CTX = int(os.environ.get("EMBEDDING_NUM_CTX", "512"))

# --- Web server ----------------------------------------------------------
# Host/port this FastAPI app itself listens on (not Ollama's).
APP_HOST = os.environ.get("APP_HOST", "0.0.0.0")
APP_PORT = int(os.environ.get("APP_PORT", "8000"))

# Set only by app.services.instance_service when it spawns a sibling
# process for the "local instances" feature (Settings > System) — 0 on
# the one process a human ever actually starts (via scripts/start.sh or
# systemd). IS_PRIMARY gates which process is allowed to spawn/reconcile
# siblings, so a sibling never tries to spawn its own siblings.
INSTANCE_INDEX = int(os.environ.get("PAIRING_INSTANCE_INDEX", "0"))
IS_PRIMARY = INSTANCE_INDEX == 0

# This build's own version — the codebase's, not a per-deployment
# setting, so it's a plain constant rather than read from the
# environment. Shown in Settings > System and near the logo in the UI.
APP_VERSION = "0.1"

# The app's own display name — a plain constant, not a per-deployment setting, same reasoning as APP_VERSION
# above. The single source of truth for every user/API-facing occurrence of "pAIring" (page titles, the login/
# chat page's own copy, the D3 workflow diagram's node labels, error/status messages) — see app/main.py's
# `templates.env.globals["app_name"]` for how HTML templates read this, and base.html's `window.APP_NAME` for how
# plain (non-Jinja-rendered) JS files do. Deliberately NOT used for internal-only identifiers that merely happen
# to be derived from it (the "pAIring-server" process title app/services/instance_process.py matches against,
# ComfyUI's ComfyUI-side install directory name) — changing those is a much bigger, riskier operational change
# than relabeling UI text, and out of scope here.
APP_NAME = "pAIring"

# Signs the login session cookie (see app/main.py's SessionMiddleware). Anyone who knows this value can forge a valid
# session for any user, so it must be set to a real secret via .env before this app is reachable by anyone but you.
# Falling back to a random value (rather than a fixed hardcoded one) at least avoids every unconfigured install
# sharing the same key — but it also means every restart invalidates all logged-in sessions, which is the visible
# nudge to go set a real one.
SECRET_KEY = os.environ.get("SECRET_KEY")
if not SECRET_KEY:
    SECRET_KEY = secrets.token_hex(32)
    logging.getLogger("llama_chat").warning(
        "SECRET_KEY not set in .env — using a random key that changes every "
        "restart (logging everyone out each time). Set SECRET_KEY before "
        "exposing this app beyond your own machine.",
    )

# Whether the session cookie (app/main.py's SessionMiddleware) is marked Secure — off by default so a plain-HTTP local
# deployment (this app's common case, no TLS terminator) isn't silently broken out of the box. Set
# SESSION_COOKIE_SECURE=true once this app is served over HTTPS, directly or behind a reverse proxy — otherwise the
# signed session cookie is sent in cleartext and anyone on the network path can capture it and hijack the session.
SESSION_COOKIE_SECURE = os.environ.get("SESSION_COOKIE_SECURE", "false").strip().lower() == "true"
if not SESSION_COOKIE_SECURE:
    logging.getLogger("llama_chat").warning(
        "SESSION_COOKIE_SECURE not set in .env — session cookie sent over plain HTTP. Set "
        "SESSION_COOKIE_SECURE=true once this app is served over HTTPS.",
    )

# --- Storage -------------------------------------------------------------
# SQLite database file holding conversations, messages, settings, and uploaded RAG documents. SQLite is a single
# file, so "moving the app to another server" is just "copy the folder" — there's no separate database service to
# install, configure, or migrate data into.
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
# DATABASE_URL only derives from DATA_DIR when DATABASE_URL itself is unset. If .env (or the
# environment) already has an explicit DATABASE_URL line, it wins outright — setting DATA_DIR alone
# will NOT relocate the database. To point a whole instance (db + files) somewhere else, e.g. for an
# isolated test run, override both DATA_DIR and DATABASE_URL explicitly.
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{DATA_DIR / 'pAIring.db'}")

# Whether the app migrates its own schema automatically on every startup — on by default, so a fresh single-instance
# install just works with no separate step. Running more than one app instance against the same database? Set this
# to false on every instance and run the migration once, separately, first (`alembic upgrade head`) — instances
# racing to migrate concurrently on cold start can fail outright, not just be slow. See ROADMAP.md's "Scaling to
# more users" section.
AUTO_MIGRATE = os.environ.get("AUTO_MIGRATE", "true").strip().lower() != "false"

# Folder the admin drops RAG source files into directly (.txt/.md/.pdf). This is the single source of truth for the
# knowledge base: on startup, and whenever the "Rescan" button in Settings is used, app/rag.py syncs the database to
# match whatever's currently in this folder — add a file and it becomes searchable, remove one and it stops being
# searchable. Deliberately a plain top-level folder (not under data/) so it's easy to find and drop files into by
# hand, independent of the app's internal SQLite storage.
KNOWLEDGE_DIR = Path(os.environ.get("KNOWLEDGE_DIR", BASE_DIR / "knowledge"))
KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

# Where a chat/channel message's own attachments (an image for a vision model to see, or a text/PDF/docx file whose
# content gets folded into the prompt — see app/services/chat_attachment_service.py) are saved. Deliberately separate
# from KNOWLEDGE_DIR: these are per-message, ad-hoc files tied to one specific Message row, not the shared/personal
# RAG knowledge base scanned as a whole folder.
ATTACHMENTS_DIR = Path(os.environ.get("ATTACHMENTS_DIR", BASE_DIR / "attachments"))
ATTACHMENTS_DIR.mkdir(parents=True, exist_ok=True)

# Fixed rather than admin-configurable for now (unlike RAG's max_file_mb — see app.services.settings_service.
# get_rag_limits) — not worth a settings UI entry until someone actually needs a different ceiling.
MAX_ATTACHMENT_MB = 10

# Per-message caps, combinable (see chat_attachment_service.save_attachments) — up to this many document files
# (text/PDF/docx/etc.) *and* this many images in the same message.
MAX_ATTACHMENT_DOCUMENTS = 5
MAX_ATTACHMENT_IMAGES = 1

# Profile picture uploaded from Settings > Account (see app/services/avatar_service.py) — shown small and round
# beside a sender's messages in a channel; only the filename is stored on the User row itself.
AVATARS_DIR = Path(os.environ.get("AVATARS_DIR", BASE_DIR / "avatars"))
AVATARS_DIR.mkdir(parents=True, exist_ok=True)
MAX_AVATAR_MB = 2

# --- Image generation (ComfyUI) -----------------------------------------
# Where ComfyUI's own HTTP API listens once running (see app/services/comfyui_client.py) — a deployment fact,
# like Ollama's own local address (see app/services/ollama_pool.LOCAL_OLLAMA_HOST), not admin-editable from the
# UI. *How to launch* ComfyUI (venv python + main.py path) is a separate, admin-configurable AppSetting instead
# (see settings_service.get_comfyui_config), settable from Settings live.
COMFYUI_HOST = os.environ.get("COMFYUI_HOST", "http://localhost:8188")

# Where generated images are saved — same idiom as ATTACHMENTS_DIR/KNOWLEDGE_DIR above: a plain folder, with only a
# relative path ever stored in the DB, so moving this later never strands a row.
IMAGES_DIR = Path(os.environ.get("IMAGES_DIR", BASE_DIR / "generated_images"))
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# --- Local-mode auto-install (Settings > External servers) ---------------
# Real, tested tags — deliberately never "latest": a moving target would make behavior unpredictable across
# restarts/reinstalls, and an untested new release could break silently. Bump these on purpose once a newer
# version has actually been verified to work with this app — see app/services/ollama_installer.py /
# comfyui_installer.py, the only readers.
OLLAMA_GITHUB_REPO = "ollama/ollama"
OLLAMA_PINNED_VERSION = "v0.33.3"
COMFYUI_GITHUB_REPO = "comfyanonymous/ComfyUI"
COMFYUI_PINNED_VERSION = "v0.35.1"
MATRICXON_GITHUB_REPO = "nesmind/matricxon"
MATRICXON_PINNED_VERSION = "v1.2"

# Where a locally auto-installed Ollama/ComfyUI ends up — kept separate from BASE_DIR's own app code so this app's
# own git status/deploys never see these as project files.
EXTERNAL_DIR = Path(os.environ.get("EXTERNAL_DIR", BASE_DIR / "external"))
EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

# --- Generation defaults ---------------------------------------------------
# These are the out-of-the-box values for the "adjustable LLM params" the
# Settings page exposes. A brand-new conversation starts with these; the
# user can override them per-conversation from the Settings page, and
# those overrides are what actually gets sent to Ollama on every request
# (see app/ollama_client.py and app/routers/settings.py).
DEFAULT_GENERATION_PARAMS = {
    # Randomness of token choice. 0 = deterministic/greedy, higher = more
    # creative/less predictable. Ollama's own default is 0.8.
    "temperature": 0.8,
    # Nucleus sampling: only consider tokens whose cumulative probability
    # is <= top_p. Lower values narrow the model to "safer" word choices.
    "top_p": 0.9,
    # Only consider the top_k most likely next tokens at each step.
    "top_k": 40,
    # Penalizes tokens that already appeared recently, to discourage the
    # model from repeating itself. 1.0 = no penalty.
    "repeat_penalty": 1.1,
    # Size (in tokens) of the model's context window — how much
    # conversation history it can "see" at once. Must not exceed what the
    # underlying model supports — kept conservative here since it applies
    # across every model, not just whichever one is currently the default.
    "num_ctx": 4096,
    # Maximum number of tokens the model is allowed to generate in a
    # single reply. -1 means "no limit" (model stops on its own).
    "num_predict": 1024,
    # Random seed for reproducible output. -1 means "pick a new random
    # seed every time" (i.e. non-deterministic replies).
    "seed": -1,
    # How many of the most relevant document chunks to retrieve and
    # inject into the prompt per question. Every chat automatically
    # searches the shared knowledge base (app.config.KNOWLEDGE_DIR) —
    # there's no per-conversation on/off switch, only this "how many
    # chunks" knob. More chunks means more potentially-relevant context
    # but also more irrelevant noise and a larger prompt. Retrieval is a
    # silent no-op when the knowledge base is empty or no embedding
    # model is available, so this has no effect until both exist.
    "rag_top_k": 4,
    # Which persona/rules/skill slots this conversation has explicitly
    # turned off from the chat page's icons (see app/notes.py) — empty
    # by default, meaning every slot falls back to the user's default
    # note for that type.
    "disabled_default_notes": [],
}

# Fallback used only where a conversation's own rag_top_k isn't
# available (e.g. app.rag.retrieve_context's default argument).
RAG_TOP_K = DEFAULT_GENERATION_PARAMS["rag_top_k"]

# Target size (in characters, not tokens) of each text chunk stored for retrieval, and how much consecutive chunks
# overlap so an idea split across a chunk boundary isn't lost entirely. Kept well under a naive characters-per-token
# estimate: token count depends heavily on script, not just length (EMBEDDING_MODEL's WordPiece tokenizer is
# overwhelmingly Latin-script and inflates badly for Hebrew and other non-Latin text — measured ~4x more tokens for
# the same character count). EMBEDDING_NUM_CTX above (512) is a hard ceiling the model was trained with; this size
# leaves real margin even for token-inflating scripts, and app.rag.ingest_document_stream skips (rather than fails
# the whole document over) any individual chunk that still overflows despite this.
RAG_CHUNK_SIZE = 500
RAG_CHUNK_OVERLAP = 75

# Theme catalogs (code-block syntax theme + app-wide UI theme) live in app/theme_config.py, not here — split out
# once this file hit CLAUDE.md's line cap; see that file's own docstring for why they're kept independent.
