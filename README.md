<br>
<div align="center">

<img src="docs/logo.svg" alt="pAIring" height="86">

</div>


# pAIring

Private AI ring is a self-hosted, production-ready platform for deploying private LLMs & image generation models across a single node, a team, a department, or an organization — role-based access, load balancing failover across servers, and rock solid UI validations to avoid configuration mistakes. Start chatting with open sourced LLMs, create images and much more. Supports every existing public GGUF model out of the box, browsable from Hugging Face or your own private repository. Nothing leaves your network.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # then set a real SECRET_KEY — see README.md
.venv/bin/python run.py
```

Or just run `bash scripts/install.sh` (Linux/macOS) — same steps with extra validations.

Open **http://localhost:8000**, log in with the starter account, and
**change that password immediately** (Settings → Users) — whether
you're about to invite three friends or roll this out to a whole team.

Configure it as a system service: `scripts/pairing.service` is a
systemd unit template (Linux) — edit a couple of values and
`systemctl enable` it.

Supports MySQL migration and multi-server deployment.

**After install, open the UI and configure the essentials to get started:**

1. Go to **Settings → External servers** and choose your inference engine. Ollama and Matricxon are supported (Matricxon has many more options but still has some important limitations — see the Matricxon repository for details). Both can be pulled and installed directly from there.
2. After configuring the engine, go to **Settings → Models** and download some models. Embedding models are needed for RAG support, and vision projectors are needed for image understanding in chats and channels.
3. Once the models are ready, go to **Settings → System** and set the important parameters (such as the default vision model).
4. Run a few tests.

## ✨ Features

**Advanced model management**
- A model management system that lets you:
  - Search, browse and download models directly from Hugging Face repositories, with advanced search and visual indexing to make choosing the right model simple.
  - Pull recommended default models right away and get started.
  - Manage installed models simply and efficiently, and get started with a few recommended models through a simple pull request.

**Behavior tuning**
- Supports all the generation parameters (Temperature, Top-P, Top-K, Context window (tokens), and others).
- Behavior tuning is per user: each user can change these values, and they affect only their own chats.

**Chat**
- Real-time streaming replies, with attachments (images + documents, up
  to 5 + 1 combined per message).
- Per-conversation model switching and generation parameters
  (temperature, top-p/k, context window, repeat penalty, reply length).
- Stop/cancel a reply mid-stream, smart or simple auto-titling.

**Real multi-user, with shared Channels**
- Every user has private conversations no one else can see or list.
- **Channels** for shared, group conversations — a team standup, a
  student class testing models, whatever fits — with "live" (instant
  push).
- Per-channel managers — trusted members who can steer that channel's model/settings and pinned notes, without
  needing full admin rights over the whole system.

**Notes — persona / rules / skill**
- Reusable, pinnable notes that shape the system prompt per
  conversation — who the model is, what it must follow, what it's
  helping with — editable right from the chat header.

**Knowledge base (RAG)**
- Drop in `.txt`/`.md`/`.pdf`/`.docx`/`.odt`/`.rtf`/`.html`/`.csv`/`.json`
  and every conversation can automatically search them — a household
  manual or a team's internal docs, same mechanism either way
- Private-per-user and shared/global scopes, with per-user storage
  quotas; answers cite which document they drew on.

**Live dashboards**
- A real usage overview (conversations, messages, users, knowledge
  base).
- An interactive D3 diagram of the app's own request pipeline, click
  any step for detail — genuinely useful documentation.
- Telemetry data via OpenTelemetry (latency, tokens/sec, error
  rate, currently-loaded models) and live CPU/RAM/disk/GPU monitoring
  for the host — the kind of visibility that's nice to have and, for a
  shared/production deployment.

**Robust theme support**
- Robust themes schema, covering the whole app
- Pick one from Settings or the header, with instant live preview.

**Security, taken seriously either way**
- bcrypt password hashing, signed session cookies, login rate limiting,
  timing-safe unknown-username handling.
- Role-based access (admin/user); disabling an account keeps its
  history intact rather than deleting anything.
- SQLite out of the box, MySQL when you need more — switchable live.
- Outbound HTTP proxy support with encrypted-at-rest credentials.

**GPU-aware**
- Live GPUs usage shows up right in the dashboard, alongside CPU/RAM/disk.

**Load balancing**
- Point at extra ML engine (Ollama or Matricxon) servers (more GPUs, more machines) and this app spreads requests across them, with automatic failover if one goes down.
- Can also run multiple copies of itself behind one URL, no reverse proxy needed.
- Or, if you'd rather test it for enterprise use, use a reverse proxy (nginx, Caddy, ...) to balance across a ring of pAIring servers — that works too. All configurable directly from the UI.

**Connectors**
- Extend the system with connectors, which add support for more inference engines, such as cloud services that offer GPU and serverless ML. pAIring is intended to run privately or locally without external services and providers (except for downloading models), but this is a much appreciated optional extra.
- At the moment there is only one connector, for RunPod.

**Guardrails for admin changes**
- Risky settings (switching database, external servers) get tested before they're saved, so a bad edit doesn't take the UI down.
- If something's out of sync with what's actually running (like a hand-edited config file), you get updated.

**Image generation**
- Optional text-to-image Image generation,
  with browsable generation history (requires ComfyUI installed).
  
**Nothing exotic to run**
- Plain HTML/CSS/JS + Tailwind's browser build — no Node, no build
  step, no extra services beyond the app itself.

**Efficient API — fully documented**
- The web UI is a client of pAIring's own JSON REST API — 90+ endpoints covering chat, channels,
  documents/RAG, notes, users, stats, telemetry, and every admin setting, so anything the UI can do, your own
  scripts/automation can do too.
- Live, interactive docs at `/docs` (Swagger UI) and `/redoc`
- `GET /health` reports whether *this* instance can actually serve a chat request right now.

<br>

**Runs on**
- Linux — All features installed.
- macOS also works — install Ollama or Matricxon yourself and point pAIring at it.
  Windows isn't supported at this point.
- x86_64 or arm64, GPU optional — CPU-only works fine, a GPU (NVIDIA or Apple Silicon) just makes replies much faster.
- No fixed RAM requirement — depends on which model(s) you run; Models settings section shows what actually fits your hardware.

<br><br>

## 📸 Screenshots

<table>
  <tr>
    <th width="50%">Chat</th>
    <th width="50%">Usage dashboard</th>
  </tr>
  <tr>
    <td><img src="docs/screenshots/chat.png" alt="Chat" width="100%"></td>
    <td><img src="docs/screenshots/stats-overview.png" alt="Stats overview" width="100%"></td>
  </tr>
  <tr>
    <th width="50%">Visual chat workflow (D3)</th>
    <th width="50%">System resource monitor</th>
  </tr>
  <tr>
    <td><img src="docs/screenshots/stats-workflow.png" alt="Workflow diagram" width="100%"></td>
    <td><img src="docs/screenshots/stats-system.png" alt="System dashboard" width="100%"></td>
  </tr>
  <tr>
    <th width="50%">Settings-&gt;models</th>
    <th width="50%">Settings-&gt;behavior</th>
  </tr>
  <tr>
    <td><img src="docs/screenshots/settings_models.png" alt="Settings models" width="100%"></td>
    <td><img src="docs/screenshots/settings-behavior.png" alt="Settings behavior" width="100%"></td>
  </tr>
  <tr>
    <th width="50%">Settings-&gt;account</th>
    <th width="50%">Settings-&gt;system</th>
  </tr>
  <tr>
    <td><img src="docs/screenshots/settings-account.png" alt="Settings account" width="100%"></td>
    <td><img src="docs/screenshots/settings-system.png" alt="Settings system" width="100%"></td>
  </tr>
  <tr>
    <th width="50%">Engine config</th>
    <th width="50%">Supported architectures</th>
  </tr>
    <tr>
    <td><img src="docs/screenshots/settings-external-matricxon.png" alt="Engine config" width="100%"></td>
    <td><img src="docs/screenshots/stats-architectures.png" alt="Supported architectures" width="100%"></td>
  </tr>
</table>


<br><br>

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-async-009688)
![Self-hosted](https://img.shields.io/badge/self--hosted-yes-pink)
![License](https://img.shields.io/badge/license-AGPL--3.0-blue)

<br><br>

## License

[GNU AGPL-3.0](LICENSE) — you're free to run it, modify it, and
self-host it, no strings attached, for personal or internal use. The
one thing it asks in return: if you take a modified version and offer
it to other people as a hosted service, you share those changes back
too. If that's relevant to how you're planning to use this (it usually
isn't, for a personal setup or an internal company deployment), it's
worth a quick look either way.

If you try it, a ⭐ is always appreciated — and issues/PRs are genuinely
welcome, whoever you're running this for.

<!-- ================================================================ -->
