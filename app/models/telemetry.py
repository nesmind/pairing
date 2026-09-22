"""Ollama/Matricxon telemetry — see app/services/ollama_telemetry.py and app/services/matricxon_telemetry.py for
how these rows actually get written (an OpenTelemetry span exporter, not a plain service-layer insert) and
app/services/telemetry_service.py for the aggregation queries the Telemetry page's dashboard runs against them,
one engine at a time (see that module's own docstring for why `engine` is filtered on, not grouped by).

Both tables are pure append-only event logs — nothing ever looks a row
up by its own id from outside this app (unlike e.g. ImageGenerationJob,
whose id is embedded in a URL) — so they use plain autoincrementing
integer keys instead of this codebase's usual new_id() convention,
which exists specifically for ids that get exposed externally.

Class/table names predate Matricxon support and still read Ollama-specific — kept as-is rather than renamed
(see migration 20260920_.*_add_telemetry_engine_column's own message for the full reasoning) since a rename
would touch every existing caller for no functional gain; `engine` is what actually makes each row's source
unambiguous now, not the name."""

from sqlalchemy import BigInteger, Boolean, Column, DateTime, Float, Integer, String, Text

from app.database import Base


class TelemetryEvent(Base):
    """One completed (or failed) call to Ollama or Matricxon — one row per app.services.ollama_client or
    app.services.matricxon_client's chat_stream/embed invocation, written by that engine's own DBSpanExporter
    (see telemetry_exporter.py, shared by both) once the call's OpenTelemetry span ends. duration_ms is our own
    wall-clock measurement (span start to end); the total_duration_ms/load_duration_ms/*_eval_* fields are the
    engine's own server-side timing, only ever populated for "chat" events — neither engine's embedding response
    carries timing/token fields at all."""

    __tablename__ = "telemetry_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # "ollama" | "matricxon" — which engine (see app.services.engine_service) actually served this call,
    # regardless of which one is active *now*: a row is stamped at call time, from the OpenTelemetry
    # instrumentation scope that produced its span (see telemetry_exporter._persist_spans), so switching the
    # active engine later never rewrites history.
    engine = Column(String(20), nullable=False, default="ollama")
    kind = Column(String(20), nullable=False)  # "chat" | "embed"
    model = Column(String(255), nullable=True)
    host = Column(String(255), nullable=False)
    success = Column(Boolean, nullable=False)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=False)
    duration_ms = Column(Float, nullable=False)
    total_duration_ms = Column(Float, nullable=True)
    load_duration_ms = Column(Float, nullable=True)
    prompt_eval_duration_ms = Column(Float, nullable=True)
    eval_duration_ms = Column(Float, nullable=True)
    prompt_eval_count = Column(Integer, nullable=True)
    eval_count = Column(Integer, nullable=True)
    # How many hosts chat_stream's failover tried (or embed's cold-start
    # retry loop attempted) before this call resolved — 1 in the common
    # case; see app.services.ollama_client's own docstrings.
    attempt_count = Column(Integer, nullable=False, default=1)


class OllamaModelSnapshot(Base):
    """One row per (host, currently-loaded model) on every poll of an engine's GET /api/ps — see
    app.services.ollama_ps_poller and app.services.matricxon_ps_poller, both of which write here. Appended
    unconditionally rather than upserted: at a 30s poll interval this stays small for a typical single/few-host
    deployment, and doubling as a cheap time series (was model X loaded on host Y at time T) is the whole point —
    app.services.ollama_snapshot_service reads back only the newest row per (engine, host, model_name) for the
    "currently loaded" dashboard table, one engine at a time."""

    __tablename__ = "ollama_model_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # "ollama" | "matricxon" — see TelemetryEvent.engine's own comment for the identical reasoning.
    engine = Column(String(20), nullable=False, default="ollama")
    host = Column(String(255), nullable=False)
    model_name = Column(String(255), nullable=False)
    size_bytes = Column(BigInteger, nullable=True)
    size_vram_bytes = Column(BigInteger, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    polled_at = Column(DateTime, nullable=False)
