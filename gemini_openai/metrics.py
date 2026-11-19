"""Per-request metrics for chat completions, to see where the time goes.

One JSON line per request in GEMINI_REQUEST_LOG (default ~/.gemini-web-api/requests.jsonl,
0600; "off" disables it) and a one-line summary in the server log. Never message
content: sizes, timings, model and tool NAMES, and what happened to the request
(thread continued or new, refusal detected, retries, the library's stall recovery,
cookies applied while it ran).

The record of the request in progress lives in a context variable, so the backends
and the library's log sink add to it without it being passed around.
"""
from __future__ import annotations

import contextvars
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

LOG_PATH = os.getenv("GEMINI_REQUEST_LOG", "~/.gemini-web-api/requests.jsonl")

# Replies Gemini gives when it did not do the work: generic refusals, and the budget
# message of its built-in tools (it tried one instead of asking for the client's).
REFUSALS = {
    "tool_budget": (
        "exceeded the maximum number of tool",
        "0 remaining",
        "presupuesto de llamadas",
        "no tengo presupuesto",
        "no tengo acceso a herramientas",
        "no tengo herramientas",
    ),
    "generic_refusal": (
        "soy un modelo de lenguaje", "no me han programado", "no puedo ayudarte",
        "i'm a language model", "i am a language model", "i'm not able to help with that",
        "i'm just a language model",
    ),
}


def classify_refusal(text: str | None) -> str | None:
    low = (text or "").lower()
    for kind, needles in REFUSALS.items():
        if any(n in low for n in needles):
            return kind
    return None


@dataclass
class RequestMetrics:
    id: str
    at: float
    mode: str
    stream: bool
    model_requested: str | None
    model: str | None = None
    messages: int = 0
    tools: int = 0
    tool_choice: str | None = None
    in_flight: int = 0
    session: str | None = None
    thread: str | None = None          # new | continued | continue_failed
    thread_error: str | None = None
    prompt_chars: int = 0              # the full prompt (a new thread's)
    sent_chars: int = 0                # what was actually sent (continuation or full)
    tools_prompt: str | None = None    # full | reminder: how a continued turn sent the tools
    lock_wait_s: float | None = None
    client_s: float | None = None
    first_chunk_s: float | None = None
    gemini_s: float | None = None
    parse_s: float | None = None
    total_s: float | None = None
    finish_reason: str | None = None
    tool_calls: list[str] = field(default_factory=list)
    tool_parse: str | None = None      # parsed | no_tool_call | not_requested
    response_chars: int = 0
    refusal: str | None = None
    retries: int = 0
    library_events: list[str] = field(default_factory=list)
    cookie_updates: int = 0
    # The client's side of a tool turn: time from the reply that asked for tools to
    # this request carrying their results (the tools' run time plus the client's own).
    client_gap_s: float | None = None
    gap_tools: list[str] = field(default_factory=list)
    error: str | None = None
    _t0: float = field(default=0.0, repr=False)
    _session_key: str | None = field(default=None, repr=False)

    def since_start(self) -> float:
        return round(time.monotonic() - self._t0, 2)


CURRENT: contextvars.ContextVar[RequestMetrics | None] = contextvars.ContextVar("REQUEST_METRICS", default=None)
_active: set[str] = set()
_by_id: dict[str, RequestMetrics] = {}
# Per session: when the last reply that asked for tools went out, and which tools.
_awaiting_tools: dict[str, tuple[float, list[str]]] = {}
_AWAITING_MAX = 1000


def start(mode: str, stream: bool, model_requested: str | None) -> RequestMetrics:
    m = RequestMetrics(id=uuid.uuid4().hex[:10], at=round(time.time(), 3), mode=mode,
                       stream=stream, model_requested=model_requested, in_flight=len(_active))
    m._t0 = time.monotonic()
    _active.add(m.id)
    _by_id[m.id] = m
    CURRENT.set(m)
    return m


def current() -> RequestMetrics | None:
    return CURRENT.get()


def note(**fields) -> None:
    """Set fields on the request in progress (no-op outside one)."""
    m = CURRENT.get()
    if m is not None:
        for k, v in fields.items():
            setattr(m, k, v)


def add_event(text: str) -> None:
    m = CURRENT.get()
    if m is not None and len(m.library_events) < 20:
        m.library_events.append(f"{m.since_start()}s {text[:120]}")


def session_turn(m: RequestMetrics, session_key: str, last_role: str | None) -> None:
    """Tie the request to its session; if it carries tool results for a reply that asked
    for tools, record how long the client took to run them and come back."""
    m.session = session_key[:8]
    m._session_key = session_key
    pending = _awaiting_tools.pop(session_key, None)
    if pending and last_role == "tool":
        sent_at, names = pending
        m.client_gap_s = round(time.time() - sent_at, 2)
        m.gap_tools = names


def cookies_applied() -> None:
    """Pushed cookies were applied to the live client: count it on every running request."""
    for rid in list(_active):
        m = _by_id.get(rid)
        if m is not None:
            m.cookie_updates += 1


def finish(m: RequestMetrics) -> None:
    if m.id not in _active:
        return
    _active.discard(m.id)
    _by_id.pop(m.id, None)
    m.total_s = m.since_start()
    if m._session_key and m.tool_calls:
        _awaiting_tools[m._session_key] = (time.time(), list(m.tool_calls))
        while len(_awaiting_tools) > _AWAITING_MAX:
            _awaiting_tools.pop(next(iter(_awaiting_tools)))
    record = {k: v for k, v in asdict(m).items() if not k.startswith("_")}
    print("[req] " + " ".join(f"{k}={record[k]}" for k in (
        "id", "session", "model", "thread", "sent_chars", "gemini_s", "total_s", "finish_reason",
        "tool_calls", "client_gap_s", "gap_tools", "refusal", "cookie_updates", "error")
        if record[k] not in (None, [], 0)
        or k in ("id", "total_s")), flush=True)
    _write(record)


def _write(record: dict) -> None:
    if LOG_PATH.lower() in ("", "off", "0", "false", "no"):
        return
    path = Path(LOG_PATH).expanduser()
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"[req] could not write {path}: {e}", flush=True)


_sink_installed = False


def install_library_sink() -> None:
    """Record gemini_webapi's stall watchdog and recovery messages on the request they hit."""
    global _sink_installed
    if _sink_installed:
        return
    try:
        from loguru import logger
    except ImportError:
        return

    def sink(message) -> None:
        text = message.record["message"]
        if any(k in text for k in ("Watchdog", "Recovery", "Retrying", "temporary error")):
            add_event(text)

    logger.add(sink, level="DEBUG", format="{message}")
    _sink_installed = True
