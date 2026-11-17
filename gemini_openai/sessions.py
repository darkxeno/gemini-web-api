"""Chat sessions: which conversation a request continues, for both chat backends.

OpenAI clients (OpenCode) resend the whole conversation on every call and carry
no conversation id. server.py derives one per request (`ChatSession`, set in
`CURRENT_CHAT`) and each backend uses it to continue the same Gemini thread
instead of starting a new one with the full history:

  * the chrome backend pins the session to a tab (chrome_backend.Hub);
  * the `api` backend keeps the thread's gemini_webapi metadata
    ([cid, rid, rcid, ...]) in `ApiSessionStore`, persisted across restarts.

A thread is continued only when the request provably extends the exchange it
last answered; anything else (an edited or compacted history, a lost thread)
starts a new thread with the full prompt.
"""
from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .openai_schemas import ChatMessage, render_turns


@dataclass(frozen=True)
class ChatSession:
    """The conversation a chat request belongs to."""

    key: str
    # What to send to a thread that already holds the conversation: the messages
    # since the model's last reply. None on a first turn.
    continuation: str | None = None
    # The request's messages, to check a stored thread is still this history.
    messages: tuple[ChatMessage, ...] = ()
    # The same continuation with the tools only REMINDED (tools.build_tools_reminder),
    # for a thread already given the full explanation of the same tool set
    # (tools_digest). None without tools.
    continuation_brief: str | None = None
    tools_digest: str | None = None


CURRENT_CHAT: contextvars.ContextVar[ChatSession | None] = contextvars.ContextVar(
    "CURRENT_CHAT", default=None
)


def history_digest(messages: tuple[ChatMessage, ...] | list[ChatMessage]) -> str:
    """A fingerprint of a message history (roles, texts, tool calls and results)."""
    turns, _ = render_turns(list(messages))
    h = hashlib.sha256()
    for role, text in turns:
        h.update(role.encode())
        h.update(b"\0")
        h.update(text.encode())
        h.update(b"\0")
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# api backend: thread state per session
# --------------------------------------------------------------------------- #
SESSION_STORE_FILE = Path(
    os.environ.get("GEMINI_API_SESSIONS", "~/.gemini-web-api/sessions.json")
).expanduser()
SESSION_STORE_MAX = 500


def _write_private(path: Path, data: Any) -> None:
    """Write JSON readable by this user only, atomically."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


class ApiSessionStore:
    """gemini_webapi thread metadata per chat session, persisted to disk.

    An entry is {"metadata": [...], "n": messages answered, "digest": their
    history_digest, "account": the cookie session it belongs to, "updated": time}.
    A per-session lock keeps two calls of one session from racing on its thread;
    different sessions run in parallel.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or SESSION_STORE_FILE
        self._data: dict[str, dict] | None = None
        self._locks: dict[str, asyncio.Lock] = {}

    def _load(self) -> dict[str, dict]:
        if self._data is None:
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = {}
            self._data = data if isinstance(data, dict) else {}
        return self._data

    def lock(self, key: str) -> asyncio.Lock:
        return self._locks.setdefault(key, asyncio.Lock())

    def thread_for(self, chat: ChatSession, account: str) -> list | None:
        """The stored thread metadata, if `chat` continues exactly what it answered.

        The request must repeat the stored history unchanged, then carry exactly
        one assistant message (the reply to it), then the new turns; and the
        thread must belong to the current cookie session (another account can't
        see it). The reply itself isn't compared: clients store a tool-call reply
        in another form than the text it came as, so only an edit of the latest
        reply alone would go unnoticed.
        """
        entry = self._load().get(chat.key)
        if not entry or not chat.continuation or entry.get("account") != account:
            return None
        n = entry.get("n")
        msgs = chat.messages
        if not isinstance(n, int) or len(msgs) <= n + 1:
            return None
        if history_digest(msgs[:n]) != entry.get("digest"):
            return None
        if msgs[n].role != "assistant" or any(m.role == "assistant" for m in msgs[n + 1:]):
            return None
        metadata = entry.get("metadata")
        return list(metadata) if isinstance(metadata, list) else None

    def tools_explained(self, chat: ChatSession) -> bool:
        """Whether the session's thread was already given the full explanation of the
        request's tool set, so a reminder is enough."""
        entry = self._load().get(chat.key) or {}
        return bool(chat.tools_digest) and entry.get("tools_digest") == chat.tools_digest

    def save(self, chat: ChatSession, metadata: list, account: str) -> None:
        data = self._load()
        previous = data.pop(chat.key, None) or {}
        data[chat.key] = {  # newest last
            "metadata": list(metadata),
            "n": len(chat.messages),
            "digest": history_digest(chat.messages),
            "account": account,
            # Every turn sent with tools explained (full) or reminded (same set) them.
            "tools_digest": chat.tools_digest or previous.get("tools_digest"),
            "updated": time.time(),
        }
        while len(data) > SESSION_STORE_MAX:
            data.pop(next(iter(data)))
        self._persist()

    def drop(self, key: str) -> None:
        if self._load().pop(key, None) is not None:
            self._persist()

    def _persist(self) -> None:
        try:
            _write_private(self.path, self._load())
        except OSError as e:
            print(f"[api] could not save sessions to {self.path}: {e}", flush=True)
