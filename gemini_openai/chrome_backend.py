"""Chrome-extension backend: drive logged-in gemini.google.com tabs.

An alternative to the cookie/CDP gemini_webapi backend (gemini_pool.py). Instead
of harvesting `__Secure-1PSID` and minting OSID cookies over CDP, a Chrome
extension content-script runs *inside* the user's own logged-in tab, types the
prompt, clicks send, and scrapes the reply. Auth is whatever the browser already
has -- nothing to copy, nothing to expire.

This module exposes a `manager` with the SAME interface as
`gemini_pool.GeminiManager` (`generate` / `generate_stream` returning objects
with `.text` / `.text_delta` / `.images` / `.videos`), so `server.py`'s chat path
is backend-agnostic. Select it with `GEMINI_BACKEND=chrome`.

Every open Gemini tab connects to the server's `/ws` endpoint and becomes a
worker in a POOL. Jobs are dispatched to any free tab, so N tabs = N parallel
jobs; within one tab requests stay serialized (the content script is
single-flight). A tab identifies itself in its hello message with a `tabId` and
the Google multi-login account (`authuser`) its URL is on, which lets callers
prefer a tab on a specific account. Tabs running an older content script that
sends a bare hello still work -- they get an auto-generated id and no authuser.

A chat that carries a session (`CURRENT_CHAT`, set by server.py per request) is
PINNED to one tab: its first turn starts a new Gemini conversation there with
the full prompt, and later turns send only what is new since the model's last
reply into that same conversation. A session whose tab is gone, has moved to
another conversation, or failed starts over in a new conversation with the full
prompt -- never continuing a conversation it doesn't own. Pins survive a server
restart through the session map file (session -> conversation URL).

WS protocol (server <-> extension):
    server -> ext : {"type":"chat"|"image"|"video","id","prompt","fresh",...}
    ext -> server : {"type":"hello","tabId","authuser","href","worker"}
                    {"type":"url_changed","tabId","href","worker"}  (SPA navigation)
                    {"type":"ka"}  (keepalive, ignored)
                    {"type":"delta","id","text"} | {"type":"result","id","text","media"?}
                    {"type":"error","id","message"}
"""
from __future__ import annotations

import asyncio
import contextvars
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# How long to wait for the tab to finish a reply. Gemini "Pro" mode plus long
# answers can run well past a minute.
REQUEST_TIMEOUT = 300.0

# How long a job waits for a free tab when all connected tabs are busy.
ACQUIRE_TIMEOUT = 600.0

# Media failover knobs. Real image generations land in 14-102s, so the per-attempt
# cap detects a quota stall without burning the caller's whole budget on one dead
# account. An account that quota-fails is skipped for the cooldown window.
MEDIA_ATTEMPT_TIMEOUT = float(os.environ.get("GEMINI_MEDIA_ATTEMPT_TIMEOUT", "180"))
MEDIA_QUOTA_COOLDOWN = float(os.environ.get("GEMINI_MEDIA_QUOTA_COOLDOWN", "1800"))
# Accounts media may never touch (comma-separated authuser indices). Empty by
# default — note that u/N indices RESHUFFLE when the Google multi-login order
# changes, so an index-based ban must be re-verified after any sign-in/out.
MEDIA_EXCLUDE_AUTHUSERS = {
    a.strip() for a in os.environ.get("GEMINI_MEDIA_EXCLUDE_AUTHUSERS", "").split(",")
    if a.strip()
}

# Session -> conversation URL, so pins survive a server restart. Capped: the
# oldest sessions drop out first.
SESSION_MAP_FILE = Path(os.environ.get("GEMINI_SESSION_MAP", "/tmp/gemini-session-map.json"))
SESSION_MAP_MAX = 500

# A tab inside a conversation, not on a bare /app.
IN_CONVERSATION = re.compile(r"/app/[0-9a-f]{6,}")


@dataclass(frozen=True)
class ChatSession:
    """The conversation a chat request belongs to (see server.py)."""

    key: str
    # What to send to a tab that already holds the conversation: the messages
    # since the model's last reply. None on a first turn.
    continuation: str | None = None


CURRENT_CHAT: contextvars.ContextVar[ChatSession | None] = contextvars.ContextVar(
    "CURRENT_CHAT", default=None
)


def _load_session_map() -> dict[str, str]:
    try:
        data = json.loads(SESSION_MAP_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def _save_session_map(data: dict[str, str]) -> None:
    try:
        SESSION_MAP_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as e:
        print(f"[hub] could not save the session map {SESSION_MAP_FILE}: {e}", flush=True)


@dataclass
class ChromeOutput:
    """Minimal stand-in for gemini_webapi's ModelOutput.

    server.py only reads `.text`, `.text_delta`, `.images`, `.videos`. The chrome
    backend produces text only (media generation via the UI is not wired yet), so
    images/videos are always empty.
    """

    text: str = ""
    text_delta: str = ""
    images: list = field(default_factory=list)
    videos: list = field(default_factory=list)


@dataclass
class TabConn:
    """One connected Gemini tab (one pool worker)."""

    ws: WebSocket
    key: int
    tab_id: str
    authuser: str | None = None
    href: str = ""
    connected_at: float = field(default_factory=time.time)
    busy: bool = False
    # A tab that connected while INSIDE an existing conversation belongs to the
    # human (e.g. a restored work chat) -- never dispatch jobs into it. Tabs on a
    # bare /app become workers; conversations they then create are the bridge's.
    eligible: bool = True
    # The tab has served a bridge job (sessionStorage), so its conversations are ours.
    worker: bool = False

    async def send(self, msg: dict) -> None:
        await self.ws.send_text(json.dumps(msg))


class Hub:
    """Pool of connected extension tabs plus in-flight requests.

    Each pending non-stream request parks a future keyed by id; streaming
    requests park a queue. `job_conn` remembers which tab a job went to so a
    disconnect only fails ITS jobs, not the whole pool's.
    """

    def __init__(self) -> None:
        self.conns: dict[int, TabConn] = {}
        self.pending: dict[str, asyncio.Future] = {}
        self.streams: dict[str, asyncio.Queue] = {}
        self.job_conn: dict[str, int] = {}
        self._counter = 0
        self._key = 0
        # Set whenever a tab frees up / joins / leaves; acquire() waits on it.
        self._changed = asyncio.Event()
        # Chat sessions: the tab each is pinned to, when each last ran (for LRU
        # eviction), and its conversation URL (loaded from SESSION_MAP_FILE).
        self.session_tabs: dict[str, int] = {}
        self.session_used: dict[str, float] = {}
        self._session_urls: dict[str, str] | None = None

    def online(self) -> bool:
        return bool(self.conns)

    def tabs(self) -> list[dict]:
        return [
            {
                "tabId": c.tab_id,
                "authuser": c.authuser,
                "busy": c.busy,
                "eligible": c.eligible,
                "href": c.href,
                "connected_s": round(time.time() - c.connected_at),
            }
            for c in self.conns.values()
        ]

    def next_id(self) -> str:
        self._counter += 1
        return f"req_{int(time.time())}_{self._counter}"

    def next_key(self) -> int:
        self._key += 1
        return self._key

    # ---- tab choice ---------------------------------------------------------
    def _pool(self, authuser: str | None, exclude_key: int | None) -> list[TabConn]:
        """Eligible tabs (on `authuser`, strictly, if given). Raises when there are none."""
        pool = [c for c in self.conns.values() if c.eligible and c.key != exclude_key]
        if not pool and exclude_key is not None:
            pool = [c for c in self.conns.values() if c.eligible]  # only choice
        if not pool:
            raise RuntimeError(
                "no eligible gemini tab connected -- open a FRESH "
                "gemini.google.com/app tab (tabs already inside a "
                "conversation are left alone)"
            )
        if authuser is not None:
            # Strict: quota is per-account, so falling back to another
            # account's tab would silently burn the wrong quota.
            pool = [c for c in pool if c.authuser == str(authuser)]
            if not pool:
                raise RuntimeError(
                    f"no eligible gemini tab on account u/{authuser} -- open "
                    f"gemini.google.com/u/{authuser}/app in the browser"
                )
        return pool

    @staticmethod
    def _best(tabs: list[TabConn]) -> TabConn:
        # Tabs on a bare /app first (no conversation DOM to drag along), then the
        # newest connection: freshly (re)loaded tabs are the least likely to be
        # stale, throttled, or stuck in a bad UI state.
        bare = [c for c in tabs if not IN_CONVERSATION.search(c.href)]
        return max(bare or tabs, key=lambda c: c.connected_at)

    async def _wait(self, deadline: float) -> None:
        remaining = deadline - time.time()
        if remaining <= 0:
            raise RuntimeError(
                f"all {len(self.conns)} gemini tab(s) busy for "
                f"{ACQUIRE_TIMEOUT:.0f}s -- open more tabs for more parallelism"
            )
        self._changed.clear()
        try:
            await asyncio.wait_for(self._changed.wait(), timeout=min(remaining, 5.0))
        except asyncio.TimeoutError:
            pass  # re-check the pool; the deadline check raises eventually

    async def acquire(self, authuser: str | None = None,
                      exclude_key: int | None = None) -> TabConn:
        """Reserve a free ELIGIBLE tab (on `authuser` if given, strictly).

        For jobs outside a chat session (media, chats without one). Tabs pinned to
        a session are taken only when no other tab is free, and lose their pin:
        the job starts a new conversation there.

        Single event loop => no await between picking a tab and marking it busy,
        so two acquirers can never grab the same tab. `exclude_key` skips the tab
        a failed attempt just ran on (retry-elsewhere).
        """
        deadline = time.time() + ACQUIRE_TIMEOUT
        while True:
            self._drop_stale_pins()
            free = [c for c in self._pool(authuser, exclude_key) if not c.busy]
            if free:
                pinned = set(self.session_tabs.values())
                conn = self._best([c for c in free if c.key not in pinned] or free)
                self._unpin_tab(conn.key)
                conn.busy = True
                return conn
            await self._wait(deadline)

    # ---- chat sessions -------------------------------------------------------
    def _urls(self) -> dict[str, str]:
        if self._session_urls is None:
            self._session_urls = _load_session_map()
        return self._session_urls

    def _drop_stale_pins(self) -> None:
        """Unpin sessions whose tab disconnected or is no longer eligible."""
        for sk, key in list(self.session_tabs.items()):
            conn = self.conns.get(key)
            if conn is None or not conn.eligible:
                del self.session_tabs[sk]

    def _unpin_tab(self, key: int) -> None:
        for sk, k in list(self.session_tabs.items()):
            if k == key:
                del self.session_tabs[sk]

    def _session_tab(self, session: str, authuser: str | None,
                     exclude_key: int | None) -> TabConn | None:
        """The tab `session` is pinned to, re-pinning by conversation URL after a
        reconnect or server restart. None if it has no usable tab."""
        key = self.session_tabs.get(session)
        if key is None:
            url = self._urls().get(session)
            pinned = set(self.session_tabs.values())
            match = next((c for c in self.conns.values()
                          if url and c.href == url and c.eligible
                          and c.key != exclude_key and c.key not in pinned), None)
            if match is None:
                return None
            self.session_tabs[session] = key = match.key
        conn = self.conns[key]
        if key == exclude_key or (authuser is not None and conn.authuser != str(authuser)):
            del self.session_tabs[session]
            return None
        return conn

    def _new_session_tab(self, authuser: str | None,
                         exclude_key: int | None) -> TabConn | None:
        """A free tab for a session without one: an unpinned tab if any, else the
        tab of the least recently used session, which loses its pin."""
        free = [c for c in self._pool(authuser, exclude_key) if not c.busy]
        if not free:
            return None
        pinned = set(self.session_tabs.values())
        unpinned = [c for c in free if c.key not in pinned]
        if unpinned:
            return self._best(unpinned)

        def last_used(c: TabConn) -> float:
            return max((self.session_used.get(sk, 0.0)
                        for sk, k in self.session_tabs.items() if k == c.key), default=0.0)

        conn = min(free, key=last_used)
        self._unpin_tab(conn.key)
        return conn

    async def acquire_session(self, session: str, authuser: str | None = None,
                              exclude_key: int | None = None) -> tuple[TabConn, bool]:
        """Reserve the tab of chat `session`. Returns (tab, continuing).

        continuing: the tab still shows the session's conversation, so the turn
        can be sent into it. False for a new pin (a first turn, or a session whose
        tab was lost or failed): the caller starts a new conversation with the
        full prompt. A pinned tab that is busy (the session's previous turn still
        running) is waited for.
        """
        deadline = time.time() + ACQUIRE_TIMEOUT
        if exclude_key is not None and self.session_tabs.get(session) == exclude_key:
            del self.session_tabs[session]  # retrying elsewhere after a failure
        while True:
            self._drop_stale_pins()
            conn = self._session_tab(session, authuser, exclude_key)
            if conn is not None:
                if not conn.busy:
                    conn.busy = True
                    self.session_used[session] = time.time()
                    url = self._urls().get(session)
                    # No URL yet (the first turn's url_changed hasn't arrived): the
                    # pin alone says the tab holds it.
                    return conn, url is None or conn.href == url
            else:
                conn = self._new_session_tab(authuser, exclude_key)
                if conn is not None:
                    self.session_tabs[session] = conn.key
                    self.session_used[session] = time.time()
                    if self._urls().pop(session, None) is not None:
                        _save_session_map(self._urls())  # its new conversation isn't known yet
                    conn.busy = True
                    return conn, False
            await self._wait(deadline)

    def record_url(self, conn: TabConn) -> None:
        """Remember the conversation URL of the sessions pinned to `conn`."""
        if not IN_CONVERSATION.search(conn.href):
            return
        urls = self._urls()
        changed = False
        for sk, key in self.session_tabs.items():
            if key == conn.key and urls.get(sk) != conn.href:
                urls.pop(sk, None)
                urls[sk] = conn.href  # newest last
                changed = True
        if changed:
            while len(urls) > SESSION_MAP_MAX:
                urls.pop(next(iter(urls)))
            _save_session_map(urls)

    def release(self, conn: TabConn) -> None:
        conn.busy = False
        self._changed.set()

    def notify_pool_changed(self) -> None:
        self._changed.set()


hub = Hub()


class ChromeManager:
    """Drop-in replacement for GeminiManager over the extension bridge."""

    def __init__(self) -> None:
        # Gemini web runs at most ONE media generation per Google account: of two
        # concurrent requests, the loser never renders (silent starvation until
        # our timeout). Serialize media per account; chat is unaffected.
        self._media_locks: dict[str, asyncio.Lock] = {}
        # authuser -> monotonic time of last quota-shaped media failure. Accounts
        # in cooldown are skipped by the failover walk (quota exhaustion stalls
        # silently, so every attempt on a dead account costs a full timeout).
        self.quota_bad: dict[str, float] = {}

    def _media_lock(self, authuser: str | None) -> asyncio.Lock:
        key = str(authuser) if authuser is not None else "_any"
        return self._media_locks.setdefault(key, asyncio.Lock())

    # Quota exhaustion either fast-fails (extension spotted the limit text) or
    # stalls silently until our render timeout -- both mark the ACCOUNT bad.
    QUOTA_SHAPED = ("quota exhausted", "did not render in time")

    def _quota_shaped(self, exc: Exception) -> bool:
        msg = str(exc)
        return any(sig in msg for sig in self.QUOTA_SHAPED)

    def _media_accounts(self, requested: str | None) -> list[str]:
        """Failover order for media: requested account first, then every other
        account with an eligible connected tab, healthy before in-cooldown."""
        now = time.monotonic()
        # authuser None (old content script, no /u/N/ in URL) is a real worker;
        # keep it as an "any tab" candidate rather than dropping it.
        pool = {t["authuser"] for t in hub.tabs()
                if t["eligible"] and t["authuser"] not in MEDIA_EXCLUDE_AUTHUSERS}
        in_cooldown = {a for a in pool
                       if now - self.quota_bad.get(a, -1e9) < MEDIA_QUOTA_COOLDOWN}
        healthy = sorted(pool - in_cooldown, key=lambda a: (a is None, a or ""))
        cooled = sorted(in_cooldown, key=lambda a: self.quota_bad[a])  # oldest failure first
        order = healthy + cooled
        if requested is not None and requested in order:
            order.remove(requested)
            order.insert(0, requested)
        return order

    # Failures that indicate a BROKEN TAB (not a bad request) -> retry once
    # on a different tab before giving up.
    TAB_SHAPED = ("no response appeared", "composer not found", "send button not found",
                  "sent no reply", "sent no image", "sent no video", "disconnected")

    def _tab_shaped(self, exc: Exception) -> bool:
        msg = str(exc)
        return any(sig in msg for sig in self.TAB_SHAPED)

    async def generate(self, prompt, files=None, model=None, temporary=True,
                       authuser: str | None = None, _exclude: int | None = None):
        if files:
            raise RuntimeError("the chrome backend does not support file/vision input yet")
        conn, send_prompt, fresh = await _acquire_chat(prompt, authuser, _exclude)
        rid = hub.next_id()
        print(f"[hub] chat {rid} -> {conn.tab_id} (u/{conn.authuser}, "
              f"{'new conversation' if fresh else 'continuing'})", flush=True)
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        hub.pending[rid] = fut
        hub.job_conn[rid] = conn.key
        try:
            await conn.send(
                {"type": "chat", "id": rid, "prompt": send_prompt,
                 "model": _model_name(model), "fresh": fresh}
            )
            try:
                msg = await asyncio.wait_for(fut, timeout=REQUEST_TIMEOUT)
            except asyncio.TimeoutError:
                # str() of a bare TimeoutError is "" -- name the failure.
                raise RuntimeError(
                    f"tab {conn.tab_id} sent no reply within {REQUEST_TIMEOUT:.0f}s"
                ) from None
        except Exception as e:  # noqa: BLE001
            if _exclude is None and self._tab_shaped(e):
                print(f"[hub] chat {rid} failed on {conn.tab_id} ({e}) -- retrying "
                      f"on another tab", flush=True)
                hub.pending.pop(rid, None)
                hub.job_conn.pop(rid, None)
                hub.release(conn)
                return await self.generate(prompt, files=files, model=model,
                                           temporary=temporary, authuser=authuser,
                                           _exclude=conn.key)
            raise
        finally:
            hub.pending.pop(rid, None)
            hub.job_conn.pop(rid, None)
            hub.release(conn)
        text = msg.get("text", "") if isinstance(msg, dict) else msg
        return ChromeOutput(text=text)

    async def generate_media(
        self, prompt: str, kind: str, timeout: float | None = None,
        aspect: str | None = None, authuser: str | None = None,
        _exclude: int | None = None
    ):
        """Generate an image or video in a free tab and return its bytes.

        kind: "image" | "video". aspect ("16:9" | "9:16") only applies to video,
        where the extension clicks Gemini's real aspect-ratio button. Returns
        (media, text, authuser) -- the account that actually served the job.
        Video can take minutes, so callers pass a longer timeout.

        Accounts AUTO-FAIL-OVER: the requested account is tried first, then every
        other account with a connected tab. A quota-shaped failure puts the
        account in cooldown so later requests skip straight to healthy ones.
        Attempts after the first (and any attempt on an in-cooldown account) use
        the shorter MEDIA_ATTEMPT_TIMEOUT -- a real generation lands well inside
        it, and a full timeout per dead account would stack into half an hour.
        """
        accounts = self._media_accounts(authuser)
        if not accounts:
            raise RuntimeError("no eligible gemini tab on any account")
        errors: list[str] = []
        for i, acct in enumerate(accounts):
            att_timeout = timeout
            if i > 0 or acct in self.quota_bad:
                att_timeout = min(timeout or REQUEST_TIMEOUT, MEDIA_ATTEMPT_TIMEOUT)
            try:
                async with self._media_lock(acct):
                    media, text = await self._generate_media_locked(
                        prompt, kind, timeout=att_timeout, aspect=aspect,
                        authuser=acct, _exclude=_exclude,
                    )
                self.quota_bad.pop(acct, None)
                return media, text, acct
            except Exception as e:  # noqa: BLE001
                errors.append(f"u/{acct}: {e}")
                if self._quota_shaped(e):
                    self.quota_bad[acct] = time.monotonic()
                    print(f"[hub] {kind} quota-shaped failure on u/{acct} -> cooldown; "
                          f"{len(accounts) - i - 1} account(s) left", flush=True)
                    continue
                if len(accounts) > i + 1:
                    print(f"[hub] {kind} failed on u/{acct} ({e}) -> next account",
                          flush=True)
                    continue
                raise
        raise RuntimeError("media failed on all accounts: " + " | ".join(errors))

    async def fetch_bytes(self, url: str, authuser: str | None = None,
                          timeout: float = REQUEST_TIMEOUT) -> bytes:
        """Download `url` with a connected tab's browser session.

        Used for media the cookie backend generated: the usercontent host 403s
        any server-side GET, but the logged-in browser passes. Doesn't reserve
        the tab -- a fetch touches no DOM, so it can share a busy one.
        """
        import base64

        conns = [c for c in hub.conns.values()
                 if authuser is None or c.authuser == str(authuser)]
        if not conns:
            raise RuntimeError(f"no gemini tab connected on u/{authuser}")
        conn = max(conns, key=lambda c: c.connected_at)
        rid = hub.next_id()
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        hub.pending[rid] = fut
        hub.job_conn[rid] = conn.key
        try:
            await conn.send({"type": "fetch", "id": rid, "url": url})
            msg = await asyncio.wait_for(fut, timeout=timeout)
        finally:
            hub.pending.pop(rid, None)
            hub.job_conn.pop(rid, None)
        media = (msg.get("media") or [{}])[0]
        return base64.b64decode(media.get("b64") or "")

    async def _generate_media_locked(
        self, prompt: str, kind: str, timeout: float | None = None,
        aspect: str | None = None, authuser: str | None = None,
        _exclude: int | None = None
    ):
        conn = await hub.acquire(authuser, exclude_key=_exclude)
        rid = hub.next_id()
        print(f"[hub] {kind} {rid} -> {conn.tab_id} (u/{conn.authuser})", flush=True)
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        hub.pending[rid] = fut
        hub.job_conn[rid] = conn.key
        try:
            await conn.send(
                {"type": kind, "id": rid, "prompt": prompt, "fresh": True, "aspect": aspect}
            )
            t = timeout or REQUEST_TIMEOUT
            try:
                msg = await asyncio.wait_for(fut, timeout=t)
            except asyncio.TimeoutError:
                raise RuntimeError(
                    f"tab {conn.tab_id} sent no {kind} reply within {t:.0f}s"
                ) from None
        except Exception as e:  # noqa: BLE001
            if _exclude is None and self._tab_shaped(e):
                print(f"[hub] {kind} {rid} failed on {conn.tab_id} ({e}) -- retrying "
                      f"on another tab", flush=True)
                hub.pending.pop(rid, None)
                hub.job_conn.pop(rid, None)
                hub.release(conn)
                return await self._generate_media_locked(prompt, kind, timeout=timeout,
                                                          aspect=aspect, authuser=authuser,
                                                          _exclude=conn.key)
            raise
        finally:
            hub.pending.pop(rid, None)
            hub.job_conn.pop(rid, None)
            hub.release(conn)
        media = (msg.get("media") if isinstance(msg, dict) else None) or []
        if not media:
            raise RuntimeError("extension returned no media")
        return media, (msg.get("text", "") if isinstance(msg, dict) else "")

    async def generate_stream(self, prompt, files=None, model=None, temporary=True,
                              authuser: str | None = None):
        if files:
            raise RuntimeError("the chrome backend does not support file/vision input yet")
        conn, send_prompt, fresh = await _acquire_chat(prompt, authuser, None)
        rid = hub.next_id()
        q: asyncio.Queue = asyncio.Queue()
        hub.streams[rid] = q
        hub.job_conn[rid] = conn.key
        try:
            await conn.send(
                {"type": "chat", "id": rid, "prompt": send_prompt,
                 "model": _model_name(model), "fresh": fresh}
            )
            sent = ""
            while True:
                kind, data = await asyncio.wait_for(q.get(), timeout=REQUEST_TIMEOUT)
                if kind == "delta":
                    # extension sends cumulative text; emit only the new suffix
                    new = data[len(sent):] if data.startswith(sent) else data
                    sent = data
                    if new:
                        yield ChromeOutput(text=sent, text_delta=new)
                elif kind == "done":
                    if data and data != sent:
                        tail = data[len(sent):] if data.startswith(sent) else data
                        if tail:
                            yield ChromeOutput(text=data, text_delta=tail)
                    return
                elif kind == "error":
                    raise RuntimeError(data)
        finally:
            hub.streams.pop(rid, None)
            hub.job_conn.pop(rid, None)
            hub.release(conn)


async def _acquire_chat(prompt: str, authuser: str | None,
                        exclude_key: int | None) -> tuple[TabConn, str, bool]:
    """A tab for a chat, the prompt to send it and whether to start a new conversation.

    With a session (CURRENT_CHAT) whose tab still holds its conversation, only the
    continuation goes, into that conversation. Otherwise the full prompt goes into a
    new conversation (Gemini web keeps it in history either way; `temporary` has no
    effect on this backend).
    """
    chat = CURRENT_CHAT.get()
    if chat is None:
        return await hub.acquire(authuser, exclude_key=exclude_key), prompt, True
    conn, continuing = await hub.acquire_session(chat.key, authuser, exclude_key)
    if continuing and chat.continuation:
        return conn, chat.continuation, False
    return conn, prompt, True


def _model_name(model) -> str | None:
    """server.py passes a gemini_webapi Model enum; the extension just needs a
    string hint (or None -- it uses whatever mode the tab has selected)."""
    if model is None:
        return None
    return getattr(model, "model_name", None) or str(model)


manager = ChromeManager()


def _update_eligibility(conn: TabConn) -> None:
    """Eligible if on a bare /app OR a self-declared worker (its conversation is
    bridge-made, not the human's). A busy tab keeps its state: its job navigated."""
    if not conn.busy:
        conn.eligible = conn.worker or not IN_CONVERSATION.search(conn.href)


def register_ws(app: FastAPI) -> None:
    """Register the /ws endpoint the extension tabs connect to."""

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:  # noqa: ANN202
        await ws.accept()
        key = hub.next_key()
        conn = TabConn(ws=ws, key=key, tab_id=f"tab{key}")
        hub.conns[key] = conn
        hub.notify_pool_changed()
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                mtype = msg.get("type")
                rid = msg.get("id")
                if mtype == "hello":
                    conn.tab_id = str(msg.get("tabId") or conn.tab_id)
                    conn.href = str(msg.get("href") or "")
                    conn.worker = bool(msg.get("worker"))
                    au = msg.get("authuser")
                    if au is None:
                        # Older content scripts don't send authuser; the tab URL
                        # still tells us the account (/u/N/, default 0).
                        m = re.search(r"/u/(\d+)/", conn.href)
                        au = m.group(1) if m else ("0" if conn.href else None)
                    conn.authuser = str(au) if au is not None else None
                    _update_eligibility(conn)
                    print(f"[hub] hello {conn.tab_id} u/{conn.authuser} "
                          f"eligible={conn.eligible} {conn.href[:60]}", flush=True)
                    continue
                if mtype == "url_changed":
                    # Gemini is a SPA: /app -> /app/<id> after a first prompt
                    # reloads nothing and sends no new hello.
                    conn.href = str(msg.get("href") or conn.href)
                    conn.worker = bool(msg.get("worker", conn.worker))
                    _update_eligibility(conn)
                    hub.record_url(conn)
                    hub.notify_pool_changed()
                    print(f"[hub] url_changed {conn.tab_id} eligible={conn.eligible} "
                          f"{conn.href[:60]}", flush=True)
                    continue
                if mtype in ("pong", "ka"):
                    continue
                if mtype == "delta" and rid in hub.streams:
                    await hub.streams[rid].put(("delta", msg.get("text", "")))
                elif mtype == "result":
                    if rid in hub.streams:
                        await hub.streams[rid].put(("done", msg.get("text", "")))
                    fut = hub.pending.get(rid)
                    if fut and not fut.done():
                        fut.set_result(msg)  # full dict: {text, media?}
                elif mtype == "error":
                    emsg = msg.get("message", "extension error")
                    if rid in hub.streams:
                        await hub.streams[rid].put(("error", emsg))
                    fut = hub.pending.get(rid)
                    if fut and not fut.done():
                        fut.set_exception(RuntimeError(emsg))
        except WebSocketDisconnect:
            pass
        finally:
            hub.conns.pop(key, None)
            # Fail ONLY this tab's in-flight jobs; other tabs' work continues.
            for job_id, job_key in list(hub.job_conn.items()):
                if job_key != key:
                    continue
                fut = hub.pending.get(job_id)
                if fut and not fut.done():
                    fut.set_exception(RuntimeError(f"gemini tab {conn.tab_id} disconnected"))
                q = hub.streams.get(job_id)
                if q is not None:
                    q.put_nowait(("error", f"gemini tab {conn.tab_id} disconnected"))
            hub.notify_pool_changed()
