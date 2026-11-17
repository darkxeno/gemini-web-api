"""Lazy, shared gemini_webapi client with re-init on failure.

A single GeminiClient is shared across all requests, initialized on first use.

It does NOT refresh __Secure-1PSIDTS when the cookies came from Chrome:
rotation mints a new cookie and invalidates the old one, which logs the browser
out of that Google account. Chrome owns that session and keeps it fresh itself,
so a stale token here simply raises AuthError and the retry path below re-reads
Chrome's current jar. See `config.rotate_cookies_ourselves()`.

In `api` mode the cookies come from the cookie-provider extension
(`apply_cookies`, fed by POST /v1/cookies), and a chat session continues its own
Gemini thread: its [cid, rid, rcid, ...] metadata lives in `sessions`, so a
later turn sends only what is new. A thread that can't be continued (history
changed, another account, Google refused it) is replaced by a new one with the
full prompt.
"""

from __future__ import annotations

import asyncio

from gemini_webapi import GeminiClient
from gemini_webapi.exceptions import APIError, AuthError

from . import config, web_models
from .sessions import CURRENT_CHAT, ApiSessionStore, ChatSession

# Errors worth a one-shot re-init + retry: auth expiry, and transient Google
# API errors (e.g. 1097) that leave a long-running client wedged on a stale
# build label / session — a fresh client clears them.
_RETRYABLE = (AuthError, APIError)


def _for_account(client: GeminiClient, kwargs: dict) -> dict:
    """Resolve a model NAME against the models this account actually offers.

    The registry is per account and comes from Google at init, so a name that
    works on one profile can be absent on another: on 2026-10-13 some accounts
    started listing only `gemini-auto`, and `gemini-3-flash` raised
    `ValueError: Unknown model name` -- a 500 on every request. A name the
    account lacks becomes `model=None`, i.e. Google's own default for that
    account (the library's documented "omit it" behaviour). Non-string specs
    (the video path's raw header dict) and libraries without `resolve_model`
    (2.0.x) pass through untouched.
    """
    name = kwargs.get("model")
    web = web_models.lookup(name) if config.BACKEND == "api" else None
    if web is not None:
        # Asked for the web app's way (web_models): the registry entry stalls.
        return {**kwargs, "model": web_models.library_model(web)}
    resolve = getattr(client, "resolve_model", None)
    if not isinstance(name, str) or resolve is None:
        return kwargs
    try:
        return {**kwargs, "model": resolve(name)}
    except ValueError:
        print(f"[gemini] account has no model {name!r}; using its default", flush=True)
        return {**kwargs, "model": None}


def _no_library_browser_scan() -> None:
    """Stop gemini_webapi reading local browsers' cookies on every init.

    Its init collects cookie candidates up front, and that includes every local
    browser's store through browser_cookie3 -- on macOS a Keychain prompt per
    start, and possibly another account. api mode's cookies come from the extension.
    """
    import sys

    import gemini_webapi.utils.get_access_token  # noqa: F401  (load the module)

    sys.modules["gemini_webapi.utils.get_access_token"].load_browser_cookies = lambda **_: {}


class GeminiManager:
    def __init__(self) -> None:
        self._client: GeminiClient | None = None
        self._lock = asyncio.Lock()
        # The cookie session (config.account_id) the client was made with, and
        # whether pushed cookies of ANOTHER session want a new client.
        self._account = ""
        self._stale = False
        self._closing: set[asyncio.Task] = set()
        self.sessions = ApiSessionStore()

    async def _make_client(self) -> GeminiClient:
        # Route to the configured Google multi-login account (u/N) before init,
        # so the access-token fetch and all requests target the right account.
        # Also make the session carry the full google.com cookie jar, which is
        # required to authenticate a non-default account (and hardens auth for
        # the default one so tokens refresh without frequent re-login).
        from .account import apply_authuser, install_full_jar

        apply_authuser(config.AUTHUSER)
        install_full_jar(config.get_full_jar)
        api_mode = config.BACKEND == "api"
        if api_mode:
            _no_library_browser_scan()

        psid, psidts = config.get_cookies()
        if not psid:
            if api_mode:
                raise AuthError(
                    "No cookies from the cookie-provider extension yet: load extension-cookies/ "
                    "in the Chrome profile signed in to Google (ocweb opens it on first run)."
                )
            raise AuthError(
                "No Gemini credentials. Set GEMINI_1PSID/GEMINI_1PSIDTS or log in "
                "to gemini.google.com in local Chrome."
            )
        proxy = config.os.getenv("GEMINI_PROXY") or None
        self._account = config.account_id(psid)
        client = GeminiClient(psid, psidts, proxy=proxy)
        # See config.rotate_cookies_ourselves(): rotating a session Chrome also
        # holds logs the browser out, so by default Chrome owns the refresh.
        await client.init(
            timeout=120,
            auto_refresh=config.rotate_cookies_ourselves(),
            refresh_interval=540,
        )
        if api_mode:
            web_models.set_session(getattr(client, "language", None),
                                   getattr(client, "build_label", None))
            web_models.install()
        if api_mode and (not getattr(client, "access_token", None)
                         or getattr(client, "_cookie_source", "") == "Guest"):
            # When no cookie group authenticates, gemini_webapi falls back to a
            # logged-out guest session -- which would answer every chat anonymously.
            await client.close()
            raise AuthError(
                "The cookies the extension pushed don't sign in to Google (only a guest "
                "session came up): sign in again in the extension's Chrome profile."
            )
        return client

    async def get(self) -> GeminiClient:
        if self._client is not None and not self._stale:
            return self._client
        async with self._lock:
            if self._stale and self._client is not None:
                # Requests still running on the old client finish first.
                old, self._client = self._client, None
                task = asyncio.create_task(old.close(delay=300))
                self._closing.add(task)
                task.add_done_callback(self._closing.discard)
            self._stale = False
            if self._client is None:
                self._client = await self._make_client()
        return self._client

    def apply_cookies(self, jar: dict[str, str]) -> str:
        """Take cookies the extension pushed. Returns what happened.

        The same session (Chrome rotated __Secure-1PSIDTS): set on the live
        client in place, nothing re-initializes. Another session (a different
        account or a new login): the next request makes a new client, whose
        access token belongs to it.
        """
        if self._client is None:
            return "stored"  # the first request initializes with them
        if config.account_id(jar.get("__Secure-1PSID")) != self._account:
            self._stale = True
            return "reinit"
        self._client.cookies = dict(jar)
        return "updated"

    async def reset(self) -> None:
        """Drop the client so the next get() re-initializes (e.g. after AuthError)."""
        async with self._lock:
            if self._client is not None:
                try:
                    await self._client.close()
                except Exception:  # noqa: BLE001
                    pass
            self._client = None

    async def generate(self, *args, **kwargs):
        """generate_content with one automatic re-init retry on auth/API failure.

        A long-running shared client can go stale (rotated build label, expired
        session) and throw AuthError or a transient APIError (e.g. 1097). We drop
        the wedged client and retry once against a freshly-initialized one.
        """
        chat = self._session(kwargs)
        if chat is not None:
            return await self._generate_session(chat, *args, **kwargs)
        client = await self.get()
        try:
            return await client.generate_content(*args, **_for_account(client, kwargs))
        except _RETRYABLE:
            await self.reset()
            client = await self.get()
            return await client.generate_content(*args, **_for_account(client, kwargs))

    async def generate_stream(self, *args, **kwargs):
        """True token streaming: yields ModelOutput chunks (with `text_delta`).

        Same one-shot re-init retry as `generate`, but only if the failure happens
        before any chunk was yielded (we can't cleanly restart mid-stream).
        """
        chat = self._session(kwargs)
        if chat is not None:
            async for out in self._stream_session(chat, *args, **kwargs):
                yield out
            return
        client = await self.get()
        yielded = False
        try:
            async for out in client.generate_content_stream(*args, **_for_account(client, kwargs)):
                yielded = True
                yield out
            return
        except _RETRYABLE:
            if yielded:
                raise
        # Nothing was emitted yet — safe to re-init and restart once.
        await self.reset()
        client = await self.get()
        async for out in client.generate_content_stream(*args, **_for_account(client, kwargs)):
            yield out

    # ---- api mode: a Gemini thread per chat session -------------------------
    @staticmethod
    def _session(kwargs: dict) -> ChatSession | None:
        """The chat session to continue, in `api` mode (files always go fresh)."""
        if config.BACKEND != "api" or kwargs.get("files"):
            return None
        return CURRENT_CHAT.get()

    async def _generate_session(self, chat: ChatSession, prompt, **kwargs):
        async with self.sessions.lock(chat.key):
            client = await self.get()
            thread = self.sessions.thread_for(chat, self._account)
            if thread is not None:
                gchat = client.start_chat(metadata=thread)
                try:
                    out = await client.generate_content(
                        chat.continuation, chat=gchat, **_for_account(client, kwargs))
                    print(f"[api] session {chat.key[:8]}: continued its thread", flush=True)
                    self.sessions.save(chat, gchat.metadata, self._account)
                    return out
                except Exception as e:  # noqa: BLE001
                    print(f"[api] session {chat.key[:8]}: its thread failed "
                          f"({type(e).__name__}: {e}); starting a new one", flush=True)
                    self.sessions.drop(chat.key)
            for attempt in (1, 2):
                client = await self.get()
                gchat = client.start_chat()
                try:
                    out = await client.generate_content(
                        prompt, chat=gchat, **_for_account(client, kwargs))
                    break
                except _RETRYABLE:
                    if attempt == 2:
                        raise
                    await self.reset()
            print(f"[api] session {chat.key[:8]}: new thread", flush=True)
            self.sessions.save(chat, gchat.metadata, self._account)
            return out

    async def _stream_session(self, chat: ChatSession, prompt, **kwargs):
        async with self.sessions.lock(chat.key):
            client = await self.get()
            thread = self.sessions.thread_for(chat, self._account)
            if thread is not None:
                gchat = client.start_chat(metadata=thread)
                yielded = False
                try:
                    async for out in client.generate_content_stream(
                            chat.continuation, chat=gchat, **_for_account(client, kwargs)):
                        yielded = True
                        yield out
                    print(f"[api] session {chat.key[:8]}: continued its thread", flush=True)
                    self.sessions.save(chat, gchat.metadata, self._account)
                    return
                except Exception as e:  # noqa: BLE001
                    self.sessions.drop(chat.key)
                    if yielded:
                        raise  # can't restart a reply the client has half-received
                    print(f"[api] session {chat.key[:8]}: its thread failed "
                          f"({type(e).__name__}: {e}); starting a new one", flush=True)
            for attempt in (1, 2):
                client = await self.get()
                gchat = client.start_chat()
                yielded = False
                try:
                    async for out in client.generate_content_stream(
                            prompt, chat=gchat, **_for_account(client, kwargs)):
                        yielded = True
                        yield out
                    break
                except _RETRYABLE:
                    if yielded or attempt == 2:
                        raise
                    await self.reset()
            print(f"[api] session {chat.key[:8]}: new thread", flush=True)
            self.sessions.save(chat, gchat.metadata, self._account)


manager = GeminiManager()
