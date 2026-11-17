"""GEMINI_BACKEND=api: cookies pushed by the cookie-provider extension, and a
Gemini thread ([cid, rid, rcid, ...]) per chat session.

No network: the endpoint runs in a TestClient, and a fake gemini_webapi client
records what would have been sent to Google.
"""
import asyncio
import json
import stat

import pytest
from fastapi.testclient import TestClient

from gemini_openai import config, server
from gemini_openai import gemini_pool as gp
from gemini_openai.openai_schemas import ChatMessage, continuation, session_key
from gemini_openai.sessions import CURRENT_CHAT, ApiSessionStore, ChatSession

PSID = "psid-VALUE-never-logged"
PSIDTS = "psidts-1"


@pytest.fixture(autouse=True)
def cookie_store(tmp_path, monkeypatch):
    path = tmp_path / "cookies.json"
    monkeypatch.setattr(config, "COOKIE_STORE", path)
    monkeypatch.setattr(config, "_pushed", None)
    monkeypatch.setattr(config, "BACKEND", "api")
    return path


# --------------------------------------------------------------------------- #
# POST /v1/cookies
# --------------------------------------------------------------------------- #
class FakeManager:
    def __init__(self):
        self.applied = []

    def apply_cookies(self, jar):
        self.applied.append(dict(jar))
        return "updated"


@pytest.fixture
def api(monkeypatch):
    manager = FakeManager()
    monkeypatch.setattr(server, "webapi_manager", manager)
    # TestClient's peer is "testclient"; stand it in for this machine.
    monkeypatch.setattr(server, "_LOOPBACK", server._LOOPBACK | {"testclient"})
    return TestClient(server.app), manager


def push(client, body=None, **headers):
    body = body if body is not None else {"secure_1psid": PSID, "secure_1psidts": PSIDTS,
                                          "cookies": {"SID": "sid", "SAPISID": "sap"}}
    return client.post("/v1/cookies", content=json.dumps(body),
                       headers={"Content-Type": "application/json", **headers})


def test_a_push_is_stored_privately_and_applied(api, cookie_store, capsys):
    client, manager = api
    r = push(client, Origin="chrome-extension://abcdef")
    assert r.status_code == 200 and r.json()["changed"]
    jar = {"SID": "sid", "SAPISID": "sap", "__Secure-1PSID": PSID, "__Secure-1PSIDTS": PSIDTS}
    assert manager.applied == [jar]
    assert json.loads(cookie_store.read_text())["jar"] == jar
    assert stat.S_IMODE(cookie_store.stat().st_mode) == 0o600
    # Live session cookies: never in the response or the log.
    assert PSID not in r.text and PSID not in capsys.readouterr().out


def test_an_unchanged_push_is_not_reapplied(api):
    client, manager = api
    push(client)
    r = push(client)
    assert r.json()["applied"] == "unchanged" and len(manager.applied) == 1


def test_another_mode_stores_but_does_not_apply(api, monkeypatch):
    client, manager = api
    monkeypatch.setattr(config, "BACKEND", "webapi")
    r = push(client)
    assert r.status_code == 200 and not r.json()["used"] and manager.applied == []


def test_a_push_from_another_machine_is_refused(monkeypatch):
    monkeypatch.setattr(server, "webapi_manager", FakeManager())
    r = push(TestClient(server.app))  # peer "testclient", not loopback
    assert r.status_code == 403 and not config.pushed_jar()


def test_a_push_from_a_web_page_is_refused(api):
    client, manager = api
    r = push(client, Origin="https://evil.example")
    assert r.status_code == 403 and manager.applied == []


def test_a_push_must_be_json(api):
    client, _ = api
    r = client.post("/v1/cookies", content="secure_1psid=x", headers={"Content-Type": "text/plain"})
    assert r.status_code == 415


def test_a_push_needs_the_session_cookie(api):
    client, _ = api
    assert push(client, {"secure_1psidts": "x"}).status_code == 422


def test_health_reports_how_long_ago_cookies_were_pushed(api):
    client, _ = api
    assert asyncio.run(server.health())["cookies_pushed_s_ago"] is None
    push(client)
    assert asyncio.run(server.health())["cookies_pushed_s_ago"] == 0


# --------------------------------------------------------------------------- #
# Cookie source
# --------------------------------------------------------------------------- #
def test_api_mode_reads_the_pushed_cookies(monkeypatch):
    monkeypatch.delenv("GEMINI_1PSID", raising=False)
    config.store_pushed_jar({"__Secure-1PSID": PSID, "__Secure-1PSIDTS": PSIDTS, "SID": "sid"})
    assert config.get_cookies() == (PSID, PSIDTS)
    assert config.get_full_jar()["SID"] == "sid"


def test_other_modes_ignore_pushed_cookies(monkeypatch):
    config.store_pushed_jar({"__Secure-1PSID": PSID})
    monkeypatch.setattr(config, "BACKEND", "webapi")
    assert not config._use_pushed()


def test_pushed_cookies_survive_a_restart(cookie_store, monkeypatch):
    config.store_pushed_jar({"__Secure-1PSID": PSID})
    monkeypatch.setattr(config, "_pushed", None)  # a new process
    assert config.pushed_jar() == {"__Secure-1PSID": PSID}


def test_pushed_cookies_are_never_rotated_by_the_server(monkeypatch):
    monkeypatch.delenv("GEMINI_1PSID", raising=False)
    monkeypatch.delenv("GEMINI_AUTO_REFRESH", raising=False)
    config.store_pushed_jar({"__Secure-1PSID": PSID})
    assert not config.rotate_cookies_ourselves()


# --------------------------------------------------------------------------- #
# Applying pushed cookies to the live client
# --------------------------------------------------------------------------- #
class FakeClient:
    """Stands in for gemini_webapi.GeminiClient."""

    def __init__(self, fail_on=None):
        self.cookies = {}
        self.sent = []          # (prompt, metadata it continued, temporary)
        self.closed = None
        self.fail_on = fail_on  # metadata cid whose thread Google refuses
        self.n = 0

    def start_chat(self, metadata=None):
        return FakeChat(metadata)

    async def generate_content(self, prompt, chat=None, files=None, model=None, temporary=False):
        self.sent.append((prompt, list(chat.metadata) if chat and chat.metadata else None, temporary))
        if chat and self.fail_on and chat.metadata and chat.metadata[0] == self.fail_on:
            raise gp.APIError("thread gone")
        self.n += 1
        if chat is not None:
            cid = chat.metadata[0] if chat.metadata else f"c{self.n}"
            chat.metadata = [cid, f"r{self.n}", f"rc{self.n}"]
        return type("Out", (), {"text": f"reply {self.n}"})()

    async def generate_content_stream(self, prompt, chat=None, **kw):
        yield await self.generate_content(prompt, chat=chat, **kw)

    async def close(self, delay=0):
        self.closed = delay


class FakeChat:
    def __init__(self, metadata):
        self.metadata = list(metadata) if metadata else None


def manager_with(client, account="acct"):
    m = gp.GeminiManager()
    m._client, m._account = client, account
    return m


def test_a_rotated_cookie_updates_the_live_client_in_place():
    client = FakeClient()
    m = manager_with(client, config.account_id(PSID))
    assert m.apply_cookies({"__Secure-1PSID": PSID, "__Secure-1PSIDTS": "new"}) == "updated"
    assert client.cookies["__Secure-1PSIDTS"] == "new" and m._client is client


def test_another_session_makes_a_new_client_and_lets_the_old_one_finish(monkeypatch):
    old, new = FakeClient(), FakeClient()
    m = manager_with(old, config.account_id(PSID))
    assert m.apply_cookies({"__Secure-1PSID": "another-account"}) == "reinit"

    async def make():
        return new

    monkeypatch.setattr(m, "_make_client", make)

    async def go():
        assert await m.get() is new
        await asyncio.sleep(0)  # let the delayed close start
        assert old.closed == 300

    asyncio.run(go())


# --------------------------------------------------------------------------- #
# A Gemini thread per chat session
# --------------------------------------------------------------------------- #
SYSTEM = ChatMessage(role="system", content="You are a coding agent.")


def chat_for(*pairs):
    msgs = [SYSTEM] + [ChatMessage(role=r, content=c) for r, c in pairs]
    return ChatSession(key=session_key(msgs), continuation=continuation(msgs), messages=tuple(msgs))


TURN1 = [("user", "fix the bug")]
TURN2 = [*TURN1, ("assistant", "reply 1"), ("user", "now add a test")]
TURN3 = [*TURN2, ("assistant", "reply 2"), ("user", "and docs")]


@pytest.fixture
def sessions(tmp_path, monkeypatch):
    path = tmp_path / "sessions.json"
    monkeypatch.setattr(gp, "ApiSessionStore", lambda: ApiSessionStore(path))
    return path


def run_turns(m, *turns, stream=False):
    async def go():
        for pairs in turns:
            CURRENT_CHAT.set(chat_for(*pairs))
            if stream:
                async for _ in m.generate_stream("FULL", files=None, temporary=True):
                    pass
            else:
                await m.generate("FULL", files=None, temporary=True)

    asyncio.run(go())


def test_later_turns_continue_the_thread_with_only_what_is_new(sessions):
    client = FakeClient()
    m = manager_with(client)
    run_turns(m, TURN1, TURN2, TURN3)
    assert client.sent == [
        ("FULL", None, True),                          # new thread
        ("now add a test", ["c1", "r1", "rc1"], True),  # continued
        ("and docs", ["c1", "r2", "rc2"], True),
    ]


def test_streaming_continues_the_thread_too(sessions):
    client = FakeClient()
    run_turns(manager_with(client), TURN1, TURN2, stream=True)
    assert client.sent[1] == ("now add a test", ["c1", "r1", "rc1"], True)


def test_a_changed_history_starts_a_new_thread(sessions):
    """OpenCode compacted or edited the conversation: the stored thread isn't it."""
    client = FakeClient()
    m = manager_with(client)
    edited = [("user", "fix the bug"), ("assistant", "reply 1"), ("user", "EDITED"),
              ("assistant", "reply 2"), ("user", "and docs")]
    run_turns(m, TURN1, TURN2, edited)
    assert client.sent[2] == ("FULL", None, True)


def test_a_missed_turn_starts_a_new_thread(sessions):
    client = FakeClient()
    m = manager_with(client)
    run_turns(m, TURN1, TURN3)  # turn 2 never went through this thread
    assert client.sent[1] == ("FULL", None, True)


def test_another_account_never_continues_a_thread_it_cant_see(sessions):
    client = FakeClient()
    m = manager_with(client)
    run_turns(m, TURN1)
    m._account = "another-account"
    run_turns(m, TURN2)
    assert client.sent[1] == ("FULL", None, True)


def test_a_refused_thread_falls_back_to_a_new_one(sessions):
    client = FakeClient(fail_on="c1")
    m = manager_with(client)
    run_turns(m, TURN1, TURN2)
    assert client.sent[1][1] == ["c1", "r1", "rc1"]   # tried to continue
    assert client.sent[2] == ("FULL", None, True)     # then started over


def test_threads_survive_a_restart(sessions):
    client = FakeClient()
    run_turns(manager_with(client), TURN1)
    restarted = manager_with(client)                  # new store, same file
    run_turns(restarted, TURN2)
    assert client.sent[1] == ("now add a test", ["c1", "r1", "rc1"], True)
    assert stat.S_IMODE(sessions.stat().st_mode) == 0o600


def test_sessions_run_in_parallel_on_their_own_threads(sessions):
    client = FakeClient()
    m = manager_with(client)
    other = [("user", "write docs")]

    async def go():
        async def turn(pairs):
            CURRENT_CHAT.set(chat_for(*pairs))
            await m.generate("FULL", files=None, temporary=True)

        await asyncio.gather(turn(TURN1), turn(other))

    asyncio.run(go())
    stored = json.loads(sessions.read_text())
    assert len(stored) == 2
    assert {e["metadata"][0] for e in stored.values()} == {"c1", "c2"}


def test_other_modes_never_use_sessions(sessions, monkeypatch):
    monkeypatch.setattr(config, "BACKEND", "webapi")
    client = FakeClient()
    run_turns(manager_with(client), TURN1, TURN2)
    assert [s[1] for s in client.sent] == [None, None]  # no chat= at all
    assert not sessions.exists()


# --------------------------------------------------------------------------- #
# No macOS Keychain, no guest sessions
# --------------------------------------------------------------------------- #
def test_api_mode_never_reads_chromes_cookie_store_even_before_a_push(monkeypatch):
    """Chrome's store is decrypted with a Keychain key: a prompt on every read."""
    monkeypatch.delenv("GEMINI_1PSID", raising=False)

    def keychain(*_a, **_k):
        raise AssertionError("read Chrome's cookie store")

    monkeypatch.setattr(config, "_cookie_stores", keychain)
    monkeypatch.setattr(config, "cookies_via_cdp", keychain)
    assert config.get_cookies() == (None, None)
    assert config.get_full_jar() == {}


MADE_CLIENTS: list = []


class FakeGeminiClient:
    def __init__(self, psid, psidts, proxy=None):
        self.access_token, self._cookie_source, self.closed = "token", "Base Cookies", False
        MADE_CLIENTS.append(self)

    async def init(self, **_):
        pass

    async def close(self, delay=0):
        self.closed = True


def make_client_with(monkeypatch, **state):
    monkeypatch.delenv("GEMINI_1PSID", raising=False)
    config.store_pushed_jar({"__Secure-1PSID": PSID})
    MADE_CLIENTS.clear()

    class Client(FakeGeminiClient):
        async def init(self, **_):
            self.__dict__.update(state)

    monkeypatch.setattr(gp, "GeminiClient", Client)
    return asyncio.run(gp.GeminiManager()._make_client())


def test_api_mode_turns_off_the_librarys_own_browser_scan(monkeypatch):
    import sys

    make_client_with(monkeypatch)
    scan = sys.modules["gemini_webapi.utils.get_access_token"].load_browser_cookies
    assert scan(domain_name=".google.com") == {}


def test_a_guest_session_is_refused_not_used(monkeypatch):
    with pytest.raises(gp.AuthError, match="guest"):
        make_client_with(monkeypatch, access_token=None, _cookie_source="Guest")
    assert MADE_CLIENTS[-1].closed


def test_no_push_yet_names_the_extension(monkeypatch):
    monkeypatch.delenv("GEMINI_1PSID", raising=False)
    with pytest.raises(gp.AuthError, match="extension-cookies"):
        asyncio.run(gp.GeminiManager()._make_client())


def test_health_says_whether_cookies_came_since_this_server_started(api, monkeypatch):
    """A jar stored by an earlier run may have been invalidated by a rotation since."""
    monkeypatch.setattr(config, "_pushed_since_start", False)
    client, _ = api
    assert asyncio.run(server.health())["cookies_pushed_since_start"] is False
    push(client)
    assert asyncio.run(server.health())["cookies_pushed_since_start"] is True
