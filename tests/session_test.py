"""Chat sessions on the chrome backend: prompts, tab pinning, eviction, URL tracking.

An OpenAI client (OpenCode) resends the whole conversation every call. A session's
first turn starts a new Gemini conversation in a tab with the full prompt; later
turns send only what is new, into that same conversation. Whenever the session
can't be sure its tab still shows its conversation, it starts over with the full
prompt -- it must never continue another session's conversation.
"""
import asyncio
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gemini_openai import chrome_backend as cb
from gemini_openai import server
from gemini_openai.openai_schemas import (
    ChatMessage,
    continuation,
    flatten_messages,
    session_key,
)

APP = "https://gemini.google.com/app"
CONV_A = "https://gemini.google.com/app/aaaaaaaa11"
CONV_B = "https://gemini.google.com/app/bbbbbbbb22"


def msgs(*pairs, **extra):
    return [ChatMessage(role=r, content=c, **extra) for r, c in pairs]


@pytest.fixture(autouse=True)
def session_map(tmp_path, monkeypatch):
    path = tmp_path / "session-map.json"
    monkeypatch.setattr(cb, "SESSION_MAP_FILE", path)
    return path


def make_hub(*hrefs):
    hub = cb.Hub()
    for i, href in enumerate(hrefs, start=1):
        hub.conns[i] = cb.TabConn(ws=object(), key=i, tab_id=f"t{i}", authuser="0",
                                  href=href, connected_at=float(i))
    return hub


def navigate(hub, conn, href):
    """What a url_changed message does."""
    conn.href = href
    hub.record_url(conn)


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
FIRST = [("system", "You are a coding agent."), ("user", "fix the bug")]


def test_session_key_is_stable_across_turns_and_differs_between_conversations():
    turn1 = msgs(*FIRST)
    turn2 = msgs(*FIRST, ("assistant", "done"), ("user", "now add a test"))
    other = msgs(("system", "You are a coding agent."), ("user", "write docs"))
    assert session_key(turn1) == session_key(turn2)
    assert session_key(turn1) != session_key(other)


def test_continuation_is_none_on_the_first_turn():
    assert continuation(msgs(*FIRST)) is None


def test_continuation_is_the_users_next_message():
    turn2 = msgs(*FIRST, ("assistant", "done"), ("user", "now add a test"))
    assert continuation(turn2) == "now add a test"


def test_continuation_carries_every_result_of_a_parallel_tool_call():
    calls = [
        {"id": f"c{i}", "type": "function", "function": {"name": "read", "arguments": "{}"}}
        for i in (1, 2, 3)
    ]
    turn = [
        *msgs(*FIRST),
        ChatMessage(role="assistant", content=None, tool_calls=calls),
        ChatMessage(role="tool", content="file one", name="read", tool_call_id="c1"),
        ChatMessage(role="tool", content="file two", name="read", tool_call_id="c2"),
        ChatMessage(role="tool", content="file three", name="read", tool_call_id="c3"),
    ]
    cont = continuation(turn)
    assert "file one" in cont and "file two" in cont and "file three" in cont


def test_flatten_messages_keeps_the_whole_history():
    """The full prompt is for a NEW conversation, so it must carry everything."""
    turn2 = msgs(*FIRST, ("assistant", "done"), ("user", "now add a test"))
    prompt, _ = flatten_messages(turn2)
    assert "fix the bug" in prompt and "done" in prompt and "now add a test" in prompt


# --------------------------------------------------------------------------- #
# Pinning
# --------------------------------------------------------------------------- #
def test_first_turn_pins_a_tab_and_later_turns_continue_in_it():
    hub = make_hub(APP, APP)

    async def go():
        conn, continuing = await hub.acquire_session("s1")
        assert not continuing
        navigate(hub, conn, CONV_A)
        hub.release(conn)
        again, continuing = await hub.acquire_session("s1")
        assert again is conn and continuing

    asyncio.run(go())


def test_sessions_get_their_own_tabs():
    hub = make_hub(APP, APP)

    async def go():
        a, _ = await hub.acquire_session("s1")
        hub.release(a)
        b, continuing = await hub.acquire_session("s2")
        assert b is not a and not continuing

    asyncio.run(go())


def test_a_new_session_evicts_the_least_recently_used_idle_session():
    hub = make_hub(APP, APP)

    async def go():
        a, _ = await hub.acquire_session("s1")
        navigate(hub, a, CONV_A)
        hub.release(a)
        b, _ = await hub.acquire_session("s2")
        navigate(hub, b, CONV_B)
        hub.release(b)
        hub.session_used["s1"] -= 100  # s1 is the older one

        c, continuing = await hub.acquire_session("s3")
        assert c is a and not continuing
        assert "s1" not in hub.session_tabs and hub.session_tabs["s2"] == b.key

    asyncio.run(go())


def test_an_evicted_session_starts_over_instead_of_continuing_another_conversation():
    hub = make_hub(APP)

    async def go():
        a, _ = await hub.acquire_session("s1")
        navigate(hub, a, CONV_A)
        hub.release(a)
        b, _ = await hub.acquire_session("s2")  # takes s1's only tab
        navigate(hub, b, CONV_B)
        hub.release(b)
        _, continuing = await hub.acquire_session("s1")
        assert not continuing  # the tab now shows s2's conversation

    asyncio.run(go())


def test_a_pinned_tab_that_moved_to_another_conversation_does_not_continue():
    hub = make_hub(APP)

    async def go():
        conn, _ = await hub.acquire_session("s1")
        navigate(hub, conn, CONV_A)
        hub.release(conn)
        conn.href = CONV_B  # a human clicked another chat in the sidebar
        again, continuing = await hub.acquire_session("s1")
        assert again is conn and not continuing

    asyncio.run(go())


def test_a_busy_pinned_tab_is_waited_for():
    hub = make_hub(APP, APP)

    async def go():
        conn, _ = await hub.acquire_session("s1")
        navigate(hub, conn, CONV_A)
        waiter = asyncio.create_task(hub.acquire_session("s1"))
        await asyncio.sleep(0.05)
        assert not waiter.done()  # never jumps to the other tab mid-conversation
        hub.release(conn)
        again, continuing = await asyncio.wait_for(waiter, timeout=2)
        assert again is conn and continuing

    asyncio.run(go())


def test_a_reconnected_tab_is_repinned_by_its_conversation_url():
    hub = make_hub(APP)

    async def go():
        conn, _ = await hub.acquire_session("s1")
        navigate(hub, conn, CONV_A)
        hub.release(conn)
        del hub.conns[conn.key]  # the service worker dropped the socket
        hub.conns[9] = cb.TabConn(ws=object(), key=9, tab_id="t1", authuser="0",
                                  href=CONV_A, worker=True)
        again, continuing = await hub.acquire_session("s1")
        assert again.key == 9 and continuing

    asyncio.run(go())


def test_pins_survive_a_server_restart_through_the_session_map(session_map):
    hub = make_hub(APP)

    async def first():
        conn, _ = await hub.acquire_session("s1")
        navigate(hub, conn, CONV_A)
        hub.release(conn)

    asyncio.run(first())
    assert json.loads(session_map.read_text()) == {"s1": CONV_A}

    restarted = make_hub(CONV_A)

    async def second():
        _, continuing = await restarted.acquire_session("s1")
        assert continuing

    asyncio.run(second())


def test_a_retry_after_a_tab_failure_starts_over_on_another_tab():
    hub = make_hub(APP, APP)

    async def go():
        conn, _ = await hub.acquire_session("s1")
        navigate(hub, conn, CONV_A)
        hub.release(conn)
        other, continuing = await hub.acquire_session("s1", exclude_key=conn.key)
        assert other is not conn and not continuing

    asyncio.run(go())


def test_media_jobs_avoid_pinned_tabs_and_unpin_one_they_must_take():
    hub = make_hub(APP, APP)

    async def go():
        pinned, _ = await hub.acquire_session("s1")
        hub.release(pinned)
        media = await hub.acquire()
        assert media is not pinned and "s1" in hub.session_tabs
        hub.release(media)
        hub.conns.pop(media.key)
        only = await hub.acquire()
        assert only is pinned and "s1" not in hub.session_tabs

    asyncio.run(go())


# --------------------------------------------------------------------------- #
# What the tab is sent
# --------------------------------------------------------------------------- #
def test_chat_sends_the_continuation_only_into_its_own_conversation(monkeypatch):
    hub = make_hub(APP)
    monkeypatch.setattr(cb, "hub", hub)

    async def go():
        cb.CURRENT_CHAT.set(cb.ChatSession(key="s1"))
        conn, prompt, fresh = await cb._acquire_chat("FULL turn 1", None, None)
        assert (prompt, fresh) == ("FULL turn 1", True)
        navigate(hub, conn, CONV_A)
        hub.release(conn)

        cb.CURRENT_CHAT.set(cb.ChatSession(key="s1", continuation="next"))
        conn, prompt, fresh = await cb._acquire_chat("FULL turn 2", None, None)
        assert (prompt, fresh) == ("next", False)

    asyncio.run(go())


def test_chat_without_a_session_always_starts_a_new_conversation(monkeypatch):
    hub = make_hub(APP)
    monkeypatch.setattr(cb, "hub", hub)

    async def go():
        _, prompt, fresh = await cb._acquire_chat("FULL", None, None)
        assert (prompt, fresh) == ("FULL", True)

    asyncio.run(go())


def test_a_session_that_lost_its_tab_sends_the_full_prompt(monkeypatch):
    hub = make_hub(APP, APP)
    monkeypatch.setattr(cb, "hub", hub)

    async def go():
        cb.CURRENT_CHAT.set(cb.ChatSession(key="s1"))
        conn, _, _ = await cb._acquire_chat("FULL turn 1", None, None)
        navigate(hub, conn, CONV_A)
        hub.release(conn)
        del hub.conns[conn.key]  # tab closed

        cb.CURRENT_CHAT.set(cb.ChatSession(key="s1", continuation="next"))
        _, prompt, fresh = await cb._acquire_chat("FULL turn 2", None, None)
        assert (prompt, fresh) == ("FULL turn 2", True)

    asyncio.run(go())


# --------------------------------------------------------------------------- #
# The extension's messages, over the real /ws endpoint
# --------------------------------------------------------------------------- #
def ws_client(monkeypatch):
    hub = cb.Hub()
    monkeypatch.setattr(cb, "hub", hub)
    app = FastAPI()
    cb.register_ws(app)
    return hub, TestClient(app)


def wait_for(pred, timeout=2.0):
    async def go():
        for _ in range(int(timeout / 0.01)):
            if pred():
                return
            await asyncio.sleep(0.01)
        raise AssertionError("condition never held")

    asyncio.run(go())


def test_a_human_starting_a_chat_in_a_bare_tab_makes_it_ineligible(monkeypatch):
    hub, client = ws_client(monkeypatch)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "hello", "tabId": "t1", "href": APP, "worker": False})
        wait_for(lambda: hub.conns and next(iter(hub.conns.values())).href == APP)
        conn = next(iter(hub.conns.values()))
        assert conn.eligible
        ws.send_json({"type": "url_changed", "tabId": "t1", "href": CONV_A, "worker": False})
        wait_for(lambda: conn.href == CONV_A)
        assert not conn.eligible


def test_a_worker_tab_stays_eligible_in_its_conversation_and_records_it(monkeypatch, session_map):
    hub, client = ws_client(monkeypatch)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "hello", "tabId": "t1", "href": APP, "worker": False})
        wait_for(lambda: hub.conns and next(iter(hub.conns.values())).href == APP)
        conn = next(iter(hub.conns.values()))
        hub.session_tabs["s1"] = conn.key
        ws.send_json({"type": "ka"})  # ignored
        ws.send_json({"type": "url_changed", "tabId": "t1", "href": CONV_A, "worker": True})
        wait_for(lambda: conn.href == CONV_A)
        assert conn.eligible
        assert json.loads(session_map.read_text()) == {"s1": CONV_A}


# --------------------------------------------------------------------------- #
# /health
# --------------------------------------------------------------------------- #
def test_health_counts_eligible_and_free_tabs(monkeypatch):
    hub = make_hub(APP, APP, CONV_B)
    hub.conns[3].eligible = False  # a human's conversation
    hub.session_tabs["s1"] = 1     # pinned, so not free for a new session
    monkeypatch.setattr(server, "_chrome_hub", hub)
    health = asyncio.run(server.health())
    assert health["eligible_tabs"] == 2
    assert health["free_tabs"] == 1
