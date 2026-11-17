"""Live check of GEMINI_BACKEND=api: two chat turns that must share one Gemini thread.

Turn 1 asks the model to remember a random number; turn 2 (the full history, as
OpenAI clients send it) asks for it back. The server should send turn 2 as only
the new question into turn 1's thread, so it passes when the reply has the
number AND the server log says the session continued its thread (a fallback to
a new thread with the full history would know the number too). Two real
requests on the Google account whose cookies the extension pushed.

Usage (`ocweb --selftest` runs it after starting the proxy and Chrome):
    PYTHONPATH=. uv run --active python tools/api_selftest.py
    GEMINI_API_BASE=http://localhost:8100 GEMINI_PROXY_LOG=/tmp/gemini-web-proxy.log \\
        SELFTEST_MODEL=gemini-3-flash uv run --active python tools/api_selftest.py
"""

from __future__ import annotations

import os
import random
import sys
import time
import uuid

import httpx

from gemini_openai.openai_schemas import ChatMessage, session_key

BASE = os.getenv("GEMINI_API_BASE", "http://localhost:8100").rstrip("/")
KEY = os.getenv("GEMINI_API_KEY", "")
LOG = os.getenv("GEMINI_PROXY_LOG", "/tmp/gemini-web-proxy.log")
MODEL = os.getenv("SELFTEST_MODEL", "")
HEADERS = {"Authorization": f"Bearer {KEY}"} if KEY else {}


def chat(client: httpx.Client, model: str, messages: list[dict]) -> str:
    r = client.post(f"{BASE}/v1/chat/completions", headers=HEADERS,
                    json={"model": model, "messages": messages})
    if r.status_code != 200:
        raise SystemExit(f"FAIL: chat returned HTTP {r.status_code}: {r.text[:300]}")
    return (r.json()["choices"][0]["message"].get("content") or "").strip()


def log_lines_for(session: str, since: int) -> list[str]:
    try:
        with open(LOG, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()[since:]
    except OSError:
        return []
    return [ln.strip() for ln in lines if f"[api] session {session[:8]}" in ln]


def log_length() -> int:
    try:
        with open(LOG, encoding="utf-8", errors="replace") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def main() -> int:
    with httpx.Client(timeout=300) as client:
        health = client.get(f"{BASE}/health").json()
        print(f"health: backend_mode={health.get('backend_mode')} "
              f"cookies_pushed_s_ago={health.get('cookies_pushed_s_ago')}")
        if health.get("backend_mode") != "api":
            print("FAIL: the server isn't in GEMINI_BACKEND=api mode")
            return 1
        model = MODEL or client.get(f"{BASE}/v1/models", headers=HEADERS).json()["data"][0]["id"]

        number = random.randint(1000, 9999)
        # A unique system prompt: a new session key, so this run starts its own thread.
        system = {"role": "system", "content": f"ocweb selftest {uuid.uuid4()}. Follow instructions exactly."}
        turn1 = [system, {"role": "user", "content": f"Remember the number {number}. Reply only with: OK"}]
        session = session_key([ChatMessage(**m) for m in turn1])
        start = log_length()

        t0 = time.time()
        reply1 = chat(client, model, turn1)
        print(f"turn 1 ({time.time() - t0:.1f}s, {model}): {reply1[:120]!r}")
        turn2 = [*turn1, {"role": "assistant", "content": reply1},
                 {"role": "user", "content": "Which number did I ask you to remember? Reply with the number only."}]
        t0 = time.time()
        reply2 = chat(client, model, turn2)
        print(f"turn 2 ({time.time() - t0:.1f}s): {reply2[:120]!r}")

    lines = log_lines_for(session, start)
    for ln in lines:
        print(f"log: {ln}")
    remembered = str(number) in reply2
    continued = any("continued its thread" in ln for ln in lines)
    print(f"remembered {number}: {'yes' if remembered else 'NO'}")
    if lines:
        print(f"turn 2 continued turn 1's thread: {'yes' if continued else 'NO'}")
    else:
        print(f"turn 2 continued turn 1's thread: unknown (no [api] lines for it in {LOG})")
    ok = remembered and (continued or not lines)
    print("PASS" if ok and lines else ("PASS (reply only)" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
