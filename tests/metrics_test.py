"""Per-request metrics (gemini_openai/metrics.py) and the tools prompt they led to.

The request log must say where the time went and what happened -- never what was said.
"""
import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from gemini_openai import config, metrics, server, tools

SECRET = "the user's secret prompt text"
TOOLS = [{"type": "function", "function": {
    "name": "bash", "description": "Run a shell command",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                   "required": ["command"]}}}]


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


# --------------------------------------------------------------------------- #
# The tools prompt
# --------------------------------------------------------------------------- #
def test_tools_are_offered_as_actions_with_no_built_in_tools(monkeypatch):
    monkeypatch.delenv("GEMINI_TOOLS_PROMPT", raising=False)
    prompt = tools.build_tools_prompt(TOOLS)
    assert prompt.startswith("# ACTIONS")
    assert "do NOT use Google Search" in prompt and "- bash" in prompt


def test_the_legacy_tools_prompt_is_still_available(monkeypatch):
    monkeypatch.setenv("GEMINI_TOOLS_PROMPT", "legacy")
    assert tools.build_tools_prompt(TOOLS).startswith("# TOOLS")


# --------------------------------------------------------------------------- #
# metrics.py
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text, kind", [
    ("Soy un modelo de lenguaje, así que no tengo esa capacidad.", "generic_refusal"),
    ("You have exceeded the maximum number of tool and search calls (0 remaining)", "tool_budget"),
    ("I'm a language model and don't have the capacity to help with that.", "generic_refusal"),
    ("Here are the files: a.py, b.py", None),
    (None, None),
])
def test_refusals_are_recognised(text, kind):
    assert metrics.classify_refusal(text) == kind


def test_a_request_is_written_once_with_its_total_time(tmp_path, monkeypatch):
    log = tmp_path / "requests.jsonl"
    monkeypatch.setattr(metrics, "LOG_PATH", str(log))
    m = metrics.start("api", False, "gemini-4-hydrogen")
    metrics.note(model="gemini-4-hydrogen", gemini_s=1.5)
    metrics.finish(m)
    metrics.finish(m)  # a second finish is a no-op
    [rec] = records(log)
    assert rec["model"] == "gemini-4-hydrogen" and rec["gemini_s"] == 1.5
    assert rec["total_s"] is not None and not any(k.startswith("_") for k in rec)


def test_the_log_can_be_turned_off(tmp_path, monkeypatch):
    monkeypatch.setattr(metrics, "LOG_PATH", "off")
    metrics.finish(metrics.start("api", False, None))
    assert not list(tmp_path.iterdir())


def test_cookies_applied_while_a_request_runs_are_counted(tmp_path):
    m = metrics.start("api", False, None)
    metrics.cookies_applied()
    metrics.cookies_applied()
    metrics.finish(m)
    after = metrics.start("api", False, None)
    metrics.cookies_applied()
    assert m.cookie_updates == 2 and after.cookie_updates == 1
    metrics.finish(after)


def test_the_clients_tool_run_time_is_measured_between_turns(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(metrics.time, "time", lambda: clock[0])
    asks = metrics.start("api", False, None)
    metrics.session_turn(asks, "session-1", "user")
    asks.tool_calls = ["bash"]
    metrics.finish(asks)

    clock[0] += 7.5  # opencode runs bash, then comes back with its result
    result = metrics.start("api", False, None)
    metrics.session_turn(result, "session-1", "tool")
    assert result.client_gap_s == 7.5 and result.gap_tools == ["bash"]
    metrics.finish(result)

    plain = metrics.start("api", False, None)  # a later user turn: no tools pending
    metrics.session_turn(plain, "session-1", "user")
    assert plain.client_gap_s is None
    metrics.finish(plain)


# --------------------------------------------------------------------------- #
# End to end through /v1/chat/completions
# --------------------------------------------------------------------------- #
class Output:
    def __init__(self, text):
        self.text, self.images, self.videos = text, [], []


class FakeManager:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    async def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        await asyncio.sleep(0.01)
        return Output(self.reply)


def post(reply, monkeypatch, tmp_path, **body):
    log = tmp_path / "requests.jsonl"
    monkeypatch.setattr(metrics, "LOG_PATH", str(log))
    monkeypatch.setattr(config, "BACKEND", "api")
    monkeypatch.setattr(server, "pick_manager", lambda has_files=False: FakeManager(reply))
    r = TestClient(server.app).post("/v1/chat/completions", json={
        "model": "gemini-4-hydrogen",
        "messages": [{"role": "system", "content": "agent"}, {"role": "user", "content": SECRET}],
        **body})
    return r, records(log)


def test_a_tool_call_turn_is_recorded_without_its_content(monkeypatch, tmp_path):
    reply = '```json\n{"tool_calls": [{"name": "bash", "arguments": {"command": "ls"}}]}\n```'
    r, [rec] = post(reply, monkeypatch, tmp_path, tools=TOOLS)
    assert r.json()["choices"][0]["finish_reason"] == "tool_calls"
    assert rec["finish_reason"] == "tool_calls" and rec["tool_calls"] == ["bash"]
    assert rec["tool_parse"] == "parsed" and rec["tools"] == 1 and rec["messages"] == 2
    assert rec["prompt_chars"] > len(SECRET) and rec["session"]
    assert SECRET not in json.dumps(rec) and '"command"' not in json.dumps(rec)


def test_a_refusal_is_flagged(monkeypatch, tmp_path):
    _, [rec] = post("Soy un modelo de lenguaje, así que no tengo esa capacidad.",
                    monkeypatch, tmp_path, tools=TOOLS)
    assert rec["refusal"] == "generic_refusal" and rec["tool_parse"] == "no_tool_call"


def test_a_failed_request_is_recorded_with_its_error(monkeypatch, tmp_path):
    class Broken:
        async def generate(self, prompt, **kwargs):
            raise RuntimeError("upstream down")

    log = tmp_path / "requests.jsonl"
    monkeypatch.setattr(metrics, "LOG_PATH", str(log))
    monkeypatch.setattr(server, "pick_manager", lambda has_files=False: Broken())
    r = TestClient(server.app).post("/v1/chat/completions", json={
        "model": "gemini-4-hydrogen", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 502
    [rec] = records(log)
    assert rec["error"] == "RuntimeError: upstream down"
