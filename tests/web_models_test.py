"""Argon asked for the web app's way (api mode).

Argon stalls over StreamGenerate as the library selects it (gemini-webapi #374) and
answers when the model header has the web app's shape; see gemini_openai/web_models.py.
No network: the transport hook is checked against a recording stand-in.
"""
import asyncio
import json

import pytest
from curl_cffi.requests import AsyncSession

from gemini_openai import config, server, web_models
from gemini_openai import gemini_pool as gp

LIB_ARGON = '[1,null,null,null,"argon",null,null,0,[4,5,6,8],null,null,2,null,null,8,1,"SESSION"]'
LIB_HYDROGEN = '[1,null,null,null,"56fdd199312815e2",null,null,0,[4,5,6,8],null,null,2,null,null,1,1,"SESSION"]'


def freq(inner_len=81):
    inner = [None] * inner_len
    inner[79], inner[80] = 8, 1
    return json.dumps([None, json.dumps(inner)])


@pytest.mark.parametrize("name", ["gemini-argon", "argon", "gemini-4-argon", "gemini-carbon",
                                  "models/gemini-argon", " Gemini-Argon "])
def test_argon_names_resolve_to_argon(name):
    assert web_models.lookup(name) is web_models.ARGON


@pytest.mark.parametrize("name", ["gemini-4-hydrogen", "gemini-flash-lite", "", None, {"a": 1}])
def test_other_names_are_not_web_models(name):
    assert web_models.lookup(name) is None


def test_the_library_gets_the_slug_where_it_expects_the_id():
    model = web_models.library_model(web_models.ARGON)
    header = json.loads(model["model_header"][web_models.HEADER_KEY])
    assert model["model_name"] == "gemini-argon"
    assert header[4] == "argon" and header[-1] == 8  # the library appends level + session


def test_an_argon_request_is_reshaped_to_the_web_apps_form():
    headers, data = web_models.reshape(
        {web_models.HEADER_KEY: LIB_ARGON, "x-other": "kept"}, {"f.req": freq(), "at": "kept"},
        language="es", build="boq_gemini-web-uiserver_20261116.16_p0", level=4)
    web = json.loads(headers[web_models.HEADER_KEY])
    assert len(web) == 25
    assert (web[0], web[4], web[6], web[7]) == (1, "argon", "es", 0)
    assert web[8] == [4, 33, 22, 5, 6, 8, 16, 19, 20, 38]
    assert (web[11], web[14], web[15], web[16]) == (2, 8, 4, "SESSION")
    assert web[17] == ["boq_gemini-web-uiserver_20261116.16_p0"] and web[24] == "argon"
    assert headers["x-other"] == "kept" and data["at"] == "kept"
    inner = json.loads(json.loads(data["f.req"])[1])
    assert (inner[79], inner[80]) == (8, 4)


def test_other_models_requests_are_left_alone():
    assert web_models.reshape({web_models.HEADER_KEY: LIB_HYDROGEN}, {}, "es", None, 4) is None
    assert web_models.reshape({}, {}, "es", None, 4) is None
    assert web_models.reshape({web_models.HEADER_KEY: "not json"}, {}, "es", None, 4) is None


@pytest.mark.parametrize("value, level", [(None, 4), ("1", 1), ("2", 2), ("4", 4), ("3", 4), ("x", 4)])
def test_the_reasoning_level_is_the_webs(monkeypatch, value, level):
    if value is None:
        monkeypatch.delenv("GEMINI_ARGON_THINKING", raising=False)
    else:
        monkeypatch.setenv("GEMINI_ARGON_THINKING", value)
    assert web_models.thinking_level() == level


def test_the_transport_hook_reshapes_only_web_model_requests(monkeypatch):
    sent = []

    def recorder(self, method, url, *args, **kwargs):
        sent.append((url, kwargs))
        return "response"

    monkeypatch.setattr(AsyncSession, "stream", recorder)
    monkeypatch.setattr(web_models, "_installed", False)
    monkeypatch.setitem(web_models._session, "language", "es")
    monkeypatch.setitem(web_models._session, "build", "BUILD")
    web_models.install()

    gen = "https://gemini.google.com/_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate"
    AsyncSession.stream(None, "POST", gen, headers={web_models.HEADER_KEY: LIB_ARGON}, data={"f.req": freq()})
    AsyncSession.stream(None, "POST", gen, headers={web_models.HEADER_KEY: LIB_HYDROGEN}, data={})
    AsyncSession.stream(None, "POST", "https://gemini.google.com/other",
                        headers={web_models.HEADER_KEY: LIB_ARGON}, data={})
    reshaped = json.loads(sent[0][1]["headers"][web_models.HEADER_KEY])
    assert reshaped[24] == "argon" and reshaped[17] == ["BUILD"]
    assert sent[1][1]["headers"][web_models.HEADER_KEY] == LIB_HYDROGEN
    assert sent[2][1]["headers"][web_models.HEADER_KEY] == LIB_ARGON


def test_api_mode_asks_for_argon_the_web_apps_way(monkeypatch):
    monkeypatch.setattr(config, "BACKEND", "api")
    out = gp._for_account(object(), {"model": "gemini-argon", "temporary": True})
    assert out["model"]["model_name"] == "gemini-argon" and out["temporary"] is True


def test_other_modes_leave_argon_names_to_the_registry(monkeypatch):
    monkeypatch.setattr(config, "BACKEND", "webapi")
    out = gp._for_account(object(), {"model": "gemini-argon"})
    assert out["model"] == "gemini-argon"


class RegistryModel:
    def __init__(self, model_id, name, description, display):
        self.model_id, self.model_name = model_id, name
        self.description, self.display_name, self.is_available = description, display, True


def test_models_list_argon_in_place_of_its_stalling_registry_entry(monkeypatch):
    registry = [RegistryModel("56fdd199312815e2", "gemini-4-hydrogen", "4 Hydrogen", "Ayuda en tu día a día"),
                RegistryModel("95a4c0abcba24364", "gemini-carbon", "4 Carbon", "El trabajo más complejo")]
    client = type("C", (), {"list_models": lambda self: registry})()

    class Manager:
        async def get(self):
            return client

    monkeypatch.setattr(config, "BACKEND", "api")
    monkeypatch.setattr(config, "DEFAULT_MODEL", "gemini-4-hydrogen")
    monkeypatch.setattr(server, "webapi_manager", Manager())
    data = asyncio.run(server.list_models())["data"]
    assert [m["id"] for m in data] == ["gemini-4-hydrogen", "gemini-argon"]
    assert data[1]["description"].startswith("4 Argon (Código y análisis) — experimental")
