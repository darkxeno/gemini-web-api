"""Models that only answer when asked the way the web app asks (api mode).

Since Google's 2026-11-12 web build, Argon ("4 Argon", "Código y análisis"; registry
id 95a4c0abcba24364, which gemini_webapi lists as `gemini-carbon`) stalls over
StreamGenerate when selected the library's way -- the conversation is created, then
only a `{"37":[n]}` heartbeat a minute (gemini-webapi issue #374). Sent with the model
header in the web app's own shape it SOMETIMES answers: 1 of 6 tries on 2026-11-17 ("OK"
in 27s); the others stalled or got a generic refusal ("Soy un modelo de lenguaje, así
que no tengo esa capacidad") -- which a client receives as the reply. Experimental: the
web app itself sends messages over processSession with an anti-bot token this lacks.

    library: [1,null,null,null,"<hex id>",null,null,0,[4,5,6,8],null,null,2,null,null,8,<thinking>,<session>]
    web:     [1,null,null,null,"argon",null,"<lang>",0,[4,33,22,5,6,8,16,19,20,38],null,null,2,
              null,null,8,<level>,<session>,["<web build>"],null,null,null,null,null,null,"argon"]

with the reasoning level the web's own (1 low, 4 medium, 2 high) in the header and
in f.req's inner[80]. Which of those differences is the decisive one is untested:
the slug alone was never enough (#374), nor level 4 alone.

The library builds and sends the request itself, so the reshaping happens at the
transport: a request for one of these models carries its slug where the library puts
the model id, and `install()`'s hook rewrites it on its way out.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

HEADER_KEY = "x-goog-ext-525001261-jspb"
WEB_CAPABILITIES = [4, 33, 22, 5, 6, 8, 16, 19, 20, 38]


@dataclass(frozen=True)
class WebModel:
    name: str
    slug: str
    number: int
    description: str
    registry_id: str  # the id gemini_webapi's registry lists it under


ARGON = WebModel("gemini-argon", "argon", 8,
                 "4 Argon (Código y análisis) — experimental: a veces no responde",
                 "95a4c0abcba24364")
MODELS = {m.name: m for m in (ARGON,)}
ALIASES = {"argon": ARGON, "gemini-4-argon": ARGON, "gemini-carbon": ARGON, "carbon": ARGON}
SLUGS = {m.slug: m for m in MODELS.values()}

# The session's language and web build, set from the client after init.
_session = {"language": "en", "build": None}


def lookup(name: object) -> WebModel | None:
    if not isinstance(name, str):
        return None
    key = name.strip().lower().split("/")[-1]
    return MODELS.get(key) or ALIASES.get(key)


def thinking_level() -> int:
    """The web's reasoning level for these models: 1 low, 4 medium (default), 2 high."""
    try:
        level = int(os.getenv("GEMINI_ARGON_THINKING", "4"))
    except ValueError:
        return 4
    return level if level in (1, 2, 4) else 4


def library_model(model: WebModel) -> dict:
    """The library's custom-model form: the slug where it expects the id, so the
    transport hook knows which requests to reshape."""
    return {
        "model_name": model.name,
        "model_header": {
            HEADER_KEY: f'[1,null,null,null,"{model.slug}",null,null,0,[4,5,6,8],null,null,2,null,null,{model.number}]',
            "x-goog-ext-73010989-jspb": "[0]",
            "x-goog-ext-73010990-jspb": "[0,0,0]",
        },
    }


def set_session(language: str | None, build: str | None) -> None:
    _session["language"] = language or "en"
    _session["build"] = build


def reshape(headers: dict, data: dict, language: str, build: str | None,
            level: int) -> tuple[dict, dict] | None:
    """The web-shaped headers and form for a request the library made for a web model,
    or None when the request isn't one (its header has no known slug)."""
    raw = headers.get(HEADER_KEY)
    if not raw:
        return None
    try:
        lib = json.loads(raw)
    except ValueError:
        return None
    model = SLUGS.get(lib[4]) if len(lib) > 4 else None
    if model is None:
        return None
    session = lib[-1] if isinstance(lib[-1], str) else ""
    web: list = [None] * 25
    web[0], web[4], web[6], web[7] = 1, model.slug, language, 0
    web[8] = WEB_CAPABILITIES
    web[11], web[14], web[15], web[16] = 2, model.number, level, session
    web[17] = [build] if build else None
    web[24] = model.slug
    new_headers = {**headers, HEADER_KEY: json.dumps(web, separators=(",", ":"))}
    new_data = dict(data)
    if data.get("f.req"):
        outer = json.loads(data["f.req"])
        inner = json.loads(outer[1])
        inner[79], inner[80] = model.number, level
        outer[1] = json.dumps(inner, separators=(",", ":"))
        new_data["f.req"] = json.dumps(outer, separators=(",", ":"))
    return new_headers, new_data


_installed = False


def install() -> None:
    """Reshape web-model requests on their way out (once per process)."""
    global _installed
    if _installed:
        return
    from curl_cffi.requests import AsyncSession

    original = AsyncSession.stream

    def stream(self, method, url, *args, **kwargs):
        if "StreamGenerate" in str(url):
            shaped = reshape(kwargs.get("headers") or {}, kwargs.get("data") or {},
                             _session["language"], _session["build"], thinking_level())
            if shaped is not None:
                kwargs = {**kwargs, "headers": shaped[0], "data": shaped[1]}
        return original(self, method, url, *args, **kwargs)

    AsyncSession.stream = stream
    _installed = True
