"""OpenAI-compatible request/response schemas and message conversion helpers."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import time
import uuid
from typing import Any

import httpx
from pydantic import BaseModel


def _safe_json(s: Any) -> Any:
    if isinstance(s, (dict, list)):
        return s
    try:
        return json.loads(s)
    except Exception:  # noqa: BLE001
        return {}

# --------------------------------------------------------------------------- #
# Request schemas (subset of OpenAI chat.completions we support)
# --------------------------------------------------------------------------- #
class ChatMessage(BaseModel):
    role: str
    # content is either a plain string or a list of parts (OpenAI vision format)
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    # tool-calling round-trip fields (assistant emits tool_calls; tool role replies)
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    # tool / function calling (emulated via prompt engineering — see tools.py)
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None  # "auto" | "none" | "required" | {"type":"function",...}
    # accepted but ignored (no upstream equivalent); present for SDK compatibility
    top_p: float | None = None
    n: int | None = None
    stop: Any | None = None
    user: str | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "model": "gemini-3-flash",
                "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
                "stream": False,
            }
        }
    }


# --------------------------------------------------------------------------- #
# Message flattening
# --------------------------------------------------------------------------- #
def _decode_image_part(url: str) -> bytes | None:
    """Turn an OpenAI image_url into raw bytes (data: URL or http(s) URL)."""
    if url.startswith("data:"):
        try:
            _, b64 = url.split(",", 1)
            return base64.b64decode(b64)
        except (ValueError, binascii.Error):
            return None
    if url.startswith(("http://", "https://")):
        try:
            r = httpx.get(url, timeout=30, follow_redirects=True)
            r.raise_for_status()
            return r.content
        except Exception:  # noqa: BLE001
            return None
    return None


def _compact_system(txt: str) -> str:
    if "# TOOLS" in txt:
        idx = txt.index("# TOOLS")
        head = txt[:idx].strip()
        return (head[:1200] + "\n\n" + txt[idx:]) if len(head) > 1200 else txt
    return txt[:2000]

def _render_turns(messages: list[ChatMessage]) -> tuple[list[tuple[str, str]], list[bytes]]:
    """Each message as (role, text), tool calls and results rendered inline, plus image bytes."""
    files: list[bytes] = []
    turns: list[tuple[str, str]] = []  # (role, text)

    for m in messages:
        text_parts: list[str] = []
        if isinstance(m.content, str):
            text_parts.append(m.content)
        elif isinstance(m.content, list):
            for part in m.content:
                ptype = part.get("type")
                if ptype == "text":
                    text_parts.append(part.get("text", ""))
                elif ptype == "image_url":
                    iu = part.get("image_url", {})
                    url = iu.get("url") if isinstance(iu, dict) else iu
                    if url:
                        data = _decode_image_part(url)
                        if data:
                            files.append(data)
        text = "\n".join(tp for tp in text_parts if tp)

        # Round-trip tool calls / results back into the transcript so the model
        # sees its own prior tool invocations and their results.
        if m.role == "assistant" and m.tool_calls:
            rendered = [
                {
                    "name": (tc.get("function") or {}).get("name"),
                    "arguments": _safe_json((tc.get("function") or {}).get("arguments", "{}")),
                }
                for tc in m.tool_calls
            ]
            tc_text = json.dumps({"tool_calls": rendered}, ensure_ascii=False)
            text = f"{text}\n{tc_text}".strip() if text else tc_text
        elif m.role == "tool":
            name = m.name or "tool"
            text = f"Tool Result ({name}):\n{text}"

        turns.append((m.role, text))
    return turns, files


def session_key(messages: list[ChatMessage]) -> str:
    """A stable id for the conversation these messages belong to.

    OpenAI clients resend the whole history every call and carry no conversation
    id, so the key is the system prompt's start plus the first user message: the
    same on every turn of one conversation. Two conversations that open with the
    same system prompt AND the same first message share a key.
    """
    turns, _ = _render_turns(messages)
    system = "\n\n".join(t for r, t in turns if r == "system" and t.strip())
    first_user = next((t for r, t in turns if r == "user"), "")
    return hashlib.sha256(f"{system[:500]}|{first_user}".encode()).hexdigest()[:16]


def continuation(messages: list[ChatMessage]) -> str | None:
    """What is new since the model's last reply, for a tab that already holds the conversation.

    Every message after the last assistant one: the user's next message, or ALL
    the tool results of a parallel tool call, not just the last. None when the
    model hasn't replied yet (the first turn), which needs the full prompt.
    """
    turns, _ = _render_turns(messages)
    last_assistant = max((i for i, (r, _) in enumerate(turns) if r == "assistant"), default=None)
    if last_assistant is None:
        return None
    new = [t.strip() for r, t in turns[last_assistant + 1:] if r != "system" and t.strip()]
    return "\n\n".join(new) or None


def flatten_messages(messages: list[ChatMessage]) -> tuple[str, list[bytes]]:
    """Collapse an OpenAI message list into (prompt, image_files).

    OpenAI clients send the full conversation each call, so we render a plain
    transcript. A single user turn is sent as-is; multi-turn history is labeled.
    Image parts are extracted as bytes to pass to gemini_webapi's `files=`.
    """
    turns, files = _render_turns(messages)
    non_empty = [(r, t) for r, t in turns if t.strip() or r == "user"]

    # Fast path: a single user message (optionally with a system message).
    if len([t for t in non_empty if t[0] not in ("system",)]) == 1:
        sys_txt = _compact_system("\n\n".join(t for r, t in turns if r == "system" and t.strip()))
        usr_txt = "\n\n".join(t for r, t in turns if r not in ("system",) and t.strip())
        prompt = f"{sys_txt}\n\n{usr_txt}".strip() if sys_txt else usr_txt
        return prompt, files

    # Multi-turn: render a labeled transcript ending with an Assistant cue.
    label = {"system": "System", "user": "User", "assistant": "Assistant", "tool": "Tool"}
    lines: list[str] = []
    for r, t in turns:
        if not t.strip():
            continue
        lines.append(f"{label.get(r, r.capitalize())}: {t}")
    lines.append("Assistant:")
    return "\n\n".join(lines), files


# --------------------------------------------------------------------------- #
# Response builders
# --------------------------------------------------------------------------- #
def _rid() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def completion_response(
    model: str,
    content: str | None,
    prompt_tokens: int,
    completion_tokens: int,
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict:
    message: dict[str, Any] = {"role": "assistant"}
    if tool_calls:
        message["content"] = None
        message["tool_calls"] = tool_calls
        finish = "tool_calls"
    else:
        message["content"] = content
        finish = "stop"
    return {
        "id": _rid(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def chunk(model: str, rid: str, created: int, delta: dict, finish: str | None = None) -> dict:
    return {
        "id": rid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
