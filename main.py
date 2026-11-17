"""Run the Gemini OpenAI-compatible API server."""

from datetime import datetime
import re
import sys

import uvicorn
from uvicorn.config import LOGGING_CONFIG

from gemini_openai import config

_TS_RE = re.compile(r"^20\d{2}-\d{2}-\d{2} ")


class _TimestampStream:
    def __init__(self, stream):
        self._stream = stream
        self._at_line_start = True

    def write(self, data: str) -> int:
        if not data:
            return 0
        out = []
        for part in data.splitlines(keepends=True):
            if self._at_line_start and part.strip() and not _TS_RE.match(part):
                out.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {part}")
            else:
                out.append(part)
            self._at_line_start = part.endswith(("\n", "\r"))
        self._stream.write("".join(out))
        return len(data)

    def flush(self) -> None:
        self._stream.flush()

    def isatty(self) -> bool:
        return getattr(self._stream, "isatty", lambda: False)()

    def fileno(self) -> int:
        return self._stream.fileno()


def install_timestamps() -> None:
    if not isinstance(sys.stdout, _TimestampStream):
        sys.stdout = _TimestampStream(sys.stdout)
    if not isinstance(sys.stderr, _TimestampStream):
        sys.stderr = _TimestampStream(sys.stderr)


def uvicorn_log_config() -> dict:
    cfg = {k: (v.copy() if isinstance(v, dict) else v) for k, v in LOGGING_CONFIG.items()}
    cfg["formatters"] = {
        k: {**v, "fmt": f"%(asctime)s | {v['fmt']}", "datefmt": "%Y-%m-%d %H:%M:%S"}
        for k, v in LOGGING_CONFIG["formatters"].items()
    }
    return cfg


def run_server() -> None:
    install_timestamps()
    uvicorn.run(
        "gemini_openai.server:app",
        host=config.HOST,
        port=config.PORT,
        log_level="info",
        log_config=uvicorn_log_config(),
    )


if __name__ == "__main__":
    run_server()