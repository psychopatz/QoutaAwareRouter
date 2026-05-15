import json
from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class SSEMessage:
    event: Optional[str]
    data: str


class SSEDecoder:
    def __init__(self):
        self._buffer = ""

    def feed(self, chunk: bytes) -> List[SSEMessage]:
        self._buffer += chunk.decode("utf-8", errors="ignore")
        messages = []

        while "\n\n" in self._buffer:
            raw_message, self._buffer = self._buffer.split("\n\n", 1)
            parsed = self._parse_message(raw_message)
            if parsed is not None:
                messages.append(parsed)

        return messages

    def finalize(self) -> List[SSEMessage]:
        if not self._buffer.strip():
            return []

        raw_message = self._buffer
        self._buffer = ""
        parsed = self._parse_message(raw_message)
        return [parsed] if parsed is not None else []

    def _parse_message(self, raw_message: str) -> Optional[SSEMessage]:
        event = None
        data_lines = []

        for line in raw_message.splitlines():
            if not line or line.startswith(":"):
                continue
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())

        if not data_lines:
            return None

        return SSEMessage(event=event, data="\n".join(data_lines))


def encode_sse(data: Any, event: Optional[str] = None) -> bytes:
    payload = data if isinstance(data, str) else json.dumps(data)
    lines = []

    if event:
        lines.append(f"event: {event}")

    for line in payload.splitlines() or [""]:
        lines.append(f"data: {line}")

    return ("\n".join(lines) + "\n\n").encode()