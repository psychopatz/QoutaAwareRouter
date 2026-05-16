import time
from typing import Any, Dict, List, Optional

from ...responses_schemas import ResponsesOutputItem, ResponsesRequest, ResponsesResponse
from ...schemas import ProviderMetadata, Usage
from ...streaming.sse import encode_sse


class ResponsesStreamState:
    def __init__(self, response_id: str, request: ResponsesRequest, request_messages: List[Dict[str, Any]]):
        self.response_id = response_id
        self.created_at = int(time.time())
        self.model = request.model
        self.metadata = request.metadata
        self.request_messages = request_messages
        self.provider: Optional[ProviderMetadata] = None
        self.output_text = ""
        self.reasoning_text = ""
        self.audio = None
        self.refusal_text = ""
        self.usage = Usage()
        self.message_item_id = f"msg_{response_id}"
        self.message_started = False
        self.tool_calls: Dict[int, Dict[str, Any]] = {}
        self.tool_call_order: List[int] = []
        self.text_done_emitted = False
        self.content_part_indexes: Dict[str, int] = {}

    def set_provider(self, provider_data: Dict[str, Any]):
        self.provider = ProviderMetadata(**provider_data)

    def current_response(
        self,
        status: str = "in_progress",
        error: Optional[Dict[str, Any]] = None,
        incomplete_details: Optional[Dict[str, Any]] = None,
    ) -> ResponsesResponse:
        output = [self._message_output_item(status)]
        for tool_index in self.tool_call_order:
            output.append(self._tool_output_item(tool_index, status))

        return ResponsesResponse(
            id=self.response_id,
            created_at=self.created_at,
            status=status,
            model=self.model,
            output=output,
            output_text=self.output_text,
            provider=self.provider,
            usage=self.usage,
            metadata=self.metadata,
            error=error,
            incomplete_details=incomplete_details,
        )

    def finalize(
        self,
        status: str = "completed",
        error: Optional[Dict[str, Any]] = None,
        incomplete_details: Optional[Dict[str, Any]] = None,
    ) -> ResponsesResponse:
        return self.current_response(status=status, error=error, incomplete_details=incomplete_details)

    def conversation_messages(self) -> List[Dict[str, Any]]:
        assistant_message: Dict[str, Any] = {"role": "assistant"}
        if self.output_text:
            assistant_message["content"] = self.output_text
        if self.audio:
            assistant_message["audio"] = self.audio
        if self.refusal_text:
            assistant_message["refusal"] = self.refusal_text
        if self.reasoning_text:
            assistant_message["reasoning"] = self.reasoning_text
        if self.tool_call_order:
            assistant_message["tool_calls"] = [self._tool_call_dict(index) for index in self.tool_call_order]

        if "content" not in assistant_message and "tool_calls" not in assistant_message:
            assistant_message["content"] = ""

        return [*self.request_messages, assistant_message]

    def apply_payload(self, payload: Dict[str, Any]) -> List[bytes]:
        emitted = []

        if payload.get("usage"):
            self.usage = Usage(**payload["usage"])

        for choice in payload.get("choices", []):
            delta = choice.get("delta") or {}
            if delta.get("content") is not None:
                emitted.extend(self._apply_text_delta(delta.get("content", "")))
            if delta.get("reasoning") is not None:
                emitted.extend(self._apply_reasoning_delta(delta.get("reasoning", "")))
            if delta.get("audio") is not None:
                emitted.extend(self._apply_audio_delta(delta["audio"]))
            if delta.get("refusal") is not None:
                emitted.extend(self._apply_refusal_delta(delta.get("refusal", "")))
            if delta.get("tool_calls"):
                emitted.extend(self._apply_tool_calls(delta["tool_calls"]))

        return emitted

    def emit_done_events(self, status: str) -> List[bytes]:
        emitted = []
        if self.output_text and not self.text_done_emitted:
            self.text_done_emitted = True
            output_index, content_index = self._content_position("output_text")
            emitted.append(
                encode_sse(
                    {
                        "type": "response.output_text.done",
                        "response_id": self.response_id,
                        "item_id": self.message_item_id,
                        "output_index": output_index,
                        "content_index": content_index,
                        "text": self.output_text,
                    },
                    event="response.output_text.done",
                )
            )
            emitted.append(self._content_part_done_event("output_text", {"text": self.output_text}))

        if self.reasoning_text:
            output_index, content_index = self._content_position("reasoning")
            emitted.append(
                encode_sse(
                    {
                        "type": "response.reasoning.done",
                        "response_id": self.response_id,
                        "item_id": self.message_item_id,
                        "output_index": output_index,
                        "content_index": content_index,
                        "text": self.reasoning_text,
                    },
                    event="response.reasoning.done",
                )
            )
            emitted.append(self._content_part_done_event("reasoning", {"text": self.reasoning_text}))

        if self.audio:
            output_index, content_index = self._content_position("output_audio")
            emitted.append(
                encode_sse(
                    {
                        "type": "response.output_audio.done",
                        "response_id": self.response_id,
                        "item_id": self.message_item_id,
                        "output_index": output_index,
                        "content_index": content_index,
                        "audio": self.audio,
                    },
                    event="response.output_audio.done",
                )
            )
            emitted.append(self._content_part_done_event("output_audio", {"audio": self.audio}))

        if self.refusal_text:
            output_index, content_index = self._content_position("refusal")
            emitted.append(
                encode_sse(
                    {
                        "type": "response.refusal.done",
                        "response_id": self.response_id,
                        "item_id": self.message_item_id,
                        "output_index": output_index,
                        "content_index": content_index,
                        "text": self.refusal_text,
                    },
                    event="response.refusal.done",
                )
            )
            emitted.append(self._content_part_done_event("refusal", {"text": self.refusal_text}))

        final_response = self.finalize(status=status)
        for output_index, item in enumerate(final_response.output):
            emitted.append(
                encode_sse(
                    {
                        "type": "response.output_item.done",
                        "response_id": self.response_id,
                        "output_index": output_index,
                        "item": item.model_dump(exclude_none=True),
                    },
                    event="response.output_item.done",
                )
            )

        return emitted

    def _message_output_item(self, status: str) -> ResponsesOutputItem:
        message_content: List[Dict[str, Any]] = []
        if self.output_text:
            message_content.append({"type": "output_text", "text": self.output_text})
        if self.reasoning_text:
            message_content.append({"type": "reasoning", "text": self.reasoning_text})
        if self.audio:
            message_content.append({"type": "output_audio", "audio": self.audio})
        if self.refusal_text:
            message_content.append({"type": "refusal", "text": self.refusal_text})

        return ResponsesOutputItem(
            id=self.message_item_id,
            type="message",
            role="assistant",
            status=status,
            content=message_content,
        )

    def _tool_call_dict(self, tool_index: int) -> Dict[str, Any]:
        tool_call = self.tool_calls[tool_index]
        return {
            "id": tool_call["call_id"],
            "type": "function",
            "function": {
                "name": tool_call["name"],
                "arguments": tool_call["arguments"],
            },
        }

    def _tool_output_item(self, tool_index: int, status: str) -> ResponsesOutputItem:
        tool_call = self.tool_calls[tool_index]
        return ResponsesOutputItem(
            id=tool_call["item_id"],
            type="function_call",
            status=status,
            call_id=tool_call["call_id"],
            name=tool_call["name"],
            arguments=tool_call["arguments"],
        )

    def _ensure_message_started(self) -> List[bytes]:
        if self.message_started:
            return []

        self.message_started = True
        return [
            encode_sse(
                {
                    "type": "response.output_item.added",
                    "response_id": self.response_id,
                    "output_index": 0,
                    "item": self._message_output_item("in_progress").model_dump(exclude_none=True),
                },
                event="response.output_item.added",
            )
        ]

    def _apply_text_delta(self, text_delta: str) -> List[bytes]:
        emitted = self._ensure_message_started()
        if not text_delta:
            return emitted

        self.output_text += text_delta
        emitted.extend(self._ensure_content_part("output_text", {"text": self.output_text}))
        output_index, content_index = self._content_position("output_text")
        emitted.append(
            encode_sse(
                {
                    "type": "response.output_text.delta",
                    "response_id": self.response_id,
                    "item_id": self.message_item_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "delta": text_delta,
                },
                event="response.output_text.delta",
            )
        )
        return emitted

    def _apply_reasoning_delta(self, reasoning_delta: str) -> List[bytes]:
        emitted = self._ensure_message_started()
        if not reasoning_delta:
            return emitted

        self.reasoning_text += reasoning_delta
        emitted.extend(self._ensure_content_part("reasoning", {"text": self.reasoning_text}))
        output_index, content_index = self._content_position("reasoning")
        emitted.append(
            encode_sse(
                {
                    "type": "response.reasoning.delta",
                    "response_id": self.response_id,
                    "item_id": self.message_item_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "delta": reasoning_delta,
                },
                event="response.reasoning.delta",
            )
        )
        return emitted

    def _apply_audio_delta(self, audio_delta: Dict[str, Any]) -> List[bytes]:
        emitted = self._ensure_message_started()
        if not audio_delta:
            return emitted

        current_audio = self.audio or {}
        merged_audio = dict(current_audio)
        for key, value in audio_delta.items():
            if key in {"data", "transcript"} and isinstance(value, str) and isinstance(merged_audio.get(key), str):
                merged_audio[key] += value
            else:
                merged_audio[key] = value
        self.audio = merged_audio

        emitted.extend(self._ensure_content_part("output_audio", {"audio": self.audio}))
        output_index, content_index = self._content_position("output_audio")
        emitted.append(
            encode_sse(
                {
                    "type": "response.output_audio.delta",
                    "response_id": self.response_id,
                    "item_id": self.message_item_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "delta": audio_delta,
                },
                event="response.output_audio.delta",
            )
        )
        return emitted

    def _apply_refusal_delta(self, refusal_delta: str) -> List[bytes]:
        emitted = self._ensure_message_started()
        if not refusal_delta:
            return emitted

        self.refusal_text += refusal_delta
        emitted.extend(self._ensure_content_part("refusal", {"text": self.refusal_text}))
        output_index, content_index = self._content_position("refusal")
        emitted.append(
            encode_sse(
                {
                    "type": "response.refusal.delta",
                    "response_id": self.response_id,
                    "item_id": self.message_item_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "delta": refusal_delta,
                },
                event="response.refusal.delta",
            )
        )
        return emitted

    def _apply_tool_calls(self, tool_call_deltas: List[Dict[str, Any]]) -> List[bytes]:
        emitted = []

        for fallback_index, tool_call_delta in enumerate(tool_call_deltas):
            tool_index = tool_call_delta.get("index", fallback_index)
            existing = self.tool_calls.get(tool_index)
            if existing is None:
                call_id = tool_call_delta.get("id") or f"call_{self.response_id}_{tool_index}"
                existing = {
                    "item_id": call_id,
                    "call_id": call_id,
                    "name": None,
                    "arguments": "",
                }
                self.tool_calls[tool_index] = existing
                self.tool_call_order.append(tool_index)
                emitted.append(
                    encode_sse(
                        {
                            "type": "response.output_item.added",
                            "response_id": self.response_id,
                            "output_index": len(self.tool_call_order),
                            "item": self._tool_output_item(tool_index, "in_progress").model_dump(exclude_none=True),
                        },
                        event="response.output_item.added",
                    )
                )

            if tool_call_delta.get("id"):
                existing["call_id"] = tool_call_delta["id"]
                existing["item_id"] = tool_call_delta["id"]

            function = tool_call_delta.get("function") or {}
            if function.get("name"):
                existing["name"] = function["name"]
            if function.get("arguments"):
                existing["arguments"] += function["arguments"]
                emitted.append(
                    encode_sse(
                        {
                            "type": "response.function_call_arguments.delta",
                            "response_id": self.response_id,
                            "item_id": existing["item_id"],
                            "output_index": self.tool_call_order.index(tool_index) + 1,
                            "delta": function["arguments"],
                        },
                        event="response.function_call_arguments.delta",
                    )
                )

        return emitted

    def _content_position(self, part_type: str):
        return 0, self.content_part_indexes[part_type]

    def _content_part_payload(self, part_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"type": part_type}
        payload.update(data)
        return payload

    def _ensure_content_part(self, part_type: str, data: Dict[str, Any]) -> List[bytes]:
        if part_type in self.content_part_indexes:
            return []

        self.content_part_indexes[part_type] = len(self.content_part_indexes)
        output_index, content_index = self._content_position(part_type)
        return [
            encode_sse(
                {
                    "type": "response.content_part.added",
                    "response_id": self.response_id,
                    "item_id": self.message_item_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "part": self._content_part_payload(part_type, data),
                },
                event="response.content_part.added",
            )
        ]

    def _content_part_done_event(self, part_type: str, data: Dict[str, Any]) -> bytes:
        output_index, content_index = self._content_position(part_type)
        return encode_sse(
            {
                "type": "response.content_part.done",
                "response_id": self.response_id,
                "item_id": self.message_item_id,
                "output_index": output_index,
                "content_index": content_index,
                "part": self._content_part_payload(part_type, data),
            },
            event="response.content_part.done",
        )