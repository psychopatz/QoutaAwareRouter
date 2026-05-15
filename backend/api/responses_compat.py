import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from ..errors import ErrorType, GatewayError
from ..logging_config import logger
from ..responses_schemas import ResponsesRequest, ResponsesResponse, ResponsesOutputItem
from ..routing.router import Router
from ..schemas import ChatCompletionRequest, ProviderMetadata, Usage
from ..storage.responses import StoredResponse, response_store
from ..streaming.control import ProviderStreamControl
from ..streaming.sse import SSEDecoder, encode_sse

router = APIRouter()

_router_instance: Router = None


def set_router(router_instance: Router):
    global _router_instance
    _router_instance = router_instance


def _normalize_response_input_part(part: Dict[str, Any]) -> Dict[str, Any]:
    part_type = part.get("type")

    if part_type in {"input_text", "text"}:
        return {"type": "text", "text": part.get("text", "")}

    if part_type in {"input_image", "image_url"}:
        image_url = part.get("image_url") or {"url": part.get("image_url") or part.get("url")}
        return {"type": "image_url", "image_url": image_url}

    if part_type in {"input_audio", "audio"}:
        input_audio = part.get("input_audio") or part.get("audio") or {}
        return {"type": "input_audio", "input_audio": input_audio}

    raise GatewayError(
        f"Unsupported responses input part type '{part_type}'",
        type=ErrorType.UNSUPPORTED_FEATURE,
        status_code=400,
    )


def _normalize_response_message(item: Dict[str, Any]) -> Dict[str, Any]:
    content = item.get("content", "")
    if isinstance(content, list):
        normalized_content = [_normalize_response_input_part(part) for part in content]
    else:
        normalized_content = content

    message = {
        "role": item.get("role", "user"),
        "content": normalized_content,
    }

    if item.get("name"):
        message["name"] = item["name"]

    return message


def _response_format_from_request(request: ResponsesRequest) -> Optional[Dict[str, Any]]:
    if request.response_format is not None:
        return request.response_format

    if not isinstance(request.text, dict):
        return None

    text_format = request.text.get("format")
    if isinstance(text_format, dict):
        return text_format

    text_type = request.text.get("type")
    if text_type in {"text", "json_object", "json_schema"}:
        return dict(request.text)

    return None


def _normalize_request_messages(request: ResponsesRequest) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []

    if request.previous_response_id:
        previous_response = response_store.get(request.previous_response_id)
        if not previous_response:
            raise GatewayError(f"Unknown previous_response_id '{request.previous_response_id}'", status_code=404)
        messages.extend(previous_response.conversation_messages)

    if request.instructions:
        messages.append({"role": "system", "content": request.instructions})

    if isinstance(request.input, str):
        messages.append({"role": "user", "content": request.input})
    else:
        if request.input and all(item.get("type") == "message" for item in request.input if isinstance(item, dict)):
            for item in request.input:
                messages.append(_normalize_response_message(item))
        else:
            messages.append(
                {
                    "role": "user",
                    "content": [_normalize_response_input_part(part) for part in request.input],
                }
            )

    return messages


def responses_request_to_chat_request(request: ResponsesRequest) -> ChatCompletionRequest:
    messages = _normalize_request_messages(request)

    return ChatCompletionRequest(
        model=request.model,
        messages=messages,
        stream=bool(request.stream),
        tools=request.tools,
        tool_choice=request.tool_choice,
        parallel_tool_calls=request.parallel_tool_calls,
        modalities=request.modalities,
        audio=request.audio,
        response_format=_response_format_from_request(request),
        reasoning=request.reasoning,
        reasoning_effort=request.reasoning_effort,
        max_completion_tokens=request.max_output_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=request.stop,
        user=request.user,
    )


def _assistant_message_from_chat_response(chat_response) -> Dict[str, Any]:
    message = chat_response.choices[0].message
    assistant_message: Dict[str, Any] = {"role": message.role}

    if message.content is not None:
        assistant_message["content"] = message.content
    if message.tool_calls:
        assistant_message["tool_calls"] = message.tool_calls
    if message.audio:
        assistant_message["audio"] = message.audio
    if message.reasoning:
        assistant_message["reasoning"] = message.reasoning

    return assistant_message


def chat_completion_to_responses_response(chat_response, response_id: str, metadata: Optional[Dict[str, str]] = None) -> ResponsesResponse:
    choice = chat_response.choices[0]
    message = choice.message
    output: List[ResponsesOutputItem] = []

    message_content: List[Dict[str, Any]] = []
    if message.content is not None:
        message_content.append({"type": "output_text", "text": message.content})
    if message.reasoning:
        message_content.append({"type": "reasoning", "text": message.reasoning})
    if message.audio:
        message_content.append({"type": "output_audio", "audio": message.audio})

    output.append(
        ResponsesOutputItem(
            id=f"msg_{response_id}",
            type="message",
            role=message.role,
            status="completed",
            content=message_content,
        )
    )

    for tool_call in message.tool_calls or []:
        function = tool_call.get("function") or {}
        output.append(
            ResponsesOutputItem(
                id=tool_call.get("id") or f"fc_{response_id}",
                type="function_call",
                status="completed",
                call_id=tool_call.get("id"),
                name=function.get("name"),
                arguments=function.get("arguments"),
            )
        )

    return ResponsesResponse(
        id=response_id,
        created_at=chat_response.created,
        model=chat_response.model,
        output=output,
        output_text=message.content or "",
        provider=chat_response.provider,
        usage=chat_response.usage,
        metadata=metadata,
    )


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


async def _watch_for_cancel(response_id: str, stream_control: ProviderStreamControl):
    while not stream_control.cancelled:
        if response_store.is_cancel_requested(response_id):
            await stream_control.cancel()
            return
        await asyncio.sleep(0.05)


def _persist_response(
    response: ResponsesResponse,
    request_messages: List[Dict[str, Any]],
    request: ResponsesRequest,
    conversation_messages: List[Dict[str, Any]],
    stored: bool,
):
    response_store.upsert(
        StoredResponse(
            response=response,
            conversation_messages=conversation_messages,
            request=request.model_dump(exclude_none=True),
            stored=stored,
        )
    )


async def _stream_responses(
    request: ResponsesRequest,
    chat_request: ChatCompletionRequest,
    request_messages: List[Dict[str, Any]],
    response_id: str,
):
    stream_state = ResponsesStreamState(response_id, request, request_messages)
    stream_control = ProviderStreamControl()
    initial_response = stream_state.current_response()
    response_store.create_pending(
        response=initial_response,
        conversation_messages=request_messages,
        request=request.model_dump(exclude_none=True),
        stored=bool(request.store),
    )

    yield encode_sse(initial_response.model_dump(exclude_none=True), event="response.created")
    yield encode_sse(stream_state.current_response().model_dump(exclude_none=True), event="response.in_progress")

    decoder = SSEDecoder()
    cancel_watcher = asyncio.create_task(_watch_for_cancel(response_id, stream_control))

    def _on_provider_selected(provider_data: Dict[str, Any]):
        stream_state.set_provider(provider_data)
        response_store.upsert(
            StoredResponse(
                response=stream_state.current_response(),
                conversation_messages=request_messages,
                request=request.model_dump(exclude_none=True),
                stored=bool(request.store),
            )
        )

    try:
        chat_stream = await _router_instance.iter_stream(
            chat_request,
            on_provider_selected=_on_provider_selected,
            stream_control=stream_control,
        )

        async for raw_chunk in chat_stream:
            if response_store.is_cancel_requested(response_id):
                await stream_control.cancel()
                cancelled_response = stream_state.finalize(
                    status="cancelled",
                    incomplete_details={
                        "reason": "cancelled",
                        "provider_cancel_supported": stream_control.native_cancel_supported,
                    },
                )
                _persist_response(
                    cancelled_response,
                    request_messages,
                    request,
                    stream_state.conversation_messages(),
                    stored=bool(request.store),
                )
                for event_bytes in stream_state.emit_done_events("cancelled"):
                    yield event_bytes
                yield encode_sse(cancelled_response.model_dump(exclude_none=True), event="response.cancelled")
                return

            for message in decoder.feed(raw_chunk):
                if message.data == "[DONE]":
                    completed_response = stream_state.finalize(status="completed")
                    _persist_response(
                        completed_response,
                        request_messages,
                        request,
                        stream_state.conversation_messages(),
                        stored=bool(request.store),
                    )
                    for event_bytes in stream_state.emit_done_events("completed"):
                        yield event_bytes
                    yield encode_sse(completed_response.model_dump(exclude_none=True), event="response.completed")
                    return

                payload = json.loads(message.data)
                if payload.get("error"):
                    failed_response = stream_state.finalize(status="failed", error=payload["error"])
                    _persist_response(
                        failed_response,
                        request_messages,
                        request,
                        request_messages,
                        stored=bool(request.store),
                    )
                    yield encode_sse(failed_response.model_dump(exclude_none=True), event="response.failed")
                    return

                for event_bytes in stream_state.apply_payload(payload):
                    yield event_bytes

        if response_store.is_cancel_requested(response_id) or stream_control.cancelled:
            cancelled_response = stream_state.finalize(
                status="cancelled",
                incomplete_details={
                    "reason": "cancelled",
                    "provider_cancel_supported": stream_control.native_cancel_supported,
                },
            )
            _persist_response(
                cancelled_response,
                request_messages,
                request,
                stream_state.conversation_messages(),
                stored=bool(request.store),
            )
            for event_bytes in stream_state.emit_done_events("cancelled"):
                yield event_bytes
            yield encode_sse(cancelled_response.model_dump(exclude_none=True), event="response.cancelled")
            return

        for message in decoder.finalize():
            if message.data != "[DONE]":
                payload = json.loads(message.data)
                for event_bytes in stream_state.apply_payload(payload):
                    yield event_bytes

        completed_response = stream_state.finalize(status="completed")
        _persist_response(
            completed_response,
            request_messages,
            request,
            stream_state.conversation_messages(),
            stored=bool(request.store),
        )
        for event_bytes in stream_state.emit_done_events("completed"):
            yield event_bytes
        yield encode_sse(completed_response.model_dump(exclude_none=True), event="response.completed")
    except GatewayError as e:
        failed_response = stream_state.finalize(status="failed", error=e.to_dict()["error"])
        _persist_response(
            failed_response,
            request_messages,
            request,
            request_messages,
            stored=bool(request.store),
        )
        yield encode_sse(failed_response.model_dump(exclude_none=True), event="response.failed")
    except Exception as e:
        logger.exception(f"Unhandled error in responses stream: {e}")
        failed_response = stream_state.finalize(
            status="failed",
            error={"type": "internal_error", "message": str(e)},
        )
        _persist_response(
            failed_response,
            request_messages,
            request,
            request_messages,
            stored=bool(request.store),
        )
        yield encode_sse(failed_response.model_dump(exclude_none=True), event="response.failed")
    finally:
        cancel_watcher.cancel()
        response_store.clear_cancel(response_id)


@router.post("/responses")
async def responses(request: ResponsesRequest):
    if not _router_instance:
        raise HTTPException(status_code=503, detail="Router not initialized")

    try:
        chat_request = responses_request_to_chat_request(request)
        request_messages = [message.model_dump(exclude_none=True) for message in chat_request.messages]

        if request.stream:
            response_id = f"resp_{uuid.uuid4().hex}"
            return StreamingResponse(
                _stream_responses(request, chat_request, request_messages, response_id),
                media_type="text/event-stream",
            )

        chat_response = await _router_instance.route(chat_request)
        response_id = f"resp_{uuid.uuid4().hex}"
        response = chat_completion_to_responses_response(
            chat_response,
            response_id=response_id,
            metadata=request.metadata,
        )
        _persist_response(
            response,
            request_messages,
            request,
            [*request_messages, _assistant_message_from_chat_response(chat_response)],
            stored=bool(request.store),
        )
        return response
    except GatewayError as e:
        return JSONResponse(status_code=e.status_code, content=e.to_dict())
    except Exception as e:
        logger.exception(f"Unhandled error in responses endpoint: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "internal_error"}},
        )


@router.get("/responses/{response_id}")
async def get_response(response_id: str):
    stored_response = response_store.get(response_id)
    if not stored_response:
        return JSONResponse(
            status_code=404,
            content={"error": {"message": f"Response '{response_id}' not found", "type": "not_found"}},
        )
    return stored_response.response


@router.post("/responses/{response_id}/cancel")
async def cancel_response(response_id: str):
    stored_response = response_store.request_cancel(response_id)
    if not stored_response:
        return JSONResponse(
            status_code=404,
            content={"error": {"message": f"Response '{response_id}' not found", "type": "not_found"}},
        )
    return stored_response.response