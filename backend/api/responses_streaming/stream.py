import asyncio
import json
from typing import Any, Dict, List

from ...errors import GatewayError
from ...logging_config import logger
from ...responses_schemas import ResponsesRequest, ResponsesResponse
from ...schemas import ChatCompletionRequest
from ...storage.responses import StoredResponse, response_store
from ...streaming.control import ProviderStreamControl
from ...streaming.sse import SSEDecoder, encode_sse
from .state import ResponsesStreamState


async def _watch_for_cancel(response_id: str, stream_control: ProviderStreamControl):
    while not stream_control.cancelled:
        if response_store.is_cancel_requested(response_id):
            await stream_control.cancel()
            return
        await asyncio.sleep(0.05)


def persist_response(
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


async def stream_responses(
    router_instance,
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
        chat_stream = await router_instance.iter_stream(
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
                persist_response(
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
                    persist_response(
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
                    persist_response(
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
            persist_response(
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
        persist_response(
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
        persist_response(
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
        persist_response(
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