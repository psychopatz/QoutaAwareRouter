import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from ..errors import GatewayError
from ..logging_config import logger
from ..responses_schemas import ResponsesRequest
from ..routing.router import Router
from ..storage.responses import response_store
from .responses_conversion import (
    _assistant_message_from_chat_response,
    chat_completion_to_responses_response,
    responses_request_to_chat_request,
)
from .responses_streaming import persist_response
from .responses_streaming import stream_responses as _stream_responses_impl

router = APIRouter()

_router_instance: Router = None


def set_router(router_instance: Router):
    global _router_instance
    _router_instance = router_instance


async def _stream_responses(
    request: ResponsesRequest,
    chat_request,
    request_messages,
    response_id: str,
):
    async for chunk in _stream_responses_impl(
        _router_instance,
        request,
        chat_request,
        request_messages,
        response_id,
    ):
        yield chunk


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
        persist_response(
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