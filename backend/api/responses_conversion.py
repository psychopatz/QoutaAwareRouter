from typing import Any, Dict, List, Optional

from ..errors import ErrorType, GatewayError
from ..responses_schemas import ResponsesOutputItem, ResponsesRequest, ResponsesResponse
from ..schemas import ChatCompletionRequest
from ..storage.responses import response_store


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
