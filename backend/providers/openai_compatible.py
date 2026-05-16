import json
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, ConfigDict

from ..errors import ErrorType, GatewayError
from ..schemas import ChatCompletionRequest, ChatMessage


class ProviderCapabilities(BaseModel):
	model_config = ConfigDict(extra="forbid")

	supports_tools: bool = False
	supports_tool_choice: bool = False
	supports_parallel_tool_calls: bool = False
	supports_vision_input: bool = False
	supports_audio_input: bool = False
	supports_audio_output: bool = False
	supports_reasoning: bool = False
	supports_response_format: bool = False


def capability_field_names() -> List[str]:
	return list(ProviderCapabilities.model_fields.keys())


def combine_capabilities(
	base: ProviderCapabilities,
	override: Optional[ProviderCapabilities] = None,
	*,
	mode: str = "narrow",
) -> ProviderCapabilities:
	if override is None:
		return base.model_copy()

	combined: Dict[str, bool] = {}
	for field_name in capability_field_names():
		base_value = getattr(base, field_name)
		override_value = getattr(override, field_name)
		if mode == "narrow":
			combined[field_name] = base_value and override_value
		else:
			combined[field_name] = override_value

	return ProviderCapabilities(**combined)


def infer_capabilities_from_model_metadata(
	*,
	supported_parameters: Optional[List[str]] = None,
	input_modalities: Optional[List[str]] = None,
	output_modalities: Optional[List[str]] = None,
) -> ProviderCapabilities:
	parameters = set(supported_parameters or [])
	inputs = set(input_modalities or [])
	outputs = set(output_modalities or [])

	return ProviderCapabilities(
		supports_tools="tools" in parameters,
		supports_tool_choice="tool_choice" in parameters,
		supports_parallel_tool_calls="parallel_tool_calls" in parameters,
		supports_vision_input="image" in inputs,
		supports_audio_input="audio" in inputs,
		supports_audio_output="audio" in outputs,
		supports_reasoning=bool({"reasoning", "include_reasoning", "reasoning_effort"} & parameters),
		supports_response_format=bool({"response_format", "structured_outputs", "structured_output"} & parameters),
	)


def build_openai_chat_payload(
	request: ChatCompletionRequest,
	*,
	model: Optional[str] = None,
	stream: Optional[bool] = None,
) -> Dict[str, Any]:
	payload = request.model_dump(exclude_none=True, exclude_unset=True)
	payload["model"] = model or request.model

	if stream is not None:
		payload["stream"] = stream

	if request.max_completion_tokens is not None and "max_tokens" not in payload:
		payload["max_tokens"] = request.max_completion_tokens

	return payload


async def send_streaming_json_request(
	client: httpx.AsyncClient,
	*,
	url: str,
	headers: Dict[str, str],
	json_body: Dict[str, Any],
) -> httpx.Response:
	request = client.build_request(
		"POST",
		url,
		headers=headers,
		json=json_body,
	)
	return await client.send(request, stream=True)


def request_uses_tools(request: ChatCompletionRequest) -> bool:
	if request.tools:
		return True

	for message in request.messages:
		if message.role == "tool" or message.tool_calls or message.tool_call_id:
			return True

	return False


def request_uses_vision(request: ChatCompletionRequest) -> bool:
	for message in request.messages:
		for part in _iter_content_parts(message):
			if part.get("type") == "image_url":
				return True
	return False


def request_uses_audio_input(request: ChatCompletionRequest) -> bool:
	for message in request.messages:
		if message.audio:
			return True

		for part in _iter_content_parts(message):
			if part.get("type") in {"input_audio", "audio", "audio_url"}:
				return True
	return False


def request_uses_audio_output(request: ChatCompletionRequest) -> bool:
	modalities = request.modalities or []
	return "audio" in modalities or request.audio is not None


def request_uses_reasoning(request: ChatCompletionRequest) -> bool:
	return extract_reasoning_mode(request) is not None


def request_uses_response_format(request: ChatCompletionRequest) -> bool:
	return request.response_format is not None


def has_forced_tool_choice(request: ChatCompletionRequest) -> bool:
	tool_choice = request.tool_choice
	return tool_choice not in (None, "auto", "none")


def unsupported_request_features(
	request: ChatCompletionRequest,
	capabilities: ProviderCapabilities,
) -> List[str]:
	unsupported: List[str] = []

	if request_uses_tools(request) and not capabilities.supports_tools:
		unsupported.append("tool calling")

	if has_forced_tool_choice(request) and not capabilities.supports_tool_choice:
		unsupported.append("forced tool choice")

	if request.parallel_tool_calls is not None and not capabilities.supports_parallel_tool_calls:
		unsupported.append("parallel tool calls")

	if request_uses_vision(request) and not capabilities.supports_vision_input:
		unsupported.append("vision input")

	if request_uses_audio_input(request) and not capabilities.supports_audio_input:
		unsupported.append("audio input")

	if request_uses_audio_output(request) and not capabilities.supports_audio_output:
		unsupported.append("audio output")

	if request_uses_reasoning(request) and not capabilities.supports_reasoning:
		unsupported.append("reasoning")

	if request_uses_response_format(request) and not capabilities.supports_response_format:
		unsupported.append("structured response formats")

	return unsupported


def unsupported_feature_error(provider_label: str, unsupported: List[str]) -> GatewayError:
	return GatewayError(
		f"Provider '{provider_label}' does not support: {', '.join(unsupported)}",
		type=ErrorType.UNSUPPORTED_FEATURE,
		status_code=400,
	)


def ensure_supported_request(
	request: ChatCompletionRequest,
	capabilities: ProviderCapabilities,
	provider_label: str,
) -> None:
	unsupported = unsupported_request_features(request, capabilities)

	if unsupported:
		raise unsupported_feature_error(provider_label, unsupported)


def convert_messages_to_ollama(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
	converted: List[Dict[str, Any]] = []

	for message in messages:
		payload: Dict[str, Any] = {
			"role": message.role,
			"content": "",
		}

		if message.name:
			payload["name"] = message.name

		if isinstance(message.content, str):
			payload["content"] = message.content
		elif isinstance(message.content, list):
			text_parts: List[str] = []
			images: List[str] = []

			for part in message.content:
				part_type = part.get("type")
				if part_type == "text":
					text_parts.append(part.get("text", ""))
					continue

				if part_type == "image_url":
					image_url = (part.get("image_url") or {}).get("url", "")
					if not image_url:
						continue
					if not image_url.startswith("data:image"):
						raise GatewayError(
							"Ollama /api/chat only accepts base64 data URLs for image parts",
							type=ErrorType.UNSUPPORTED_FEATURE,
							status_code=400,
						)
					images.append(image_url.split(",", 1)[-1])
					continue

				if part_type in {"input_audio", "audio", "audio_url"}:
					raise GatewayError(
						"Ollama /api/chat does not support audio input content parts",
						type=ErrorType.UNSUPPORTED_FEATURE,
						status_code=400,
					)

				raise GatewayError(
					f"Ollama /api/chat does not support content part type '{part_type}'",
					type=ErrorType.UNSUPPORTED_FEATURE,
					status_code=400,
				)

			payload["content"] = "\n".join(part for part in text_parts if part)
			if images:
				payload["images"] = images

		if message.tool_calls:
			payload["tool_calls"] = normalize_tool_calls_for_ollama(message.tool_calls)

		converted.append(payload)

	return converted


def normalize_tool_calls_for_ollama(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	normalized: List[Dict[str, Any]] = []

	for tool_call in tool_calls:
		function = tool_call.get("function") or {}
		arguments = function.get("arguments", {})

		if isinstance(arguments, str):
			if not arguments.strip():
				arguments = {}
			else:
				try:
					arguments = json.loads(arguments)
				except json.JSONDecodeError as exc:
					raise GatewayError(
						f"Tool arguments for Ollama must be valid JSON: {exc}",
						type=ErrorType.UNSUPPORTED_FEATURE,
						status_code=400,
					) from exc

		if arguments is None:
			arguments = {}

		normalized_function = {
			"name": function.get("name", ""),
			"arguments": arguments,
		}
		if function.get("description"):
			normalized_function["description"] = function["description"]

		normalized.append({"function": normalized_function})

	return normalized


def normalize_tool_calls_from_ollama(tool_calls: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
	if not tool_calls:
		return None

	normalized: List[Dict[str, Any]] = []
	for index, tool_call in enumerate(tool_calls):
		function = tool_call.get("function") or {}
		arguments = function.get("arguments", {})
		if isinstance(arguments, str):
			argument_string = arguments
		else:
			argument_string = json.dumps(arguments or {})

		normalized.append(
			{
				"id": tool_call.get("id") or f"call_{index}",
				"type": "function",
				"function": {
					"name": function.get("name", ""),
					"arguments": argument_string,
				},
			}
		)

	return normalized


def ollama_response_format(request: ChatCompletionRequest) -> Optional[Any]:
	if not request.response_format:
		return None

	response_format = request.response_format
	format_type = response_format.get("type")
	if format_type == "json_object":
		return "json"

	if format_type == "json_schema":
		json_schema = response_format.get("json_schema") or {}
		return json_schema.get("schema") or json_schema

	return None


def extract_reasoning_mode(request: ChatCompletionRequest) -> Optional[Any]:
	if request.reasoning is not None:
		reasoning = request.reasoning
		if isinstance(reasoning, dict):
			if reasoning.get("effort") is not None:
				return reasoning["effort"]
			if reasoning.get("enabled") is not None:
				return reasoning["enabled"]
			return True
		return reasoning

	if request.reasoning_effort is not None:
		return request.reasoning_effort

	return None


def _iter_content_parts(message: ChatMessage) -> List[Dict[str, Any]]:
	if isinstance(message.content, list):
		return message.content
	return []
