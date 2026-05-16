import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..errors import ErrorType, GatewayError
from ..schemas import ChatCompletionRequest, ChatCompletionResponse


UNSUPPORTED_RESPONSES_TOOL_TYPES = {
    "code_interpreter",
    "computer_use_preview",
    "computer_use",
    "file_search",
    "image_generation",
    "local_shell",
    "web_search_preview",
    "web_search",
}


def normalize_responses_tools(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    if not tools:
        return tools

    normalized_tools: List[Dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            raise GatewayError(
                "Tools must be objects",
                type=ErrorType.UNSUPPORTED_FEATURE,
                status_code=400,
            )

        tool_type = tool.get("type")
        if tool_type != "function":
            if tool_type in UNSUPPORTED_RESPONSES_TOOL_TYPES:
                raise GatewayError(
                    f"Responses tool type '{tool_type}' is not implemented by this router",
                    type=ErrorType.UNSUPPORTED_FEATURE,
                    status_code=400,
                )
            raise GatewayError(
                f"Unsupported Responses tool type '{tool_type}'",
                type=ErrorType.UNSUPPORTED_FEATURE,
                status_code=400,
            )

        function_payload = tool.get("function")
        if isinstance(function_payload, dict):
            name = function_payload.get("name")
            description = function_payload.get("description")
            parameters = function_payload.get("parameters")
            strict = function_payload.get("strict")
        else:
            name = tool.get("name")
            description = tool.get("description")
            parameters = tool.get("parameters")
            strict = tool.get("strict")

        if not isinstance(name, str) or not name:
            raise GatewayError(
                "Function tools require a non-empty name",
                type=ErrorType.UNSUPPORTED_FEATURE,
                status_code=400,
            )

        normalized_function: Dict[str, Any] = {"name": name}
        if description is not None:
            normalized_function["description"] = description
        if parameters is not None:
            normalized_function["parameters"] = parameters
        if strict is not None:
            normalized_function["strict"] = strict

        normalized_tools.append({"type": "function", "function": normalized_function})

    return normalized_tools


def normalize_responses_tool_choice(tool_choice: Any) -> Any:
    if not isinstance(tool_choice, dict):
        return tool_choice

    if tool_choice.get("type") != "function" or tool_choice.get("function"):
        return tool_choice

    name = tool_choice.get("name")
    if not isinstance(name, str) or not name:
        raise GatewayError(
            "Function tool_choice requires a non-empty name",
            type=ErrorType.UNSUPPORTED_FEATURE,
            status_code=400,
        )

    return {"type": "function", "function": {"name": name}}


def _tool_root() -> Path:
    configured_root = os.environ.get("QAROUTER_TOOL_ROOT")
    if configured_root:
        return Path(configured_root).expanduser().resolve()
    return Path.cwd().resolve()


def _resolve_path(path_value: Any) -> Path:
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError("path must be a non-empty string")

    root = _tool_root()
    candidate = Path(path_value).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()

    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path '{path_value}' is outside the allowed tool root '{root}'") from exc

    return resolved


def _path_argument(arguments: Dict[str, Any], *names: str) -> Path:
    for name in names:
        if name in arguments:
            return _resolve_path(arguments[name])
    raise ValueError(f"missing required path argument ({', '.join(names)})")


def _create_directory(arguments: Dict[str, Any]) -> Dict[str, Any]:
    path = _path_argument(arguments, "path", "dirPath", "directory")
    path.mkdir(parents=True, exist_ok=True)
    return {"ok": True, "created": True, "path": str(path.relative_to(_tool_root()))}


def _create_file(arguments: Dict[str, Any]) -> Dict[str, Any]:
    path = _path_argument(arguments, "path", "filePath")
    content = arguments.get("content")
    if content is None:
        content = arguments.get("contents", "")
    if not isinstance(content, str):
        content = json.dumps(content)

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return {
            "ok": False,
            "created": False,
            "path": str(path.relative_to(_tool_root())),
            "error": "file already exists",
        }

    path.write_text(content, encoding="utf-8")
    return {
        "ok": True,
        "created": True,
        "path": str(path.relative_to(_tool_root())),
        "bytes_written": len(content.encode("utf-8")),
    }


def _write_file(arguments: Dict[str, Any]) -> Dict[str, Any]:
    path = _path_argument(arguments, "path", "filePath")
    content = arguments.get("content")
    if content is None:
        content = arguments.get("contents", "")
    if not isinstance(content, str):
        content = json.dumps(content)

    append = bool(arguments.get("append"))
    path.parent.mkdir(parents=True, exist_ok=True)
    if append:
        with path.open("a", encoding="utf-8") as file_handle:
            file_handle.write(content)
    else:
        path.write_text(content, encoding="utf-8")

    return {
        "ok": True,
        "written": True,
        "appended": append,
        "path": str(path.relative_to(_tool_root())),
        "bytes_written": len(content.encode("utf-8")),
    }


def _read_file(arguments: Dict[str, Any]) -> Dict[str, Any]:
    path = _path_argument(arguments, "path", "filePath")
    if not path.exists() or not path.is_file():
        return {
            "ok": False,
            "path": str(path.relative_to(_tool_root())),
            "error": "file not found",
        }

    start_line = int(arguments.get("startLine") or arguments.get("start_line") or 1)
    end_line = arguments.get("endLine") or arguments.get("end_line")
    lines = path.read_text(encoding="utf-8").splitlines()
    start_index = max(start_line - 1, 0)
    end_index = len(lines) if end_line is None else max(int(end_line), start_line)
    content = "\n".join(lines[start_index:end_index])
    return {
        "ok": True,
        "path": str(path.relative_to(_tool_root())),
        "start_line": start_line,
        "end_line": end_index,
        "content": content,
    }


def _list_dir(arguments: Dict[str, Any]) -> Dict[str, Any]:
    raw_path = arguments.get("path", ".")
    path = _resolve_path(raw_path)
    if not path.exists() or not path.is_dir():
        return {
            "ok": False,
            "path": str(path.relative_to(_tool_root())) if path.exists() else str(raw_path),
            "error": "directory not found",
        }

    entries = [
        {"name": child.name, "type": "dir" if child.is_dir() else "file"}
        for child in sorted(path.iterdir(), key=lambda child: child.name.lower())
    ]
    return {"ok": True, "path": str(path.relative_to(_tool_root())), "entries": entries}


def _file_search(arguments: Dict[str, Any]) -> Dict[str, Any]:
    query = arguments.get("query") or arguments.get("pattern") or arguments.get("glob")
    if not isinstance(query, str) or not query:
        raise ValueError("file_search requires a non-empty query")

    max_results = int(arguments.get("maxResults") or arguments.get("max_results") or 50)
    root = _tool_root()
    results: List[str] = []
    for match in root.glob(query):
        try:
            relative = str(match.relative_to(root))
        except ValueError:
            continue
        results.append(relative)
        if len(results) >= max_results:
            break

    return {"ok": True, "query": query, "matches": results}


EXECUTABLE_FUNCTIONS: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "create_directory": _create_directory,
    "create_file": _create_file,
    "file_search": _file_search,
    "list_dir": _list_dir,
    "read_file": _read_file,
    "write_file": _write_file,
}


def auto_executable_function_names() -> Set[str]:
    return set(EXECUTABLE_FUNCTIONS.keys())


def _declared_function_names(tools: Optional[List[Dict[str, Any]]]) -> Set[str]:
    names: Set[str] = set()
    for tool in tools or []:
        function_payload = tool.get("function") or {}
        name = function_payload.get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return names


def _assistant_message_from_response(chat_response: ChatCompletionResponse) -> Dict[str, Any]:
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


def _tool_call_name(tool_call: Dict[str, Any]) -> Optional[str]:
    function_payload = tool_call.get("function") or {}
    name = function_payload.get("name")
    return name if isinstance(name, str) and name else None


def _parse_tool_arguments(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    function_payload = tool_call.get("function") or {}
    arguments = function_payload.get("arguments")
    if arguments in (None, ""):
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        parsed = json.loads(arguments)
        if not isinstance(parsed, dict):
            raise ValueError("tool arguments must decode to an object")
        return parsed
    raise ValueError("tool arguments must be a JSON object or string")


def _execute_tool_calls(tool_calls: List[Dict[str, Any]], declared_function_names: Set[str]) -> Optional[List[Dict[str, Any]]]:
    if not tool_calls:
        return None

    tool_names = [_tool_call_name(tool_call) for tool_call in tool_calls]
    if not all(name and name in declared_function_names and name in EXECUTABLE_FUNCTIONS for name in tool_names):
        return None

    tool_messages: List[Dict[str, Any]] = []
    for tool_call, function_name in zip(tool_calls, tool_names):
        call_id = tool_call.get("id")
        if not isinstance(call_id, str) or not call_id:
            raise GatewayError(
                "Function tool calls require an id for server execution",
                type=ErrorType.UNSUPPORTED_FEATURE,
                status_code=400,
            )

        try:
            arguments = _parse_tool_arguments(tool_call)
            result = EXECUTABLE_FUNCTIONS[function_name](arguments)
        except Exception as exc:
            result = {"ok": False, "error": str(exc)}

        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": function_name,
                "content": json.dumps(result),
            }
        )

    return tool_messages


async def route_with_auto_executed_tools(
    router_instance,
    chat_request: ChatCompletionRequest,
    *,
    max_iterations: int = 8,
) -> Tuple[ChatCompletionResponse, List[Dict[str, Any]]]:
    request_payload = chat_request.model_dump(exclude_none=True)
    conversation_messages = [message.model_dump(exclude_none=True) for message in chat_request.messages]
    declared_function_names = _declared_function_names(chat_request.tools)
    current_request = chat_request

    for _ in range(max_iterations):
        chat_response = await router_instance.route(current_request)
        assistant_message = _assistant_message_from_response(chat_response)
        tool_calls = assistant_message.get("tool_calls") or []

        if not tool_calls:
            return chat_response, [*conversation_messages, assistant_message]

        tool_messages = _execute_tool_calls(tool_calls, declared_function_names)
        if not tool_messages:
            return chat_response, [*conversation_messages, assistant_message]

        conversation_messages.extend([assistant_message, *tool_messages])
        current_request = ChatCompletionRequest(**{**request_payload, "messages": conversation_messages})

    raise GatewayError(
        "Server-side tool execution exceeded the maximum tool loop iterations",
        type=ErrorType.UNSUPPORTED_FEATURE,
        status_code=400,
    )