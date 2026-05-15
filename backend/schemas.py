from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, ConfigDict, Field, model_validator

# --- OpenAI Compatible Request ---

class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    audio: Optional[Dict[str, Any]] = None
    refusal: Optional[str] = None

    @model_validator(mode="after")
    def validate_content_or_tool_calls(self):
        if self.content is None and not self.tool_calls and self.role != "assistant":
            raise ValueError("content is required unless assistant tool_calls are supplied")
        return self

class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    stream_options: Optional[Dict[str, Any]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None
    modalities: Optional[List[str]] = None
    audio: Optional[Dict[str, Any]] = None
    response_format: Optional[Dict[str, Any]] = None
    reasoning: Optional[Union[bool, str, Dict[str, Any]]] = None
    reasoning_effort: Optional[str] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None

# --- OpenAI Compatible Response ---

class ResponseMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    audio: Optional[Dict[str, Any]] = None
    refusal: Optional[str] = None
    reasoning: Optional[str] = None

class Choice(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: int
    message: ResponseMessage
    finish_reason: Optional[str] = "stop"

class Usage(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

class ProviderMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    type: str
    actual_model: str

class ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    provider: ProviderMetadata
    choices: List[Choice]
    usage: Usage

# --- Streaming Deltas ---

class ChoiceDelta(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None

class ChatCompletionChunk(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChoiceDelta]
