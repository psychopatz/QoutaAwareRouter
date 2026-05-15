from typing import List, Optional, Union, Dict, Any, Literal
from pydantic import BaseModel, Field

# --- OpenAI Compatible Request ---

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None

# --- OpenAI Compatible Response ---

class ResponseMessage(BaseModel):
    role: str = "assistant"
    content: Optional[str] = None

class Choice(BaseModel):
    index: int
    message: ResponseMessage
    finish_reason: Optional[str] = "stop"

class Usage(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

class ProviderMetadata(BaseModel):
    id: str
    type: str
    actual_model: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    provider: ProviderMetadata
    choices: List[Choice]
    usage: Usage

# --- Streaming Deltas ---

class ChoiceDelta(BaseModel):
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None

class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChoiceDelta]
