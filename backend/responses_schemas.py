from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from .schemas import ProviderMetadata, Usage


class ResponsesRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    input: Union[str, List[Dict[str, Any]]]
    instructions: Optional[str] = None
    previous_response_id: Optional[str] = None
    store: Optional[bool] = True
    stream: Optional[bool] = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None
    modalities: Optional[List[str]] = None
    audio: Optional[Dict[str, Any]] = None
    text: Optional[Dict[str, Any]] = None
    response_format: Optional[Dict[str, Any]] = None
    reasoning: Optional[Union[bool, str, Dict[str, Any]]] = None
    reasoning_effort: Optional[str] = None
    max_output_tokens: Optional[int] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    stop: Optional[Union[str, List[str]]] = None
    user: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None


class ResponsesOutputItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    type: str
    role: Optional[str] = None
    status: Optional[str] = None
    content: Optional[List[Dict[str, Any]]] = None
    call_id: Optional[str] = None
    name: Optional[str] = None
    arguments: Optional[str] = None


class ResponsesResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    object: str = "response"
    created_at: int
    status: str = "completed"
    model: str
    output: List[ResponsesOutputItem] = Field(default_factory=list)
    output_text: str = ""
    provider: Optional[ProviderMetadata] = None
    usage: Usage = Field(default_factory=Usage)
    metadata: Optional[Dict[str, str]] = None
    error: Optional[Dict[str, Any]] = None
    incomplete_details: Optional[Dict[str, Any]] = None