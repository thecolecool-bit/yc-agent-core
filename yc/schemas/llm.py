from typing import Any, List, Literal, Optional, Dict
import uuid
import time
from pydantic import BaseModel, Field, computed_field


class LLMConfig(BaseModel):
    model: str
    base_url: str
    api_key: str
    timeout: int = Field(default=60, ge=10, le=360)
    context_max_length: int = Field(default=8, ge=5, description="模型支持的上下文大小，指token数量，单位（k）")
    tpm: int = Field(default=40, ge=10, description="模型限制的每分钟最大token数量，指输入输出总token量，单位（k）")
    rpm: float = Field(default=1, ge=0.1, description="模型限制的每分钟最大请求数，单位（k）")
    client: str = Field(default="openai", description="默认使用OpenAI客户端")
    enable: bool = True


class OpenAIConfig(LLMConfig):
    model: str
    base_url: str
    api_key: str
    timeout: int = Field(default=60, ge=10, le=360)
    enable: bool = True


class LLMRequestParams(BaseModel):
    max_tokens: int = Field(default=None, ge=10)
    temperature: float = Field(default=None, ge=0, le=2)
    top_p: float = Field(default=None, ge=0, le=1)
    extra_body: Optional[Dict[str, Any]] = None


class OpenAIRequestParams(LLMRequestParams):
    max_tokens: int = Field(default=None, ge=10)
    temperature: float = Field(default=None, ge=0, le=2)
    top_p: float = Field(default=None, ge=0, le=1)
    extra_body: Optional[Dict[str, Any]] = None


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    @computed_field
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __iadd__(self, other):
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        return self


class Message(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    role: Literal["system", "user", "tool", "assistant"]
    content: Any
    token: Optional[int] = None
    metadata: Optional[dict[str, Any]] = None
    timestamp: float = Field(default_factory=time.time)


class LLMResponse(BaseModel):
    model: str
    message: Optional[Message] = None
    reasoning_content: Optional[str] = None
    usage: Optional[Usage] = None
    latency_ms: int = 0


class OpenAIResponse(LLMResponse):
    model: str
    message: Optional[Message] = None
    reasoning_content: Optional[str] = None
    usage: Optional[Usage] = None
    latency_ms: int = 0


class ToolCall(BaseModel):
    call_id: str
    name: str
    arguments: str

    def __iadd__(self, other):
        if not self.call_id:
            self.call_id = other.call_id
        if not self.name:
            self.name = ""
        if not self.arguments:
            self.arguments = ""
        self.name += other.name
        self.arguments += other.arguments
        return self


class LLMToolResponse(BaseModel):
    model: str
    reasoning_content: Optional[str] = None
    usage: Optional[Usage] = None
    latency_ms: int = 0
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None

    def __iadd__(self, other):
        if not self.model:
            self.model = ""
        if not self.reasoning_content:
            self.reasoning_content = ""
        if not self.content:
            self.content = ""
        if not self.usage:
            self.usage = Usage()
        if other.model:
            self.model = other.model
        if other.reasoning_content:
            self.reasoning_content += other.reasoning_content
        if other.content:
            self.content += other.content
        if other.usage:
            self.usage = other.usage
        self.latency_ms = other.latency_ms

        if other.tool_calls:
            current_tool_call_chunk = other.tool_calls[0]
            if self.tool_calls is None: self.tool_calls = []
            if current_tool_call_chunk.call_id:
                # 新的工具调用
                if self.tool_calls is not None: self.tool_calls.append(current_tool_call_chunk)
            else:
                # 现有工具调用的参数碎片，合并到列表最后一个工具调用上
                if self.tool_calls is not None: self.tool_calls[-1] += current_tool_call_chunk
        return self


class OpenAIToolResponse(LLMToolResponse):
    model: str
    reasoning_content: Optional[str] = None
    usage: Optional[Usage] = None
    latency_ms: int = 0
    content: Optional[Any] = None
    tool_calls: Optional[List[ToolCall]] = None
