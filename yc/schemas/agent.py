import time
from typing import Optional, Any
from pydantic import BaseModel, Field
from yc.common.event import EventType, Event


class AgentConfig(BaseModel):
    name: str
    prompt: str
    max_step: int = 100
    max_retry_count: int = 3
    rate_limit_delay: int = 60
    use_built_in_tools: bool = True

class AgentEventType(EventType):
    RUN_STARTED = "run_started"
    RUN_FINISHED = "run_finished"
    RUN_ERROR = "run_error"

    STEP_STARTED = "step_started"
    STEP_FINISHED = "step_finished"

    REQUEST = "request"

    REPLY = "reply"

    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"

    UPDATE_CONTEXT = "update_context"

    STREAM_CONTENT_START = "stream_content_start"
    STREAM_CONTENT_ING = "stream_contenting"
    STREAM_CONTENT_END = "stream_content_end"

    STREAM_REASONING_START = "steam_reasoning_start"
    STREAM_REASONING_ING = "steam_reasoning"
    STREAM_REASONING_END = "steam_reasoning_end"


class AgentEvent(Event):
    agent_name: str
    type: EventType
    data: Any = None
    content: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)
