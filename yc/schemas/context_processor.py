import time
from typing import Optional, Any
from pydantic import Field
from yc.schemas.event import EventType, Event


class ContextProcessorEventType(EventType):
    COMPRESS_START = "compress_start"
    COMPRESS_END = "compress_end"
    COMPRESS_ERROR = "compress_error"
    MEASURE = "measure"


class ContextProcessorEvent(Event):
    type: ContextProcessorEventType
    content: Optional[str] = None
    data: Optional[Any] = None
    timestamp: float = Field(default_factory=time.time)
