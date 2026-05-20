import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EventType(Enum):
    pass


class Event(BaseModel):
    type: EventType
    timestamp: float = Field(default_factory=time.time)
    data: Any = None
