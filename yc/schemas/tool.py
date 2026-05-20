from enum import Enum
from typing import Any, Dict, Optional, List
from pydantic import BaseModel


class ToolStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"


class ToolParam(BaseModel):
    name: str
    type: Optional[str] = None
    description: Optional[str] = None
    required: bool = True
    default: Optional[Any] = None
    enum: Optional[List[Any]] = None
    items: Optional[dict] = None


class ToolResponse(BaseModel):
    call_id: Optional[str] = None
    name: Optional[str] = None
    status: ToolStatus = ToolStatus.SUCCESS
    content: Any
    elapsed_ms: Optional[int] = None
    metadata: Dict[str, Any] = {}

    @staticmethod
    def success(
            name:str,
            content: Any,
            elapsed_ms: Optional[int] = None,
    ) -> ToolResponse:
        return ToolResponse(
            name=name,
            status=ToolStatus.SUCCESS,
            content=content,
            elapsed_ms=elapsed_ms
        )

    @staticmethod
    def error(
            name: str,
            content: Any,
            elapsed_ms: Optional[int] = None
    ) -> ToolResponse:
        return ToolResponse(
            name=name,
            status=ToolStatus.ERROR,
            content=content,
            elapsed_ms=elapsed_ms
        )

    def set_metadata(
            self,
            key: str,
            value: Any
    ):
        self.metadata[key] = value
