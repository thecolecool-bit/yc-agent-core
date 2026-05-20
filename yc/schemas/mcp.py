from typing import Literal, Any, Optional
from pydantic import BaseModel


class McpClientConfig(BaseModel):
    name: str
    type: Literal["http", "stdio"]
    url: Optional[str] = None
    command: Optional[str] = None
    args: Optional[list[str]] = None
    headers: Optional[dict[str, Any]] = None
    env: Optional[dict[str, Any]] = None
