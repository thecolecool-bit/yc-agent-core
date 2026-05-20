import time
from typing import Literal, Optional
import uuid
from pydantic import BaseModel, Field

from yc.schemas.llm import Message


class MemoryChunk(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    content: Optional[str]=None  # 例如："用户喜欢用支付宝支付"
    type: Literal["fact", "preference", "skill", "original"]  # 记忆类型
    created_at: float = Field(default_factory=time.time)  # 创建时间
    mete_info: dict = {}  # 元信息，例如：{"importance_score": 0.5}
    original_message: Message
