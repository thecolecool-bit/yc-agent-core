from typing import Optional
from pydantic import BaseModel, Field


class Skill(BaseModel):
    name: str = Field(..., description="技能名称，对应技能文件夹名")
    description: str = Field(..., description="技能的简短描述")
    path: str = Field(..., description="技能文件夹的绝对路径")
    content: Optional[str] = None
