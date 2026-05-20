from abc import ABC, abstractmethod
from typing import List, Callable, Optional
from yc.schemas.tool import ToolParam, ToolResponse


def agent_tool(name: Optional[str] = None, description: str = "") -> Callable:
    def decorator(function):
        function._tool_name = name if name else function.__name__
        function._tool_description = description
        return function

    return decorator


class Tool(ABC):
    def __init__(self, name: str, namespace: str, description: str):
        self.name = name
        self.description = description
        self.parameters: List[ToolParam] = self.get_parameters()
        self.namespace = namespace

    @abstractmethod
    def run(self, parameters: str) -> ToolResponse:
        raise NotImplementedError

    @abstractmethod
    async def arun(self, parameters: str) -> ToolResponse:
        raise NotImplementedError

    @abstractmethod
    def generated_schema(self):
        raise NotImplementedError

    @abstractmethod
    def get_parameters(self) -> List[ToolParam]:
        """获取工具参数定义"""
        raise NotImplementedError
