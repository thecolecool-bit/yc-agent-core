from abc import ABC, abstractmethod
from typing import List, Callable, Union, Optional

from fastmcp import Client

from yc.common.event import EventBus
from yc.llm.base import LLMClient
from yc.schemas.agent import AgentEventType, AgentEvent, AgentConfig
from yc.schemas.llm import Message, LLMToolResponse, LLMRequestParams
from yc.schemas.skill import Skill
from yc.schemas.tool import ToolResponse
from yc.tool.base import Tool
from yc.tool.toolkit import Toolkit


class Agent(ABC, EventBus[AgentEventType, AgentEvent]):
    def __init__(self, config: AgentConfig, client: LLMClient) -> None:
        super().__init__()
        self._config: AgentConfig = config
        self._context: List[Message] = []
        self._llm: LLMClient = client
        self._toolkit: Toolkit = Toolkit()
        self.load_prompt()

    def load_prompt(self):
        for_name = f'你的名字叫：{self._config.name}'
        for_skill = "\r\n###技能列表（技能名：技能简介）：\r\n"
        self._config.prompt = f'{for_name}，{self._config.prompt}\r\n{for_skill}'
        self._context.append(Message(role="system", content=self._config.prompt))

    def add_memory(self, messages: List[Message]):
        for message in messages:
            if message.role == "system":
                continue
            if message.role == "assistant":
                # 存储的记忆是json格式，应转为python对象
                message.content = LLMToolResponse.model_validate(message.content)
            if message.role == "tool":
                # 存储的记忆是json格式，应转为python对象
                message.content = ToolResponse.model_validate(message.content)
            self._context.append(message)

    def get_name(self) -> str:
        return self._config.name

    def get_llm(self) -> LLMClient:
        return self._llm

    def get_context(self) -> List[Message]:
        return self._context

    def add_tool(self, tool: Union[Tool, Callable], namespace: str):
        self._toolkit.add_tool(tool, namespace)

    def add_tool_list(self, tools: List[Union[Tool, Callable]], namespace: str):
        for tool in tools:
            self.add_tool(tool, namespace)

    def add_mcp(self, mcp_client: Client):
        self._toolkit.add_mcp(mcp_client)

    async def add_mcp_async(self, mcp_client: Client):
        await self._toolkit.add_mcp_sync(mcp_client)

    def add_skill_list(self, skills: List[Skill]):
        for skill in skills:
            self.add_skill(skill)

    def add_skill(self, skill: Skill):
        self._config.prompt += f"- {skill.name} ： {skill.description}\r\n"
        if self._context: self._context.pop(0)
        self._context.insert(0, Message(role="system", content=self._config.prompt))

    @abstractmethod
    def invoke(self, content: str, params: Optional[LLMRequestParams] = None) -> str | None:
        raise NotImplementedError

    @abstractmethod
    async def ainvoke(self, content: str, params: Optional[LLMRequestParams] = None) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def stream(self, content: str, params: Optional[LLMRequestParams] = None) -> str | None:
        raise NotImplementedError

    @abstractmethod
    async def astream(self, content: str, params: Optional[LLMRequestParams] = None) -> str | None:
        raise NotImplementedError
