from abc import ABC, abstractmethod
from yc.common.config import EnvConfigLoader
from yc.common.event import EventBus
from yc.common.exceptions import AppError
from yc.agent.base import Agent
from yc.llm.base import LLMClient
from yc.schemas.agent import AgentEventType, AgentEvent
from yc.schemas.context_processor import ContextProcessorEventType, ContextProcessorEvent
from yc.schemas.llm import Message


class ContextProcessor(ABC, EventBus[ContextProcessorEventType, ContextProcessorEvent]):

    @abstractmethod
    def register(self, agent: Agent):
        raise NotImplementedError


class SlidingWindowContextProcessor(ContextProcessor):

    def __init__(self, window_size: int):
        super().__init__()
        self._window_size = window_size

    def register(self, agent: Agent):
        async def _compress(event: AgentEvent):
            try:
                context = agent.get_context()
                message_list = [message for message in context if message.role == "user" or message.role == "assistant"]
                if len(message_list) > self._window_size * 2:
                    await self.async_broadcast(ContextProcessorEvent(type=ContextProcessorEventType.COMPRESS_START,
                                                                     data={"dialogue_count": len(message_list),
                                                                           "window_size:": self._window_size}))
                    context.pop(1)
                    message_list = [message for message in context if
                                    message.role == "user" or message.role == "assistant"]
                    await self.async_broadcast(ContextProcessorEvent(type=ContextProcessorEventType.COMPRESS_END,
                                                                     data={"dialogue_count": len(message_list),
                                                                           "window_size:": self._window_size}))
            except Exception as e:
                await self.async_broadcast(ContextProcessorEvent(type=ContextProcessorEventType.COMPRESS_ERROR,
                                                                 data=e))

        agent.subscribe_event(AgentEventType.UPDATE_CONTEXT, _compress)


class SummarizingContextProcessor(ContextProcessor):

    def __init__(self, compression_threshold: float, llm: LLMClient):
        super().__init__()
        self.prompt = None
        self._llm = llm
        self.load_prompt()
        self._compression_threshold = compression_threshold

    def register(self, agent: Agent):
        async def _compress(event: AgentEvent):
            try:
                context_usage = event.data.usage.total_tokens / (agent.get_llm().get_context_max_length() * 1000)
                await self.async_broadcast(ContextProcessorEvent(type=ContextProcessorEventType.MEASURE,
                                                                 data={
                                                                     "now_token_usage": event.data.usage.total_tokens,
                                                                     "context_max_length": agent.get_llm().get_context_max_length(),
                                                                     "compression_threshold:": self._compression_threshold}))
                if context_usage < self._compression_threshold:
                    return
                await self.async_broadcast(ContextProcessorEvent(type=ContextProcessorEventType.COMPRESS_START,
                                                                 data={"context_usage": context_usage,
                                                                       "context_max_length": agent.get_llm().get_context_max_length(),
                                                                       "compression_threshold:": self._compression_threshold}))
                context_content = ""
                context = agent.get_context()
                for message in context:
                    if not message.role == "system":
                        context_content += f'\n{message.role}：{message.content}'
                self.prompt += context_content
                sys_msg = Message(role="system", content=self.prompt)
                qu_msg = Message(role="user", content=f"这是我们之前的聊天记录：{context_content}\n现在，你开始行动吧")
                try:
                    response = self._llm.invoke([sys_msg, qu_msg])
                except AppError as e:
                    raise AppError(f"压缩历史记录出现异常：{e.message}")
                context[1:] = []
                history = response.message.content if response.message else ""
                context.append(Message(role="user", content=f"这是总结的历史聊天记录：{history}"))
                before_input_tokens = response.usage.input_tokens if response.usage else 0
                context_usage = before_input_tokens / (agent.get_llm().get_context_max_length() * 1000)
                await self.async_broadcast(ContextProcessorEvent(type=ContextProcessorEventType.COMPRESS_END,
                                                                 data={"context_usage": context_usage,
                                                                       "context_max_length": agent.get_llm().get_context_max_length(),
                                                                       "compression_threshold:": self._compression_threshold}))
            except Exception as e:
                await self.async_broadcast(ContextProcessorEvent(type=ContextProcessorEventType.COMPRESS_ERROR,
                                                                 data=e))

        agent.subscribe_event(AgentEventType.REPLY, _compress)

    def load_prompt(self):
        config_loader = EnvConfigLoader()
        self.prompt = config_loader.get_value(None, "SUMMARIZING_PROMPT")
        self.prompt = eval(self.prompt)
