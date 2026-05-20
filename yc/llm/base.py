from abc import ABC, abstractmethod
from typing import List, Any, Iterator, AsyncIterator, AsyncGenerator, Generator, Optional

from yc.tool.base import Tool
from yc.schemas.llm import Message, LLMResponse, LLMConfig, LLMRequestParams, LLMToolResponse


class LLMClient(ABC):
    def __init__(self, config: LLMConfig):
        self._config = config
        self._client = self._create_client()
        self._async_client = self._create_async_client()

    def get_name(self) -> str:
        return self._config.model

    def get_context_max_length(self) -> int:
        return self._config.context_max_length

    def get_tpm(self) -> int:
        return self._config.tpm

    def get_rpm(self) -> float:
        return self._config.rpm

    @abstractmethod
    def _create_client(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def _create_async_client(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def invoke(self, messages: List[Message], params: Optional[LLMRequestParams] = None) -> LLMResponse:
        raise NotImplementedError

    @abstractmethod
    async def ainvoke(self, messages: List[Message], params: Optional[LLMRequestParams] = None) -> LLMResponse:
        raise NotImplementedError

    @abstractmethod
    def stream(self, messages: List[Message], params: Optional[LLMRequestParams] = None) -> Iterator[LLMResponse]:
        raise NotImplementedError

    @abstractmethod
    async def astream(self, messages: List[Message], params: Optional[LLMRequestParams] = None) -> AsyncIterator[
        LLMResponse]:
        raise NotImplementedError

    @abstractmethod
    def invoke_with_tools(
            self, messages: List[Message], tools: List[Tool], params: Optional[LLMRequestParams] = None
    ) -> LLMToolResponse:
        raise NotImplementedError

    @abstractmethod
    async def ainvoke_with_tools(
            self, messages: List[Message], tools: List[Tool], params: Optional[LLMRequestParams] = None
    ) -> LLMToolResponse:
        raise NotImplementedError

    @abstractmethod
    def stream_with_tools(
            self, messages: List[Message], tools: List[Tool], params: Optional[LLMRequestParams] = None
    ) -> Generator[LLMToolResponse]:
        raise NotImplementedError

    @abstractmethod
    async def astream_with_tools(
            self, messages: List[Message], tools: List[Tool], params: Optional[LLMRequestParams] = None
    ) -> AsyncGenerator[LLMToolResponse]:
        raise NotImplementedError
