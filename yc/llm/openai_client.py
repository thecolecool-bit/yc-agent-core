import json
import time
from typing import List, Any, Iterator, AsyncIterator, Dict, Type, AsyncGenerator, Generator, Optional
from openai import OpenAI, AsyncOpenAI, APIStatusError, NotFoundError, APITimeoutError, APIConnectionError
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_function_tool_call import Function

from yc.common.exceptions import InsufficientBalanceError, ParameterError, ServerFaultError, ServerBusyError, \
    LLMClientError, LLMNotFoundError, ApiTimeoutError, APiConnectionError, BadRequestError, RateLimitError, \
    AuthenticationError
from yc.llm.base import LLMClient
from yc.tool.base import Tool
from yc.schemas.llm import Message, OpenAIResponse, ToolCall, Usage, \
    OpenAIConfig, OpenAIRequestParams, OpenAIToolResponse, LLMRequestParams


class OpenAIClient(LLMClient):

    def __init__(self, config: OpenAIConfig):
        super().__init__(config)

    def _create_client(self) -> Any:
        return OpenAI(
            base_url=self._config.base_url,
            api_key=self._config.api_key,
            timeout=self._config.timeout,
        )

    def _create_async_client(self) -> Any:
        return AsyncOpenAI(
            base_url=self._config.base_url,
            api_key=self._config.api_key,
            timeout=self._config.timeout,
        )

    def _build_messages(self, messages: List[Message]) -> List[Any]:
        messages_list = []
        for message in messages:
            if message.role == "assistant":
                chat_tool_calls = []
                for tool_call in message.content.tool_calls if message.content.tool_calls else []:
                    chat_tool_calls.append(
                        ChatCompletionMessageToolCall(id=tool_call.call_id, function=Function(
                            name=tool_call.name, arguments=tool_call.arguments), type='function'))
                messages_list.append(ChatCompletionMessage(role='assistant', content=message.content.content,
                                                           tool_calls=chat_tool_calls if chat_tool_calls else None).model_dump())
                continue
            messages_list.append({"role": message.role, "content": f"{message.content}"})
        return messages_list

    def invoke(self, messages: List[Message], params: Optional[OpenAIRequestParams] = None) -> OpenAIResponse:
        try:
            start = time.time()
            response = self._client.chat.completions.create(
                model=self._config.model,
                messages=self._build_messages(messages),
                stream=False,
                **params.model_dump() if params else {}
            )
            response.start = start
            return self._format_response(response, False)
        except Exception as exc:
            error = self._error_handle(exc)
            raise error if error else LLMClientError(
                message=f'An exception occurred while calling {self._config.model}，{exc}',
                detail={"error_info": str(exc)})

    async def ainvoke(self, messages: List[Message], params: Optional[OpenAIRequestParams] = None) -> OpenAIResponse:
        try:
            start = time.time()
            response = await self._async_client.chat.completions.create(
                model=self._config.model,
                messages=self._build_messages(messages),
                stream=False,
                **params.model_dump() if params else {}
            )
            response.start = start
            return self._format_response(response, False)
        except Exception as exc:
            error = self._error_handle(exc)
            raise error if error else LLMClientError(
                message=f'An exception occurred while calling {self._config.model}，{exc}',
                detail={"error_info": str(exc)})

    def stream(self, messages: List[Message], params: Optional[OpenAIRequestParams] = None) -> Iterator[OpenAIResponse]:
        try:
            start = time.time()
            response = self._client.chat.completions.create(
                model=self._config.model,
                messages=self._build_messages(messages),
                stream=True,
                **params.model_dump() if params else {}
            )
            for chunk in response:
                response.start = start
                yield self._format_response(chunk, True)
        except Exception as exc:
            error = self._error_handle(exc)
            raise error if error else LLMClientError(
                message=f'An exception occurred while calling {self._config.model}，{exc}',
                detail={"error_info": str(exc)})

    async def astream(self, messages: List[Message], params: Optional[OpenAIRequestParams] = None) -> AsyncIterator[
        OpenAIResponse]:
        try:
            start = time.time()
            response = await self._async_client.chat.completions.create(
                model=self._config.model,
                messages=self._build_messages(messages),
                stream=True,
                **params.model_dump() if params else {}
            )
            for chunk in response:
                response.start = start
                yield self._format_response(chunk, True)
        except Exception as exc:
            error = self._error_handle(exc)
            raise error if error else LLMClientError(
                message=f'An exception occurred while calling {self._config.model}，{exc}',
                detail={"error_info": str(exc)})

    @staticmethod
    def _format_response(response: Any, is_stream: bool) -> OpenAIResponse:
        end = time.time()
        content = None
        reasoning_content = None
        if response.choices and len(response.choices) > 0:
            if is_stream:
                data = response.choices[0].delta
            else:
                data = response.choices[0].message
            content = getattr(data, "content", None)
            reasoning_content = getattr(data, "reasoning_content", None)
        latency_ms = int((end - response.start) * 1000)
        usage = None
        if getattr(response, "usage", None):
            usage = Usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
        return OpenAIResponse(
            model=response.model,
            message=Message(role="assistant", content=content),
            reasoning_content=reasoning_content,
            usage=usage,
            latency_ms=latency_ms,
        )

    def invoke_with_tools(
            self, messages: List[Message], tools: List[Tool], params: Optional[OpenAIRequestParams] = None
    ) -> OpenAIToolResponse:
        try:
            tools_schemas = [json.loads(tool.generated_schema()) for tool in tools]
            start = time.time()
            response = self._client.chat.completions.create(
                model=self._config.model,
                messages=self._build_messages(messages),
                stream=False,
                tools=tools_schemas,
                **params.model_dump() if params else {}
            )
            response.start = start
            return self._format_tool_call_response(response, False)
        except Exception as exc:
            error = self._error_handle(exc)
            raise error if error else LLMClientError(
                message=f'An exception occurred while calling {self._config.model}，{exc}',
                detail={"error_info": str(exc)})

    async def ainvoke_with_tools(
            self, messages: List[Message], tools: List[Tool], params: Optional[OpenAIRequestParams] = None
    ) -> OpenAIToolResponse:
        try:
            tools_schemas = [json.loads(tool.generated_schema()) for tool in tools]
            start = time.time()
            response = await self._async_client.chat.completions.create(
                model=self._config.model,
                messages=self._build_messages(messages),
                stream=False,
                tools=tools_schemas,
                **params.model_dump() if params else {}
            )
            response.start = start
            return self._format_tool_call_response(response, False)
        except Exception as exc:
            error = self._error_handle(exc)
            raise error if error else LLMClientError(
                message=f'An exception occurred while calling {self._config.model}，{exc}',
                detail={"error_info": str(exc)})

    def stream_with_tools(self, messages: List[Message], tools: List[Tool],
                          params: Optional[LLMRequestParams] = None) -> Generator[OpenAIToolResponse]:
        try:
            tools_schemas = [json.loads(tool.generated_schema()) for tool in tools]
            response = self._client.chat.completions.create(
                model=self._config.model,
                messages=self._build_messages(messages),
                stream=True,
                stream_options={"include_usage": True},
                tools=tools_schemas,
                **params.model_dump() if params else {}
            )
            for chunk in response:
                yield self._format_tool_call_response(chunk, True)
        except Exception as exc:
            error = self._error_handle(exc)
            raise error if error else LLMClientError(
                message=f'An exception occurred while calling {self._config.model}，{exc}',
                detail={"error_info": str(exc)})

    async def astream_with_tools(self, messages: List[Message], tools: List[Tool],
                                 params: Optional[LLMRequestParams] = None) -> AsyncGenerator[OpenAIToolResponse]:
        try:
            tools_schemas = [json.loads(tool.generated_schema()) for tool in tools]
            response = await self._async_client.chat.completions.create(
                model=self._config.model,
                messages=self._build_messages(messages),
                stream=True,
                stream_options={"include_usage": True},
                tools=tools_schemas,
                **params.model_dump() if params else {}
            )
            async for chunk in response:
                yield self._format_tool_call_response(chunk, True)
        except Exception as exc:
            error = self._error_handle(exc)
            raise error if error else LLMClientError(
                message=f'An exception occurred while calling {self._config.model}，{exc}',
                detail={"error_info": str(exc)})

    def _format_tool_call_response(self, response: Any, is_stream: bool) -> OpenAIToolResponse:
        content = None
        reasoning_content = None
        tool_calls = []
        if response.choices and len(response.choices) > 0:
            if is_stream:
                data = response.choices[0].delta
            else:
                data = response.choices[0].message
            content = getattr(data, "content", None)
            reasoning_content = getattr(data, "reasoning_content", None)
            tool_calls = getattr(data, "tool_calls", None)
        usage = None
        if getattr(response, "usage", None):
            usage = Usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens
            )
        return OpenAIToolResponse(
            model=self._config.model,
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=[
                ToolCall(
                    call_id=tool.id if tool.id else "",
                    name=tool.function.name if tool.function.name else "",
                    arguments=tool.function.arguments if tool.function.arguments else "",
                )
                for tool in (tool_calls if tool_calls else [])
            ],
            usage=usage
        )

    @staticmethod
    def _error_mapping() -> Dict[int, Type[LLMClientError]]:
        """
            定义 HTTP 状态码与项目业务异常类的映射关系。

                Returns:
                    :return Dict[int, Type[ClientError]]: 状态码到异常类的映射字典。
        """
        error_mapping = dict()
        error_mapping[400] = BadRequestError
        error_mapping[401] = AuthenticationError
        error_mapping[402] = InsufficientBalanceError
        error_mapping[403] = InsufficientBalanceError
        error_mapping[404] = LLMNotFoundError
        error_mapping[422] = ParameterError
        error_mapping[429] = RateLimitError
        error_mapping[500] = ServerFaultError
        error_mapping[503] = ServerBusyError
        return error_mapping

    def _error_handle(self, exc: Exception) -> LLMClientError | None:
        """
            根据捕获的 OpenAI SDK 异常实例，转换并返回对应的项目业务异常对象。

            先处理已在error_mapping中注册过 HTTP 状态码与项目异常类映射关系的异常；
            然后处理网络连接、超时等特定 SDK 异常。若不匹配，则返回 None。

                Args:
                    :param exc: 捕获到的原始 OpenAI SDK 异常对象。

                Returns:
                    :return ClientError | None: 转换后的项目业务异常对象，若不匹配则返回 None。
            """
        if isinstance(exc, APIStatusError):
            err_class = self._error_mapping().get(exc.status_code)
            return err_class() if err_class else None
        elif isinstance(exc, NotFoundError):
            return LLMNotFoundError()
        elif isinstance(exc, APITimeoutError):
            return ApiTimeoutError()
        elif isinstance(exc, APIConnectionError):
            return APiConnectionError()
        return None
