from typing import List, Any
from volcenginesdkarkruntime import Ark, AsyncArk
# noinspection PyProtectedMember
from volcenginesdkarkruntime._exceptions import ArkAPIStatusError, ArkNotFoundError, ArkAPITimeoutError, \
    ArkAPIConnectionError
from volcenginesdkarkruntime.types.chat import ChatCompletionMessageToolCall, ChatCompletionMessage
from volcenginesdkarkruntime.types.chat.chat_completion_message_tool_call import Function

from yc.llm.openai_client import OpenAIClient
from yc.schemas.llm import Message
from yc.common.exceptions import LLMClientError, LLMNotFoundError, ApiTimeoutError, APiConnectionError


class ArkClient(OpenAIClient):

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
            elif message.role == "tool":
                messages_list.append(
                    {"role": message.role, "tool_call_id": message.content.call_id, "content": f"{message.content}"})
            else:
                messages_list.append({"role": message.role, "content": f"{message.content}"})
        return messages_list

    def _create_client(self) -> Any:
        return Ark(api_key=self._config.api_key)

    def _create_async_client(self) -> Any:
        return AsyncArk(api_key=self._config.api_key)

    def _error_handle(self, exc: Exception) -> LLMClientError | None:
        """
            根据捕获的 ARK SDK 异常实例，转换并返回对应的项目业务异常对象。
            """
        if isinstance(exc, ArkAPIStatusError):
            err_class = self._error_mapping().get(exc.status_code)
            return err_class() if err_class else None
        elif isinstance(exc, ArkNotFoundError):
            return LLMNotFoundError()
        elif isinstance(exc, ArkAPITimeoutError):
            return ApiTimeoutError()
        elif isinstance(exc, ArkAPIConnectionError):
            return APiConnectionError()
        return None
