from typing import List, Any

from openai.types.chat import ChatCompletionMessageToolCall, ChatCompletionMessage
from openai.types.chat.chat_completion_message_function_tool_call import Function

from yc.llm.openai_client import OpenAIClient
from yc.schemas.llm import Message


class MiniMaxClient(OpenAIClient):

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
