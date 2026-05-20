from typing import Optional
from yc.agent.base import Agent
from yc.common.exceptions import LLMClientError, ToolNotFountError
from yc.llm.base import LLMClient
from yc.tool.base import agent_tool
from yc.schemas.agent import AgentEvent, AgentEventType, AgentConfig
from yc.schemas.llm import Message, LLMRequestParams, LLMToolResponse, Usage
from yc.schemas.tool import ToolStatus


class ReActAgent(Agent):

    def __init__(self, config: AgentConfig, client: LLMClient):
        super().__init__(config, client)
        self.add_built_in_function()

    def add_built_in_function(self):

        @agent_tool(name="Finish", description="等待指令或任务结束，调用此工具以等候。")
        def finish(answer: str):
            """
            等待指令或任务结束，调用此工具以等候。
            :param answer: 执行结果
            """
            return answer

        if self._config.use_built_in_tools:
            self.add_tool_list([finish], "built_in")

    def invoke(self, content: str, params: Optional[LLMRequestParams] = None) -> str | None:
        # --- 开始新任务 ---
        self.broadcast(
            AgentEvent(type=AgentEventType.RUN_STARTED, agent_name=self._config.name, content="新任务",
                       data={"user_input": content}))

        # --- 更新上下文，保存用户消息 ---
        user_msg = Message(role="user", content=content)
        self._context.append(user_msg)
        self.broadcast(AgentEvent(type=AgentEventType.UPDATE_CONTEXT,
                                  agent_name=self.get_name(), data=user_msg))
        current_step = 0

        while True:
            # --- 步数检查 ---
            if current_step == self._config.max_step:
                self.broadcast(AgentEvent(type=AgentEventType.RUN_FINISHED, content="超出限定步数",
                                          agent_name=self._config.name,
                                          data={"current_step": current_step,
                                                "max_step": self._config.max_step}))
                return "抱歉，我没有在限定步数内完成任务。"

            # --- 开始新步骤 ---
            current_step += 1
            self.broadcast(AgentEvent(type=AgentEventType.STEP_STARTED, agent_name=self._config.name,
                                      content=f"开始第 {current_step} 步",
                                      data=current_step))

            # --- 请求模型 ---
            response = LLMToolResponse(model=self._config.name)
            try:
                self.broadcast(
                    AgentEvent(type=AgentEventType.REQUEST, agent_name=self._config.name, content="开始请求模型",
                               data=self._llm.get_name()))

                response = self._llm.invoke_with_tools(
                    self._context, self._toolkit.tool_list(), params
                )

                # --- 模型响应成功 ---
                self.broadcast(
                    AgentEvent(type=AgentEventType.REPLY, agent_name=self._config.name,
                               content=f"模型 {response.model} 响应成功", data=response))

                # --- 更新上下文，保存模型消息 ---
                assistant_msg = Message(role="assistant", content=response)
                self._context.append(assistant_msg)
                self.broadcast(AgentEvent(type=AgentEventType.UPDATE_CONTEXT,
                                          agent_name=self.get_name(), data=assistant_msg))
            # --- 请求模型失败 ---
            except LLMClientError as e:
                # --- 处理异常 ---
                self.broadcast(AgentEvent(type=AgentEventType.RUN_ERROR, content=e.message,
                                          agent_name=self._config.name, data=e))

            # --- 没有调用工具，结束流程 ---
            if not response.tool_calls:
                return response.content

            # --- 准备执行工具 ---
            for tool_call in response.tool_calls:
                tool = self._toolkit.get_tool(tool_call.name)
                # --- 工具不存在 ---
                if not tool:
                    self.broadcast(AgentEvent(type=AgentEventType.RUN_ERROR, agent_name=self._config.name,
                                              content=f"调用了不存在的工具 {tool_call.name}",
                                              data=ToolNotFountError(f"调用了不存在的工具 {tool_call.name}")))
                    # --- 添加反馈信息给模型 ---
                    user_msg = Message(role="user", content=f"调用了不存在的工具 {tool_call.name}")
                    self._context.append(user_msg)
                    self.broadcast(
                        AgentEvent(type=AgentEventType.UPDATE_CONTEXT, agent_name=self.get_name(),
                                   data=user_msg))
                    continue

                # --- 开始调用工具 ---
                self.broadcast(
                    AgentEvent(type=AgentEventType.TOOL_CALL_START, agent_name=self._config.name,
                               content=f"开始运行工具 {tool.name}",
                               data={"tool": tool, "tool_call": tool_call}))

                tool_response = tool.run(tool_call.arguments)

                # --- 工具调用结束 ---
                self.broadcast(
                    AgentEvent(type=AgentEventType.TOOL_CALL_END, content="工具执行结束",
                               agent_name=self._config.name,
                               data={"tool": tool, "tool_call": tool_call, "response": tool_response}))

                # --- 添加工具消息 ---
                tool_response.name = tool_call.name
                tool_response.call_id = tool_call.call_id
                tool_msg = Message(role="tool", content=tool_response)
                self._context.append(tool_msg)
                self.broadcast(
                    AgentEvent(type=AgentEventType.UPDATE_CONTEXT, agent_name=self.get_name(), data=tool_msg))

                # --- 调用的是“结束”工具，且工具运行成功 ---
                if tool.name == "Finish" and tool_response.status == ToolStatus.SUCCESS:
                    self.broadcast(
                        AgentEvent(type=AgentEventType.RUN_FINISHED, content=tool_response.content,
                                   agent_name=self._config.name))
                    return tool_response.content

            # --- 步骤 n 结束 ---
            self.broadcast(AgentEvent(type=AgentEventType.STEP_FINISHED, agent_name=self._config.name,
                                      content=f"步骤 {current_step} 结束",
                                      data=current_step))

    async def ainvoke(self, content: str, params: Optional[LLMRequestParams] = None) -> str | None:
        # --- 开始新任务 ---
        await self.async_broadcast(
            AgentEvent(type=AgentEventType.RUN_STARTED, agent_name=self._config.name, content="新任务",
                       data={"user_input": content}))

        # --- 更新上下文，保存用户消息 ---
        user_msg = Message(role="user", content=content)
        self._context.append(user_msg)
        await self.async_broadcast(AgentEvent(type=AgentEventType.UPDATE_CONTEXT,
                                              agent_name=self.get_name(), data=user_msg))

        current_step = 0
        while True:
            # --- 步数检查 ---
            if current_step == self._config.max_step:
                await self.async_broadcast(AgentEvent(type=AgentEventType.RUN_FINISHED, content="超出限定步数",
                                                      agent_name=self._config.name,
                                                      data={"current_step": current_step,
                                                            "max_step": self._config.max_step}))
                return "抱歉，我没有在限定步数内完成任务。"

            # --- 开始新步骤 ---
            current_step += 1
            await self.async_broadcast(AgentEvent(type=AgentEventType.STEP_STARTED, agent_name=self._config.name,
                                                  content=f"开始第 {current_step} 步",
                                                  data=current_step))

            # --- 请求模型 ---
            response = LLMToolResponse(model=self._config.name)
            try:
                await self.async_broadcast(
                    AgentEvent(type=AgentEventType.REQUEST, agent_name=self._config.name, content="开始请求模型",
                               data=self._llm.get_name()))

                response = await self._llm.ainvoke_with_tools(
                    self._context, self._toolkit.tool_list(), params
                )

                # --- 模型响应成功 ---
                await self.async_broadcast(
                    AgentEvent(type=AgentEventType.REPLY, agent_name=self._config.name,
                               content=f"模型 {response.model} 响应成功", data=response))

                # --- 更新上下文，保存模型消息 ---
                assistant_msg = Message(role="assistant", content=response)
                self._context.append(assistant_msg)
                await self.async_broadcast(AgentEvent(type=AgentEventType.UPDATE_CONTEXT,
                                                      agent_name=self.get_name(), data=assistant_msg))
            # --- 请求模型失败 ---
            except LLMClientError as e:
                # --- 处理异常 ---
                await self.async_broadcast(AgentEvent(type=AgentEventType.RUN_ERROR, content=e.message,
                                                      agent_name=self._config.name, data=e))

            # --- 没有调用工具，结束流程 ---
            if not response.tool_calls:
                return response.content

            # --- 准备执行工具 ---
            for tool_call in response.tool_calls:
                tool = self._toolkit.get_tool(tool_call.name)
                # --- 工具不存在 ---
                if not tool:
                    await self.async_broadcast(AgentEvent(type=AgentEventType.RUN_ERROR, agent_name=self._config.name,
                                                          content=f"调用了不存在的工具 {tool_call.name}",
                                                          data=ToolNotFountError(
                                                              f"调用了不存在的工具 {tool_call.name}")))
                    # --- 添加反馈信息给模型 ---
                    user_msg = Message(role="user", content=f"调用了不存在的工具 {tool_call.name}")
                    self._context.append(user_msg)
                    await self.async_broadcast(
                        AgentEvent(type=AgentEventType.UPDATE_CONTEXT, agent_name=self.get_name(),
                                   data=user_msg))
                    continue

                # --- 开始调用工具 ---
                await self.async_broadcast(
                    AgentEvent(type=AgentEventType.TOOL_CALL_START, agent_name=self._config.name,
                               content=f"开始运行工具 {tool.name}",
                               data={"tool": tool, "tool_call": tool_call}))

                tool_response = await tool.arun(tool_call.arguments)

                # --- 工具调用结束 ---
                await self.async_broadcast(
                    AgentEvent(type=AgentEventType.TOOL_CALL_END, content="工具执行结束",
                               agent_name=self._config.name,
                               data={"tool": tool, "tool_call": tool_call, "response": tool_response}))

                # --- 添加工具消息 ---
                tool_response.name = tool_call.name
                tool_response.call_id = tool_call.call_id
                tool_msg = Message(role="tool", content=tool_response)
                self._context.append(tool_msg)
                await self.async_broadcast(
                    AgentEvent(type=AgentEventType.UPDATE_CONTEXT, agent_name=self.get_name(), data=tool_msg))

                # --- 调用的是“结束”工具，且工具运行成功 ---
                if tool.name == "Finish" and tool_response.status == ToolStatus.SUCCESS:
                    await self.async_broadcast(
                        AgentEvent(type=AgentEventType.RUN_FINISHED, content=tool_response.content,
                                   agent_name=self._config.name))
                    return tool_response.content

            # --- 步骤 n 结束 ---
            await self.async_broadcast(AgentEvent(type=AgentEventType.STEP_FINISHED, agent_name=self._config.name,
                                                  content=f"步骤 {current_step} 结束",
                                                  data=current_step))

    def stream(self, content: str, params: Optional[LLMRequestParams] = None) -> str | None:
        # --- 开始新任务 ---
        self.broadcast(
            AgentEvent(type=AgentEventType.RUN_STARTED, agent_name=self._config.name, content="新任务",
                       data={"user_input": content}))

        # --- 更新上下文，保存用户消息 ---
        user_msg = Message(role="user", content=content)
        self._context.append(user_msg)
        self.broadcast(AgentEvent(type=AgentEventType.UPDATE_CONTEXT,
                                  agent_name=self.get_name(), data=user_msg))
        current_step = 0
        while True:

            # --- 步数检查 ---
            if current_step == self._config.max_step:
                self.broadcast(AgentEvent(type=AgentEventType.RUN_FINISHED, content="超出限定步数",
                                          agent_name=self._config.name,
                                          data={"current_step": current_step,
                                                "max_step": self._config.max_step}))
                return "抱歉，我没有在限定步数内完成任务。"

            # --- 开始新步骤 ---
            current_step += 1
            self.broadcast(AgentEvent(type=AgentEventType.STEP_STARTED, agent_name=self._config.name,
                                      content=f"开始第 {current_step} 步",
                                      data=current_step))

            # --- 请求模型 ---
            model = ""
            response = None
            try:
                self.broadcast(
                    AgentEvent(type=AgentEventType.REQUEST, agent_name=self._config.name, content="开始请求模型",
                               data=self._llm.get_name()))

                response_iterator = self._llm.stream_with_tools(
                    self._context, self._toolkit.tool_list(), params
                )

                is_reasoning_ing = False
                is_content_ing = False
                for chunk in response_iterator:

                    if chunk.reasoning_content and not is_reasoning_ing:
                        if len(chunk.reasoning_content) > 0:
                            is_reasoning_ing = True
                            self.broadcast(
                                AgentEvent(type=AgentEventType.STREAM_REASONING_START, agent_name=self._config.name,
                                           content=chunk.reasoning_content))

                    if is_reasoning_ing:
                        if not chunk.reasoning_content:
                            is_reasoning_ing = False
                            self.broadcast(
                                AgentEvent(type=AgentEventType.STREAM_REASONING_END, agent_name=self._config.name,
                                           content=getattr(response, "reasoning_content", "")))

                    if chunk.content and not is_content_ing:
                        if len(chunk.content) > 0:
                            is_content_ing = True
                            self.broadcast(
                                AgentEvent(type=AgentEventType.STREAM_CONTENT_START, agent_name=self._config.name,
                                           content=chunk.content))

                    if is_content_ing:
                        if not chunk.content:
                            is_content_ing = False
                            self.broadcast(
                                AgentEvent(type=AgentEventType.STREAM_CONTENT_END, agent_name=self._config.name,
                                           content=getattr(response, "content", "")))

                    model = chunk.model
                    if not response:
                        response = chunk
                    else:
                        response += chunk
                    if chunk.reasoning_content:
                        self.broadcast(
                            AgentEvent(type=AgentEventType.STREAM_REASONING_ING, agent_name=self._config.name,
                                       content=chunk.reasoning_content))
                    if chunk.content and isinstance(chunk.content, str):
                        self.broadcast(AgentEvent(type=AgentEventType.STREAM_CONTENT_ING, agent_name=self._config.name,
                                                  content=chunk.content))

            # --- 请求模型失败 ---
            except LLMClientError as e:
                # --- 处理异常 ---
                self.broadcast(AgentEvent(type=AgentEventType.RUN_ERROR, content=e.message,
                                          agent_name=self._config.name, data=e))
            if not response:
                return "抱歉，我无法回答这个问题"

            if not response.usage: response.usage = Usage()
            # --- 模型响应成功 ---
            self.broadcast(
                AgentEvent(type=AgentEventType.REPLY, agent_name=self._config.name,
                           content=f"模型 {model} 响应成功", data=response))

            # --- 更新上下文，保存模型消息 ---
            assistant_msg = Message(role="assistant", content=response)
            self._context.append(assistant_msg)
            self.broadcast(AgentEvent(type=AgentEventType.UPDATE_CONTEXT,
                                      agent_name=self.get_name(), data=assistant_msg))

            # --- 没有调用工具，结束流程---
            if not response.tool_calls:
                return response.content

            # --- 准备执行工具 ---
            for tool_call in response.tool_calls:
                tool = self._toolkit.get_tool(tool_call.name)
                # --- 工具不存在 ---
                if not tool:
                    self.broadcast(AgentEvent(type=AgentEventType.RUN_ERROR, agent_name=self._config.name,
                                              content=f"调用了不存在的工具 {tool_call.name}",
                                              data=ToolNotFountError(f"调用了不存在的工具 {tool_call.name}")))
                    # --- 添加反馈信息给模型 ---
                    user_msg = Message(role="user", content=f"调用了不存在的工具 {tool_call.name}")
                    self._context.append(user_msg)
                    self.broadcast(
                        AgentEvent(type=AgentEventType.UPDATE_CONTEXT, agent_name=self.get_name(),
                                   data=user_msg))
                    continue

                # --- 开始调用工具 ---
                self.broadcast(
                    AgentEvent(type=AgentEventType.TOOL_CALL_START, agent_name=self._config.name,
                               content=f"开始运行工具 {tool.name}",
                               data={"tool": tool, "tool_call": tool_call}))

                tool_response = tool.run(tool_call.arguments)

                # --- 工具调用结束 ---
                self.broadcast(
                    AgentEvent(type=AgentEventType.TOOL_CALL_END, content="工具执行结束",
                               agent_name=self._config.name,
                               data={"tool": tool, "tool_call": tool_call, "response": tool_response}))

                # --- 添加工具消息 ---
                tool_response.name = tool_call.name
                tool_response.call_id = tool_call.call_id
                tool_msg = Message(role="tool", content=tool_response)
                self._context.append(tool_msg)
                self.broadcast(
                    AgentEvent(type=AgentEventType.UPDATE_CONTEXT, agent_name=self.get_name(), data=tool_msg))

                # --- 调用的是“结束”工具，且工具运行成功 ---
                if tool.name == "Finish" and tool_response.status == ToolStatus.SUCCESS:
                    self.broadcast(
                        AgentEvent(type=AgentEventType.RUN_FINISHED, content=tool_response.content,
                                   agent_name=self._config.name))
                    return tool_response.content

            # --- 步骤 n 结束 ---
            self.broadcast(AgentEvent(type=AgentEventType.STEP_FINISHED, agent_name=self._config.name,
                                      content=f"步骤 {current_step} 结束",
                                      data=current_step))

    async def astream(self, content: str, params: Optional[LLMRequestParams] = None) -> str | None:
        # --- 开始新任务 ---
        await self.async_broadcast(
            AgentEvent(type=AgentEventType.RUN_STARTED, agent_name=self._config.name, content="新任务",
                       data={"user_input": content}))

        # --- 更新上下文，保存用户消息 ---
        user_msg = Message(role="user", content=content)
        self._context.append(user_msg)
        await self.async_broadcast(AgentEvent(type=AgentEventType.UPDATE_CONTEXT,
                                              agent_name=self.get_name(), data=user_msg))
        current_step = 0

        while True:
            # --- 步数检查 ---
            if current_step == self._config.max_step:
                await self.async_broadcast(AgentEvent(type=AgentEventType.RUN_FINISHED, content="超出限定步数",
                                                      agent_name=self._config.name,
                                                      data={"current_step": current_step,
                                                            "max_step": self._config.max_step}))
                return "抱歉，我没有在限定步数内完成任务。"

            # --- 开始新步骤 ---
            current_step += 1
            await self.async_broadcast(AgentEvent(type=AgentEventType.STEP_STARTED, agent_name=self._config.name,
                                                  content=f"开始第 {current_step} 步",
                                                  data=current_step))

            # --- 请求模型 ---
            model = ""
            response: Optional[LLMToolResponse] = None
            try:
                await self.async_broadcast(
                    AgentEvent(type=AgentEventType.REQUEST, agent_name=self._config.name, content="开始请求模型",
                               data=self._llm.get_name()))

                response_iterator = self._llm.astream_with_tools(
                    self._context, self._toolkit.tool_list(), params
                )

                is_reasoning_ing = False
                is_content_ing = False
                # noinspection PyTypeChecker
                async for chunk in response_iterator:

                    if chunk.reasoning_content and not is_reasoning_ing:
                        is_reasoning_ing = True
                        await self.async_broadcast(
                            AgentEvent(type=AgentEventType.STREAM_REASONING_START, agent_name=self._config.name,
                                       content=chunk.reasoning_content))

                    if chunk.content and not is_content_ing:
                        is_reasoning_ing = False
                        await self.async_broadcast(
                            AgentEvent(type=AgentEventType.STREAM_REASONING_END, agent_name=self._config.name,
                                       content=getattr(response, "reasoning_content", "")))
                        is_content_ing = True
                        await self.async_broadcast(
                            AgentEvent(type=AgentEventType.STREAM_CONTENT_START, agent_name=self._config.name,
                                       content=chunk.content))

                    if is_content_ing and not chunk.content:
                        is_content_ing = False
                        await self.async_broadcast(
                            AgentEvent(type=AgentEventType.STREAM_CONTENT_END, agent_name=self._config.name,
                                       content=getattr(response, "content", "")))

                    model = chunk.model
                    if not response:
                        response = chunk
                    else:
                        response += chunk
                    if chunk.reasoning_content:
                        await self.async_broadcast(
                            AgentEvent(type=AgentEventType.STREAM_REASONING_ING, agent_name=self._config.name,
                                       content=chunk.reasoning_content))
                    if chunk.content and isinstance(chunk.content, str):
                        await self.async_broadcast(
                            AgentEvent(type=AgentEventType.STREAM_CONTENT_ING, agent_name=self._config.name,
                                       content=chunk.content))

            # --- 请求模型失败 ---
            except LLMClientError as e:
                # --- 处理异常 ---
                await self.async_broadcast(AgentEvent(type=AgentEventType.RUN_ERROR, content=e.message,
                                                      agent_name=self._config.name, data=e))
            if not response:
                return "抱歉，我无法回答这个问题"

            if not response.usage: response.usage = Usage()
            # --- 模型响应成功 ---
            await self.async_broadcast(
                AgentEvent(type=AgentEventType.REPLY, agent_name=self._config.name,
                           content=f"模型 {model} 响应成功", data=response))

            # --- 更新上下文，保存模型消息 ---
            assistant_msg = Message(role="assistant", content=response)
            self._context.append(assistant_msg)
            await self.async_broadcast(AgentEvent(type=AgentEventType.UPDATE_CONTEXT,
                                                  agent_name=self.get_name(), data=assistant_msg))

            # --- 没有调用工具，结束任务 ---
            if not response.tool_calls:
                return response.content

            # --- 准备执行工具 ---
            for tool_call in response.tool_calls:
                tool = self._toolkit.get_tool(tool_call.name)
                # --- 工具不存在 ---
                if not tool:
                    await self.async_broadcast(AgentEvent(type=AgentEventType.RUN_ERROR, agent_name=self._config.name,
                                                          content=f"调用了不存在的工具 {tool_call.name}",
                                                          data=ToolNotFountError(
                                                              f"调用了不存在的工具 {tool_call.name}")))
                    # --- 添加反馈信息给模型 ---
                    user_msg = Message(role="user", content=f"调用了不存在的工具 {tool_call.name}")
                    self._context.append(user_msg)
                    await self.async_broadcast(
                        AgentEvent(type=AgentEventType.UPDATE_CONTEXT, agent_name=self.get_name(),
                                   data=user_msg))
                    continue

                # --- 开始调用工具 ---
                await self.async_broadcast(
                    AgentEvent(type=AgentEventType.TOOL_CALL_START, agent_name=self._config.name,
                               content=f"开始运行工具 {tool.name}",
                               data={"tool": tool, "tool_call": tool_call}))

                tool_response = await tool.arun(tool_call.arguments)

                # --- 工具调用结束 ---
                await self.async_broadcast(
                    AgentEvent(type=AgentEventType.TOOL_CALL_END, content="工具执行结束",
                               agent_name=self._config.name,
                               data={"tool": tool, "tool_call": tool_call, "response": tool_response}))

                # --- 添加工具消息 ---
                tool_response.name = tool_call.name
                tool_response.call_id = tool_call.call_id
                tool_msg = Message(role="tool", content=tool_response)
                self._context.append(tool_msg)
                await self.async_broadcast(
                    AgentEvent(type=AgentEventType.UPDATE_CONTEXT, agent_name=self.get_name(), data=tool_msg))

                # --- 调用的是“结束”工具，且工具运行成功 ---
                if tool.name == "Finish" and tool_response.status == ToolStatus.SUCCESS:
                    await self.async_broadcast(
                        AgentEvent(type=AgentEventType.RUN_FINISHED, content=tool_response.content,
                                   agent_name=self._config.name))
                    return tool_response.content

            # --- 步骤 n 结束 ---
            await self.async_broadcast(AgentEvent(type=AgentEventType.STEP_FINISHED, agent_name=self._config.name,
                                                  content=f"步骤 {current_step} 结束",
                                                  data=current_step))
