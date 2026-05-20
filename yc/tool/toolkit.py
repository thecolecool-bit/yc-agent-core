import asyncio
from typing import Dict, Callable, List, Union
from fastmcp import Client
from yc.common.exceptions import AppError
from yc.tool.base import Tool
from yc.tool.impl import FunctionTool, McpTool


class Toolkit:
    def __init__(self):
        self._tools: Dict[str, Dict[str, Tool]] = {}

    def add_tool(self, tool: Union[Tool, Callable], namespace: str):
        if namespace not in self._tools:
            self._tools[namespace] = {}
        if isinstance(tool, Tool):
            if tool.name in self._tools:
                raise ValueError(f"Tool {tool.name} already exists")
            tool.namespace = namespace
            self._tools[namespace][tool.name] = tool

        if isinstance(tool, Callable):
            tool_name = getattr(tool, "_tool_name")
            tool_description = getattr(tool, "_tool_description")
            self._tools[namespace][tool_name] = FunctionTool(tool_name, namespace, tool_description, tool)

    def add_tools(self, tools: List[Union[Tool, Callable]], namespace: str):
        for tool in tools:
            self.add_tool(tool, namespace)

    def add_mcp(self, mcp_client: Client):
        async def _add():
            async with mcp_client:
                try:
                    tools = await mcp_client.list_tools()
                    if mcp_client.name in self._tools:
                        raise AppError(f"重复添加mcp服务：{mcp_client.name}，已跳过")
                    self._tools[mcp_client.name] = {}
                    for tool in tools:
                        self._tools[mcp_client.name][tool.name] = McpTool(
                            mcp_client,
                            tool.name,
                            mcp_client.name,
                            tool.description if tool.description else "",
                            tool.inputSchema
                        )
                except Exception as e:
                    raise AppError("Load MCP Tools Error", detail={"error": e})

        asyncio.run(_add())

    async def add_mcp_sync(self, mcp_client: Client):
        async with mcp_client:
            try:
                tools = await mcp_client.list_tools()
                if mcp_client.name in self._tools:
                    raise AppError(f"重复添加mcp服务：{mcp_client.name}，已跳过")
                self._tools[mcp_client.name] = {}
                for tool in tools:
                    self._tools[mcp_client.name][tool.name] = McpTool(
                        mcp_client,
                        tool.name,
                        mcp_client.name,
                        tool.description if tool.description else "",
                        tool.inputSchema
                    )
            except Exception as e:
                raise AppError("Load MCP Tools Error", detail={"error": e})

    def get_tool(self, tool_name: str) -> Tool | None:
        for namespace, tools in self._tools.items():
            for tool in tools.values():
                if namespace + "_" + tool.name == tool_name:
                    return tool
        return None

    def tool_list(self) -> List[Tool]:
        tool_list = []
        for tools in self._tools.values():
            tool_list.extend(tools.values())
        return tool_list

    def remove_tool(self, tool_name: str) -> None:
        for namespace, tools in self._tools.items():
            for tool in tools.values():
                if namespace + "_" + tool.name == tool_name:
                    del self._tools[namespace][tool.name]
