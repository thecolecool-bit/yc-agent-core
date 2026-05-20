import asyncio
import inspect
import json
import re
from typing import Any, List, Callable
import time
from fastmcp import Client
from yc.common.exceptions import ToolError, AppError
from yc.tool.base import Tool
from yc.schemas.tool import ToolParam, ToolResponse


class FunctionTool(Tool):
    def __init__(self, name: str, namespace: str, description: str, function: Callable):
        self._function = function
        super().__init__(name, namespace, description)

    def run(self, parameters: str) -> ToolResponse:
        start = time.time()
        try:
            if not inspect.iscoroutinefunction(self._function):
                result = self._function(**json.loads(parameters))
            else:
                result = asyncio.run(self._function(**json.loads(parameters)))
            end = time.time()
            return ToolResponse.success(
                name=self.name,
                content=result,
                elapsed_ms=int((end - start) * 1000),
            )
        except AppError as exc:
            end = time.time()
            return ToolResponse.error(
                name=self.name,
                content=exc.message,
                elapsed_ms=int((end - start) * 1000)
            )
        except Exception as exc:
            end = time.time()
            return ToolResponse.error(
                name=self.name,
                content=str(exc),
                elapsed_ms=int((end - start) * 1000)
            )

    async def arun(self, parameters: str) -> ToolResponse:
        start = time.time()
        try:
            if inspect.iscoroutinefunction(self._function):
                result = await self._function(**json.loads(parameters))
            else:
                result = await asyncio.to_thread(self._function, **json.loads(parameters))
            end = time.time()
            return ToolResponse.success(
                name=self.name,
                content=result,
                elapsed_ms=int((end - start) * 1000),
            )
        except AppError as exc:
            end = time.time()
            return ToolResponse.error(
                name=self.name,
                content=exc.message,
                elapsed_ms=int((end - start) * 1000)
            )
        except Exception as exc:
            end = time.time()
            return ToolResponse.error(
                name=self.name,
                content=str(exc),
                elapsed_ms=int((end - start) * 1000)
            )

    def generated_schema(self):
        properties = {}
        for param in self.get_parameters():
            properties[param.name] = {
                "type": param.type,
                "description": param.description,
            }
        schema = {
            "type": "function",
            "function": {
                "name": f"{self.namespace}_{self.name}",
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": [
                        param.name for param in self.get_parameters() if param.required
                    ],
                },
            },
        }
        return json.dumps(schema, ensure_ascii=False)

    def get_parameters(self) -> List[ToolParam]:
        parameters = []
        sig = inspect.signature(self._function)
        for name, param in sig.parameters.items():
            required = param.default is inspect.Parameter.empty
            default = None if required else param.default
            parameters.append(
                ToolParam(
                    name=name,
                    description=self._extract_param_description(name),
                    type=self._python_type_to_tool_type(param.annotation),
                    required=required,
                    default=default,
                )
            )
        return parameters

    def _extract_param_description(self, param_name) -> str:
        doc = inspect.getdoc(self._function)
        if not doc:
            return ""
        pattern = rf":param\s+{param_name}:\s*(.*)"
        match = re.search(pattern, doc)
        if match:
            return match.group(1).strip()
        return ""

    @staticmethod
    def _python_type_to_tool_type(py_type) -> str:
        """将 Python 类型转换为工具类型字符串"""
        # 处理泛型类型
        origin = getattr(py_type, "__origin__", None)
        if origin is not None:
            if origin is list:
                return "array"
            elif origin is dict:
                return "object"

        # 处理基本类型
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }
        return type_map.get(py_type, "string")


class McpTool(Tool):
    def __init__(
            self, client: Client, name: str, namespace: str, description: str, input_schema: dict[str, Any]
    ):
        self._client = client
        self.input_schema = input_schema
        super().__init__(name, namespace, description)

    def run(self, parameters: str) -> ToolResponse:
        start = time.time()
        try:
            async def _call():
                async with self._client:
                    return await self._client.call_tool(self.name, json.loads(parameters))

            result = asyncio.run(_call())
            if result.is_error:
                raise ToolError(str(result))
            end = time.time()
            return ToolResponse.success(
                name=self.name,
                content=result,
                elapsed_ms=int((end - start) * 1000),
            )
        except AppError as exc:
            end = time.time()
            return ToolResponse.error(
                name=self.name,
                content=exc.message,
                elapsed_ms=int((end - start) * 1000),
            )
        except Exception as exc:
            end = time.time()
            return ToolResponse.error(
                name=self.name,
                content=str(exc),
                elapsed_ms=int((end - start) * 1000),
            )

    async def arun(self, parameters: str) -> ToolResponse:
        start = time.time()
        try:
            async with self._client:
                result = await self._client.call_tool(self.name, json.loads(parameters))
                if result.is_error:
                    raise ToolError(str(result))
                end = time.time()
                return ToolResponse.success(
                    name=self.name,
                    content=result,
                    elapsed_ms=int((end - start) * 1000),
                )
        except AppError as exc:
            end = time.time()
            return ToolResponse.error(
                name=self.name,
                content=exc.message,
                elapsed_ms=int((end - start) * 1000),
            )
        except Exception as exc:
            end = time.time()
            return ToolResponse.error(
                name=self.name,
                content=str(exc),
                elapsed_ms=int((end - start) * 1000),
            )

    def generated_schema(self):
        schema = {
            "type": "function",
            "function": {
                "name": f"{self.namespace}_{self.name}",
                "description": self.description,
                "parameters": self.input_schema,
            },
        }
        return json.dumps(schema, ensure_ascii=False)

    def get_parameters(self) -> List[ToolParam]:
        parameters = []
        properties = self.input_schema["properties"]
        if not properties:
            return parameters
        required = self.input_schema.get("required", [])
        for param in properties:
            items = properties.get(param).get("items", None)
            enum = properties.get(param).get("enum", None)
            tp = properties.get(param).get("type", None)
            description = properties.get(param).get("description", None)
            parameters.append(
                ToolParam(
                    name=param,
                    description=description,
                    type=tp,
                    required=(True if param in required else False),
                    enum=enum,
                    items=items,
                )
            )
        return parameters
