import json
from typing import Optional


class AppError(Exception):
    def __init__(self, message: str, code: int = 500, detail: Optional[dict] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail

    def to_json(self) -> str:
        return json.dumps({
            "code": self.code,
            "message": self.message,
            "detail": self.detail
        }, ensure_ascii=False)


class LLMClientError(AppError):
    def __init__(self, message: str = "大模型调用异常", code: int = 5000, detail: Optional[dict] = None):
        super().__init__(
            code=code,
            message=message,
            detail=detail
        )


class LLMNotFoundError(LLMClientError):
    def __init__(self, message: str = "找不到大模型服务，请检查 baseurl 地址和模型名称，并确定模型目前可用",
                 code: int = 5001):
        super().__init__(
            code=code,
            message=message
        )


class BadRequestError(LLMClientError):
    def __init__(self, message: str = "大模型请求 Messages 格式有误，需更换适配的 Client", code: int = 5002):
        super().__init__(
            code=code,
            message=message
        )


class AuthenticationError(LLMClientError):
    def __init__(self, message: str = "API key 错误，认证失败", code: int = 5003):
        super().__init__(
            code=code,
            message=message
        )


class InsufficientBalanceError(LLMClientError):
    def __init__(self, message: str = "模型不可用，无权限或余额不足", code: int = 5004):
        super().__init__(
            code=code,
            message=message
        )


class ParameterError(LLMClientError):
    def __init__(self, message: str = "请求体参数错误", code: int = 5005):
        super().__init__(
            code=code,
            message=message
        )


class RateLimitError(LLMClientError):
    def __init__(self, message: str = "请求速率（TPM 或 RPM）达到上限", code: int = 5006):
        super().__init__(
            code=code,
            message=message
        )


class ServerFaultError(LLMClientError):
    def __init__(self, message: str = "大模型服务器内部故障", code: int = 5007):
        super().__init__(
            code=code,
            message=message
        )


class ServerBusyError(LLMClientError):
    def __init__(self, message: str = "大模型服务器繁忙", code: int = 5008):
        super().__init__(
            code=code,
            message=message
        )


class APiConnectionError(LLMClientError):

    def __init__(self, message: str = "无法连接到大模型服务，请检查网络或 baseurl 地址", code: int = 5009):
        super().__init__(
            code=code,
            message=message
        )


class ApiTimeoutError(LLMClientError):

    def __init__(self, message: str = "大模型响应超时，请检查网络或重新再试", code: int = 5010):
        super().__init__(
            code=code,
            message=message
        )


class ToolError(AppError):
    def __init__(self, message: str = "工具调用失败", code: int = 6000, detail: Optional[dict] = None):
        super().__init__(
            code=code,
            message=message,
            detail=detail
        )


class ToolParametersError(ToolError):
    def __init__(self, message: str = "工具参数有误", code: int = 6001):
        super().__init__(
            code=code,
            message=message
        )


class ToolExecuteError(ToolError):
    def __init__(self, message: str = "工具内部发生异常", code: int = 6002):
        super().__init__(
            code=code,
            message=message
        )


class ToolTimeoutError(ToolError):
    def __init__(self, message: str = "工具调用超时", code: int = 6003):
        super().__init__(
            code=code,
            message=message
        )


class ToolNotFountError(ToolError):
    def __init__(self, message: str = "工具不存在", code: int = 6003):
        super().__init__(
            code=code,
            message=message
        )


class AgentError(AppError):
    def __init__(self, message: str = "Agent执行异常", code: int = 8000, detail: Optional[dict] = None):
        super().__init__(
            code=code,
            message=message,
            detail=detail
        )
