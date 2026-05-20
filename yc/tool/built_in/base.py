import base64
import subprocess
from pathlib import Path
import requests
from yc.common.exceptions import ToolExecuteError, ToolTimeoutError, ToolError, ToolParametersError
from yc.skill.skill_loader import skill_loader
from yc.tool.base import agent_tool


@agent_tool(name="execute_cli", description="执行 CLI 命令")
async def execute_cli(command: str, timeout: int = 60) -> dict:
    """
    执行 CLI 命令
    :param command: 要执行的命令（例如：python -v）
    :param timeout: 超时时间（s）
    """
    try:
        # shell=True 允许执行包含管道、重定向等的复杂命令
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        # 返回一个结构化的响应，便于 AI 理解
        return {
            "success": result.returncode == 0,
            "return_code": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "command_executed": command,
        }
    except subprocess.TimeoutExpired:
        raise ToolTimeoutError()
    except Exception as e:
        raise ToolExecuteError(str(e))


@agent_tool(name="web_download", description="从网络加载数据")
async def web_download(url, timeout: int = 60) -> str:
    """
    从网络加载数据
    :param url: 链接
    :param timeout: 超时时间
    """
    try:
        # 发送 GET 请求
        response = requests.get(url, timeout=timeout)
        # 检查 HTTP 状态码，如果是 404 或 500 会抛出异常
        response.raise_for_status()
        return response.content.decode("utf-8")
    except requests.exceptions.RequestException as e:
        raise e


@agent_tool(name="execute_python_script", description="执行python脚本")
async def execute_python_script(script: str) -> str | None:
    """
    执行python脚本
    :param script: python代码（代码必须包含应有的换行、缩进。在你的脚本中，必须声明一个result变量来存储脚本的返回值)
    """
    if not script:
        raise ToolParametersError()
    try:
        _global = globals().copy()
        exec(script, globals=_global, locals=_global)
        return _global.get("result")
    except Exception as e:
        raise ToolExecuteError(str(e))


@agent_tool(name='learn_skill', description="深入学习指定技能")
async def learn_skill(skill_name: str):
    """
    学习技能
    :param skill_name: 技能名
    """
    skill = skill_loader.load_skill(skill_name)
    if not skill:
        raise ToolExecuteError(f"技能[{skill_name}]不存在")
    return skill


@agent_tool(name='write', description="写入数据到本地")
async def write(output_path: str, base64_data: str):
    """
    保存数据到本地
    :param output_path: 包含文件名称在内的全路径（注意：不同操作系统的路径表示有所区别）
    :param base64_data: 数据（注意：data必须utf-8类型的的base64编码后的字符串，以避免json格式错误）
    :return:
    """
    try:
        decoded_data = base64.b64decode(base64_data).decode('utf-8')
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(decoded_data)
        return "success"
    except Exception as e:
        raise ToolExecuteError(str(e))


@agent_tool(name='read', description="从本地读取文件")
async def read(file_path: str):
    """
    从本地读取文件
    :param file_path: 包含文件名称在内的全路径（注意：不同操作系统的路径表示有所区别）
    """
    try:
        path = Path(file_path)
        if not path.exists():
            raise ToolExecuteError(f"文件 {file_path} 不存在")
        return path.read_bytes()
    except OSError as e:
        raise ToolError(str(e))
