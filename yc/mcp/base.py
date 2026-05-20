import json
from typing import List

from fastmcp import Client
from fastmcp.client import StdioTransport, StreamableHttpTransport

from yc.common.exceptions import AppError
from yc.schemas.mcp import McpClientConfig


class McpClientFactory:

    def __init__(self):
        self.mcp_config_filename = ".mcp.json"
        self.mcp_namespace = "mcp_servers"

    def load_clients(self) -> List[Client]:
        mcp_list = []
        import os
        file_path = os.path.join(os.getcwd(), self.mcp_config_filename)
        try:
            with open(file_path) as f:
                file_config = json.load(f)
        except FileNotFoundError:
            raise AppError("MCP Client Config File Not Found")
        except json.decoder.JSONDecodeError:
            raise AppError("MCP Client Config Json parse error")
        except PermissionError:
            raise AppError("MCP Client Config Permission Error")
        except Exception as e:
            raise AppError("MCP Client Config load error", detail={"error": e})
        config_dict = file_config[self.mcp_namespace]
        for mcp_name, config in config_dict.items():
            config["name"] = mcp_name
            mcp_config = McpClientConfig(**config)
            try:
                f = open("docker.txt", "w")
                if mcp_config.type == "stdio":
                    client = Client(name=mcp_name,
                                    transport=StdioTransport(command=mcp_config.command if mcp_config.command else "",
                                                             args=mcp_config.args if mcp_config.args else [],
                                                             env=mcp_config.env, log_file=f))
                    mcp_list.append(client)
                elif mcp_config.type == "http":
                    client = Client(name=mcp_name,
                                    transport=StreamableHttpTransport(url=mcp_config.url if mcp_config.url else "",
                                                                      headers=mcp_config.headers if mcp_config.headers else {}))
                    mcp_list.append(client)
            except Exception as e:
                raise AppError("FastMCP Client Create Error", detail={"error": e})
        return mcp_list


mcp_factory = McpClientFactory()
