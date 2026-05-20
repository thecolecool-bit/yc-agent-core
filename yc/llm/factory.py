from typing import Dict, Type, Optional

from yc.common.config import ConfigLoader, EnvConfigLoader
from yc.common.exceptions import AppError
from yc.llm.ark_client import ArkClient
from yc.llm.base import LLMClient
from yc.llm.minimax_client import MiniMaxClient
from yc.llm.openai_client import OpenAIClient
from yc.llm.qwen_client import QwenClient

from yc.schemas.llm import LLMConfig, OpenAIConfig


class ClientFactory:
    def __init__(self, config_loader: ConfigLoader[LLMConfig]) -> None:
        self._config_register: Dict[str, Type[LLMConfig]] = {}
        self._client_register: Dict[str, Type[LLMClient]] = {}
        self._config_center = config_loader

    def register(
            self,
            model_name,
            config_class: Optional[Type[LLMConfig]] = None,
            client_class: Optional[Type[LLMClient]] = None,
    ) -> None:
        if not config_class:
            config_class = OpenAIConfig
        if not client_class:
            client_class = OpenAIClient
        self._config_register[model_name] = config_class
        self._client_register[model_name] = client_class

    def create_client(self, model_name: str) -> LLMClient:
        config_type = self._config_register.get(model_name)
        if not config_type:
            self._config_register[model_name] = OpenAIConfig
        client_type = self._client_register.get(model_name)
        if not client_type:
            self._client_register[model_name] = OpenAIClient
        config = self._config_center.load_config_for_class("MODELS", model_name, self._config_register.get(model_name))
        if not config:
            raise AppError(f"创建模型客户端失败，原因：找不到此模型的配置文件 {model_name}")
        if "ark" in config.client.lower():
            self._client_register[model_name] = ArkClient
        if "qwen" in config.client.lower():
            self._client_register[model_name] = QwenClient
        if "minimax" in config.client.lower():
            self._client_register[model_name] = MiniMaxClient
        return self._client_register.get(model_name)(config)


llm_factory = ClientFactory(config_loader=EnvConfigLoader[LLMConfig]())
