import json
import os
from abc import ABC, abstractmethod
from typing import Type, Optional

from yc.common.exceptions import AppError


class ConfigLoader[Config](ABC):
    @abstractmethod
    def load_config_for_class(
            self, namespace: str, key: str, config_class: Type[Config]
    ) -> Config:
        raise NotImplementedError

    @abstractmethod
    def get_value(self, namespace: str, key: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_json(self, namespace: str, key: str) -> str:
        raise NotImplementedError


class EnvConfigLoader[Config](ConfigLoader):

    def get_json(self, namespace: str, key: str) -> str:
        json_data = os.getenv(key)
        if namespace:
            json_data = os.getenv(namespace)
        if json_data is None:
            raise AppError(f"No config found for {namespace}{"." if namespace else ""}{key}")
        try:
            return json.loads(json_data)
        except json.JSONDecodeError:
            raise AppError(f"加载 {key} 配置失败，配置文件格式有误")

    def get_value(self, namespace: Optional[str], key: str) -> str:
        value = os.getenv(key)
        if namespace:
            value = os.getenv(namespace)
        if value is None:
            raise AppError(f"No value found for {namespace}{"." if namespace else ""}{key}")
        return value

    def load_config_for_class(
            self, namespace: str, key: str, config_class: Type[Config]
    ) -> Config:
        config_data = os.getenv(namespace)
        if config_data is None:
            raise AppError(f"No config found for {namespace}")
        try:
            json_config = json.loads(config_data)
            return config_class(**json_config.get(key))
        except json.JSONDecodeError:
            raise AppError(f"加载 {key} 配置失败，配置文件格式有误")
        except TypeError:
            raise AppError(f"没有找到 {key} 模型的配置，检查模型名称是否有误")
