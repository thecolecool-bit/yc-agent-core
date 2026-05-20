import asyncio
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from yc.common.exceptions import AppError
from yc.agent.base import Agent
from yc.schemas.agent import AgentEvent, AgentEventType
from yc.schemas.memory import MemoryChunk


class MemoryStore(ABC):

    def register(self, agent: Agent):
        raise NotImplementedError

    @abstractmethod
    def save(self, agent_name: str, memory: MemoryChunk):
        raise NotImplementedError

    @abstractmethod
    def remove(self, memory_id: str):
        raise NotImplementedError

    @abstractmethod
    def get_all(self, agent: Agent) -> List[MemoryChunk]:
        raise NotImplementedError


class InMemoryStore(MemoryStore):
    def __init__(self) -> None:
        super().__init__()
        self._memory: List[MemoryChunk] = []

    def register(self, agent: Agent):
        pass

    def save(self, agent_name: str, memory: MemoryChunk):
        self._memory.append(memory)

    def remove(self, memory_id: str):
        for memory in self._memory:
            if memory.id == memory_id:
                self._memory.remove(memory)
                break

    def get_all(self, agent: Agent) -> List[MemoryChunk]:
        return self._memory


class LocalFileFullMemoryStore(MemoryStore):

    def __init__(self, store_dir: Path):
        super().__init__()
        self.store_dir = store_dir

    def register(self, agent: Agent):
        async def _save(event: AgentEvent):
            await asyncio.to_thread(self.save,
                                    agent.get_name(),
                                    MemoryChunk(original_message=event.data, content=event.content, type='original'))

        agent.subscribe_event(AgentEventType.UPDATE_CONTEXT, _save)

    def save(self, agent_name: str, memory: MemoryChunk):
        memory_file = Path(self.store_dir) / f"{agent_name}" / (time.strftime("%Y%m%d", time.localtime()) + ".json")
        if memory_file.exists():
            memory_file.open("a", encoding="utf-8").write(f'{memory.model_dump_json()},\n')
        else:
            memory_file.parent.mkdir(parents=True, exist_ok=True)
            memory_file.touch()
            memory_file.open("a", encoding="utf-8").write(f'[{memory.model_dump_json()},\n')

    def remove(self, memory_id: str):
        pass

    def get_all(self, agent: Agent) -> List[MemoryChunk]:
        memory_list = []
        memory_dir = Path(self.store_dir) / f'{agent.get_name()}'
        for memory_file in memory_dir.glob("*.json"):
            memory_data = memory_file.open("r", encoding="utf-8").read().strip()
            if memory_data.endswith("]"):
                memory_data = memory_data
            elif memory_data.endswith(","):
                memory_data = f'{memory_data[:-1]}]'
            else:
                raise AppError("记忆文件格式错误，无法读取记忆")
            memory_json_list = json.loads(memory_data)
            for memory_json in memory_json_list:
                memory_list.append(MemoryChunk.model_validate(memory_json))
        return memory_list
