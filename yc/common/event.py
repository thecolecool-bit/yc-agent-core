import asyncio
import inspect
from typing import Dict, Callable, List

from typing_extensions import Awaitable

from yc.common.exceptions import AppError
from yc.schemas.event import EventType, Event


class EventBus[_EventType:EventType, _Event:Event]:

    def __init__(self):
        self._subscribers: Dict[_EventType, List[Callable[[_Event], Awaitable[None]]]] = {}

    def subscribe_event(self, event_type: _EventType, callback: Callable[[_Event], Awaitable[None]]):
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)

    async def async_broadcast(self, event: _Event):
        callbacks = self._subscribers.get(event.type, [])
        task = []
        for callback in callbacks:
            if inspect.iscoroutinefunction(callback):
                task.append(callback(event))
            else:
                task.append(asyncio.to_thread(callback, event))
        await asyncio.gather(*task, return_exceptions=True)

    def broadcast(self, event: _Event):
        callbacks = self._subscribers.get(event.type, [])
        for callback in callbacks:
            try:
                if inspect.iscoroutinefunction(callback):
                    asyncio.run(callback(event))
                else:
                    callback(event)
            except Exception as e:
                raise AppError(str(e))
