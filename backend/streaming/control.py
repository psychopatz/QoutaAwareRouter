import asyncio
import inspect
from typing import Any, Callable, Optional


CancelCallback = Callable[[], Any]


class ProviderStreamControl:
    def __init__(self):
        self.cancelled = False
        self.native_cancel_supported = False
        self._cancel_callback: Optional[CancelCallback] = None
        self._cancel_lock = asyncio.Lock()

    def register_cancel_callback(self, callback: CancelCallback, native_supported: bool = False):
        self._cancel_callback = callback
        self.native_cancel_supported = native_supported

    async def cancel(self):
        async with self._cancel_lock:
            if self.cancelled:
                return

            self.cancelled = True
            if self._cancel_callback is None:
                return

            result = self._cancel_callback()
            if inspect.isawaitable(result):
                await result