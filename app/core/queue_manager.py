"""Bounded Priority Queue & Admission Controller with deadline enforcement."""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(order=True)
class QueuedItem:
    """Prioritized queue envelope ordered by (negative priority, arrival_time)."""

    sort_key: float
    item_id: str = field(compare=False)
    deadline: float = field(compare=False)
    payload: Any = field(compare=False)
    future: asyncio.Future = field(compare=False)


class BoundedPriorityQueueManager:
    """Admission controller buffering concurrent queries up to a strict capacity ceiling."""

    def __init__(self, max_depth: int = 100, max_concurrency: int = 5):
        self.max_depth = max_depth
        self.max_concurrency = max_concurrency
        self._queue: Optional[asyncio.PriorityQueue] = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        self.peak_queue_length: int = 0
        self.total_enqueued: int = 0
        self.total_dropped_deadline: int = 0
        self.total_rejected_capacity: int = 0

    @property
    def queue(self) -> asyncio.PriorityQueue:
        if self._queue is None:
            self._queue = asyncio.PriorityQueue(maxsize=self.max_depth)
        return self._queue

    @property
    def semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        return self._semaphore

    @property
    def current_depth(self) -> int:
        return self.queue.qsize()

    async def enqueue(
        self, item_id: str, payload: Any, deadline_seconds: float, priority: int = 1
    ) -> QueuedItem:
        """Enqueue request. Rejects if capacity ceiling is breached."""
        if self.queue.qsize() >= self.max_depth:
            self.total_rejected_capacity += 1
            raise RuntimeError(f"Queue capacity saturated ({self.max_depth} depth exceeded). Request shed.")

        now = time.time()
        deadline = now + deadline_seconds
        # Higher priority is popped first (negate priority so higher numbers have lower sort keys)
        sort_key = (-priority, now)

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        queued_item = QueuedItem(
            sort_key=sort_key,
            item_id=item_id,
            deadline=deadline,
            payload=payload,
            future=future,
        )

        await self.queue.put(queued_item)
        self.total_enqueued += 1
        q_len = self.queue.qsize()
        if q_len > self.peak_queue_length:
            self.peak_queue_length = q_len

        return queued_item

    async def dequeue_next(self) -> Optional[QueuedItem]:
        """Dequeue highest priority request. Enforces client deadline."""
        while not self.queue.empty():
            item = await self.queue.get()
            self.queue.task_done()

            now = time.time()
            if now > item.deadline:
                self.total_dropped_deadline += 1
                # Mark future as cancelled/timed out in queue
                if not item.future.done():
                    item.future.set_exception(
                        TimeoutError(f"Client deadline expired while waiting in admission queue ({item.item_id})")
                    )
                continue

            return item
        return None
