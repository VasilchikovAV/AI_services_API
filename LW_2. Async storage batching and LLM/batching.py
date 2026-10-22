"""Очередь отдельных запросов и фоновое формирование пакетов для модели."""

import asyncio
import logging
from dataclasses import dataclass

import config
import model
from schemas import FarmRequest


logger = logging.getLogger(__name__)


class InferenceUnavailableError(Exception):
    """Worker остановлен и не может вернуть результат ожидающему запросу."""


@dataclass
class InferenceTask:
    data: FarmRequest
    request_id: str
    future: asyncio.Future[float]


def fail_waiters(tasks: list[InferenceTask], error: Exception) -> None:
    for task in tasks:
        if not task.future.done():
            task.future.set_exception(error)


async def batch_worker(queue: asyncio.Queue[InferenceTask]) -> None:
    """Собирает до 8 задач или ждёт 50 мс и передаёт модели один список."""
    logger.info("Batch worker started")
    try:
        while True:
            batch: list[InferenceTask] = []
            try:
                first = await queue.get()
                if first.future.done():
                    queue.task_done()
                    continue
                batch.append(first)
                deadline = asyncio.get_running_loop().time() + config.MAX_BATCH_WAIT_MS / 1000

                while len(batch) < config.MAX_BATCH_SIZE:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        break
                    try:
                        async with asyncio.timeout(remaining):
                            task = await queue.get()
                    except TimeoutError:
                        break
                    if task.future.done():
                        queue.task_done()
                        continue
                    batch.append(task)

                # Запрос мог истечь, пока worker накапливал пакет.
                active = [task for task in batch if not task.future.done()]
                if not active:
                    continue
                request_ids = [task.request_id for task in active]
                logger.info(
                    "Dynamic batch formed | size=%s | request_ids=%s | queue_size=%s",
                    len(active), request_ids, queue.qsize(),
                )
                scores = await asyncio.wait_for(
                    model.model_predict_batch([task.data for task in active]),
                    timeout=config.INFERENCE_TIMEOUT_SECONDS,
                )
                pairs = list(zip(active, scores, strict=True))
                for task, score in pairs:
                    if not task.future.done():
                        task.future.set_result(score)
            except asyncio.CancelledError:
                fail_waiters(batch, InferenceUnavailableError("Inference worker is stopping"))
                raise
            except asyncio.QueueShutDown:
                fail_waiters(batch, InferenceUnavailableError("Inference queue is closed"))
                return
            except Exception as exc:
                logger.exception(
                    "Dynamic batch failed | request_ids=%s", [task.request_id for task in batch],
                )
                fail_waiters(batch, exc)
            finally:
                # Каждая извлечённая задача завершается ровно одним task_done().
                for _ in batch:
                    queue.task_done()
    finally:
        logger.info("Batch worker stopped")


async def stop_worker(queue: asyncio.Queue[InferenceTask], worker: asyncio.Task) -> None:
    """Останавливает worker и освобождает ожидающие запросы, включая остаток очереди."""
    queue.shutdown()
    worker.cancel()
    try:
        await worker
    except asyncio.CancelledError:
        pass
    finally:
        while True:
            try:
                task = queue.get_nowait()
            except (asyncio.QueueEmpty, asyncio.QueueShutDown):
                break
            if not task.future.done():
                task.future.set_exception(InferenceUnavailableError("Inference worker is stopping"))
            queue.task_done()
