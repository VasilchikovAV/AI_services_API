"""Модель расчёта риска."""

import asyncio

import config
from schemas import FarmRequest


MODEL_NAME = "agro-risk-model"
MODEL_VERSION = "1.0"
MODEL_TYPE = "risk-scoring"
MODEL_READY = True
model_lock = asyncio.Lock()


def calculate_risk(data: FarmRequest) -> float:
    """Реализует простую логику оценки риска по финансовым показателям
    и отраслевым факторам."""
    score = 0.1

    if data.payment_delay_days > 30:
        score += 0.3
    if data.previous_defaults > 0:
        score += 0.3
    if data.debt > 5_000_000:
        score += 0.2
    if data.precipitation_mm < 100:
        score += 0.1

    return round(min(score, 1.0), 2)


async def model_predict_batch(items: list[FarmRequest]) -> list[float]:
    """Обрабатывает весь список за один вызов с общей фиксированной задержкой."""
    async with model_lock:
        await asyncio.sleep(
            config.MODEL_FIXED_OVERHEAD_SECONDS + config.MODEL_PER_ITEM_SECONDS * len(items)
        )
        return [calculate_risk(item) for item in items]
