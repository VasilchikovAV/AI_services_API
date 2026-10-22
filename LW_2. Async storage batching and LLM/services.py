"""Бизнес-правила, постобработка и координация расчёта с сохранением."""

import asyncio
import logging
import time

import httpx
from fastapi import HTTPException, status

import config
import llm
import model
import storage
from batching import InferenceTask, InferenceUnavailableError
from schemas import (
    BatchItemResponse, BatchPredictionResponse, BatchRequest, FarmRequest,
    LLMChatRequest, PredictionResponse, PredictWithLLMResponse,
)


logger = logging.getLogger(__name__)
ALLOWED_REGIONS = {"Krasnodar", "Rostov", "Stavropol"}
ALLOWED_RISK_LEVELS = {"low", "medium", "high"}
WEEKDAY_NUMBERS = {
    "monday": "1",
    "tuesday": "2",
    "wednesday": "3",
    "thursday": "4",
    "friday": "5",
    "saturday": "6",
    "sunday": "0",
}


def get_risk_level(score: float) -> str:
    if score < 0.3:
        return "low"
    if score < 0.7:
        return "medium"
    return "high"


def get_recommendation(level: str) -> str:
    if level == "low":
        return "Стандартное рассмотрение"
    if level == "medium":
        return "Требуется дополнительная проверка"
    return "Высокий риск. Требуется ручное рассмотрение"


def validate_prediction_inputs(items: list[FarmRequest]) -> None:
    if not model.MODEL_READY:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is temporarily unavailable")

    for item in items:
        if item.region not in ALLOWED_REGIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown region: {item.region}. Allowed regions: {sorted(ALLOWED_REGIONS)}",
            )


async def run_inference(items: list[FarmRequest]) -> list[float]:
    """Ожидает один вызов модели, включая ожидание её блокировки, не дольше таймаута."""
    try:
        return await asyncio.wait_for(
            model.model_predict_batch(items), timeout=config.INFERENCE_TIMEOUT_SECONDS,
        )
    except TimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="Inference timeout",
        ) from exc


def build_prediction(request: FarmRequest, score: float, request_id: str) -> PredictionResponse:
    level = get_risk_level(score)
    return PredictionResponse(
        request_id=request_id,
        farm_id=request.farm_id,
        risk_score=score,
        risk_level=level,
        recommendation=get_recommendation(level),
        model_version=model.MODEL_VERSION,
    )


async def create_prediction(
    request: FarmRequest, request_id: str, *, source: str = "single",
) -> PredictionResponse:
    validate_prediction_inputs([request])
    logger.info("Prediction request received | request_id=%s | farm_id=%s", request_id, request.farm_id)
    scores = await run_inference([request])
    result = build_prediction(request, scores[0], request_id)
    await storage.save_prediction(result, source=source)
    logger.info(
        "Prediction completed | request_id=%s | farm_id=%s | risk_score=%s | risk_level=%s | model_version=%s",
        result.request_id, result.farm_id, result.risk_score, result.risk_level, result.model_version,
    )
    return result


async def create_prediction_with_llm(
    request: FarmRequest, request_id: str, client: httpx.AsyncClient,
) -> PredictWithLLMResponse:
    # This returns only after COMMIT; an LLM failure must not erase the score.
    prediction = await create_prediction(request, request_id, source="single+llm")
    prompt = llm.build_scoring_explanation_prompt(prediction)
    explanation = await llm.call_llm(
        client, LLMChatRequest(prompt=prompt), request_id,
        system_prompt=llm.SCORING_SYSTEM_PROMPT,
    )
    explanation = llm.validate_scoring_explanation(explanation, prediction)
    return PredictWithLLMResponse(prediction=prediction, llm=explanation)


async def create_batch_prediction(batch: BatchRequest, request_id: str) -> BatchPredictionResponse:
    validate_prediction_inputs(batch.items)
    logger.info("Batch request received | request_id=%s | batch_size=%s", request_id, len(batch.items))
    started = time.perf_counter()
    scores = await run_inference(batch.items)
    results = [
        build_prediction(item, score, request_id)
        for item, score in zip(batch.items, scores, strict=True)
    ]
    await storage.save_predictions_batch(results)
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "Batch completed | request_id=%s | batch_size=%s | processing_time_ms=%.3f",
        request_id, len(results), elapsed_ms,
    )
    return BatchPredictionResponse(
        request_id=request_id, batch_size=len(results), processing_time_ms=elapsed_ms,
        items=[BatchItemResponse.model_validate(result.model_dump(exclude={"request_id"})) for result in results],
    )


async def create_queued_prediction(
    request: FarmRequest, request_id: str, queue: asyncio.Queue[InferenceTask],
) -> PredictionResponse:
    validate_prediction_inputs([request])
    future = asyncio.get_running_loop().create_future()
    task = InferenceTask(data=request, request_id=request_id, future=future)
    try:
        try:
            queue.put_nowait(task)
        except asyncio.QueueFull as exc:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Inference queue is full. Try again later.", headers={"Retry-After": "1"},
            ) from exc
        except asyncio.QueueShutDown as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Inference queue is unavailable",
            ) from exc
        logger.info("Request queued | request_id=%s | queue_size=%s", request_id, queue.qsize())
        try:
            score = await asyncio.wait_for(future, timeout=config.INFERENCE_TIMEOUT_SECONDS)
        except TimeoutError as exc:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="Inference timeout",
            ) from exc
        except InferenceUnavailableError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Inference worker is unavailable",
            ) from exc
    finally:
        # При отмене HTTP-запроса worker не должен пытаться вернуть ему результат.
        if not future.done():
            future.cancel()

    result = build_prediction(request, score, request_id)
    await storage.save_prediction(result, source="queued")
    logger.info(
        "Queued prediction completed | request_id=%s | farm_id=%s | risk_score=%s",
        request_id, result.farm_id, result.risk_score,
    )
    return result


async def find_request_predictions(request_id: str) -> list[PredictionResponse]:
    """Возвращает все прогнозы HTTP-запроса или 404, если таких записей нет."""
    results = await storage.get_predictions_by_request_id(request_id)
    if not results:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found")
    return results


async def find_prediction(request_id: str) -> PredictionResponse:
    results = await find_request_predictions(request_id)
    if len(results) > 1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Multiple predictions found for this request_id. "
                f"Use /requests/{request_id}/predictions"
            ),
        )
    return results[0]


async def list_predictions(limit: int, risk_level: str | None) -> list[PredictionResponse]:
    if risk_level is not None and risk_level not in ALLOWED_RISK_LEVELS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="risk_level must be 'low', 'medium' or 'high'")

    return await storage.list_predictions(limit, risk_level)


async def list_predictions_by_weekday(
    weekday: str, page: int, per_page: int,
) -> tuple[list[PredictionResponse], int, int]:
    """Проверяет день недели и границы страницы, затем возвращает выборку и счётчики."""
    if weekday not in WEEKDAY_NUMBERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"weekday must be one of: {', '.join(WEEKDAY_NUMBERS)}",
        )
    results, total_items = await storage.get_predictions_by_weekday_page(
        WEEKDAY_NUMBERS[weekday], page, per_page,
    )
    total_pages = (total_items + per_page - 1) // per_page
    if total_pages == 0 and page > 1:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No data for this weekday")
    if total_pages > 0 and page > total_pages:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Page {page} does not exist. Total pages: {total_pages}",
        )
    return results, total_items, total_pages
