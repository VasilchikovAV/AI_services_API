"""Точка входа FastAPI: HTTP endpoints, документация и журнал запросов."""

import asyncio
import logging
import math
import time
import uuid
from contextlib import asynccontextmanager

import aiosqlite
import httpx
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

import batching
import config
import llm
import model
import services
import storage
from schemas import (
    BatchPredictionResponse,
    BatchRequest,
    ErrorResponse,
    FarmRequest,
    HealthResponse,
    LLMChatRequest,
    LLMChatResponse,
    ModelInfoResponse,
    PaginatedPredictionsResponse,
    PaginationMeta,
    PredictionResponse,
    PredictWithLLMResponse,
    QueueStatusResponse,
    ValidationErrorResponse,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Подготовка БД завершается до начала обработки HTTP-запросов.
    await storage.init_db()
    logger.info("SQLite initialized | path=%s", storage.DATABASE_PATH)
    async with httpx.AsyncClient(timeout=httpx.Timeout(config.LLM_TIMEOUT_SECONDS)) as llm_client:
        app.state.llm_client = llm_client
        app.state.inference_queue = asyncio.Queue(maxsize=config.QUEUE_MAX_SIZE)
        logger.info("Inference queue initialized | max_size=%s", app.state.inference_queue.maxsize)
        app.state.batch_worker = asyncio.create_task(
            batching.batch_worker(app.state.inference_queue), name="agro-batch-worker",
        )
        try:
            yield
        finally:
            await batching.stop_worker(app.state.inference_queue, app.state.batch_worker)


app = FastAPI(
    title="Agro Scoring API",
    description=(
        "REST API для оценки риска сельскохозяйственных предприятий. "
        "Прогнозы сохраняются в SQLite и доступны после перезапуска приложения."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

SERVER_ERROR = {500: {"model": ErrorResponse, "description": "Внутренняя ошибка сервиса"}}
VALIDATION_ERROR = {422: {"model": ValidationErrorResponse, "description": "Некорректные входные данные"}}
STORAGE_ERROR = {503: {"model": ErrorResponse, "description": "Хранилище временно недоступно"}}
INFERENCE_ERRORS = {
    503: {"model": ErrorResponse, "description": "Модель или хранилище временно недоступны"},
    504: {"model": ErrorResponse, "description": "Превышено время ожидания модели"},
}


@app.middleware("http")
async def log_request(request: Request, call_next):
    """Назначает сквозной ID и измеряет время обработки HTTP-запроса."""
    started = time.perf_counter()
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    logger.info(
        "Request started | request_id=%s | method=%s | path=%s",
        request_id, request.method, request.url.path,
    )

    try:
        response = await call_next(request)
    except Exception as exc:
        response = await handle_internal_error(request, exc)
    elapsed = time.perf_counter() - started
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{elapsed:.6f}"
    logger.info(
        "HTTP %s %s | request_id=%s | status=%s | time=%.6fs",
        request.method, request.url.path, request_id, response.status_code, elapsed,
    )
    return response


@app.exception_handler(Exception)
async def handle_internal_error(request: Request, exc: Exception):
    """Детали внутренней ошибки остаются в журнале, клиент получает HTTP 500."""
    request_id = getattr(request.state, "request_id", None)
    logger.error(
        "Internal error | request_id=%s | %s %s",
        request_id, request.method, request.url.path, exc_info=exc,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": request_id},
    )


@app.exception_handler(aiosqlite.Error)
async def handle_storage_error(request: Request, exc: aiosqlite.Error):
    """Возвращает 503 при сбое SQLite; детали остаются в журнале."""
    request_id = getattr(request.state, "request_id", None)
    logger.error("Database error | request_id=%s", request_id, exc_info=exc)
    return JSONResponse(
        status_code=503,
        content={"detail": "Prediction storage is unavailable", "request_id": request_id},
    )


@app.exception_handler(StarletteHTTPException)
async def handle_http_error(request: Request, exc: StarletteHTTPException):
    """Добавляет ID в HTTP-ошибки сервиса и маршрутизации."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "request_id": getattr(request.state, "request_id", None)},
        headers=exc.headers,
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError):
    """Сохраняет список ошибок FastAPI и связывает его с текущим запросом."""
    return JSONResponse(
        status_code=422,
        content={
            # A rejected NaN/Infinity must also be serializable in the error body.
            "detail": jsonable_encoder(
                exc.errors(), custom_encoder={float: lambda value: value if math.isfinite(value) else str(value)},
            ),
            "request_id": getattr(request.state, "request_id", None),
        },
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Состояние сервиса"],
    summary="Проверить доступность API",
    description="Показывает, что приложение запущено и принимает HTTP-запросы.",
    responses=SERVER_ERROR,
)
def health():
    return HealthResponse(status="ok")


@app.get(
    "/queue-status",
    response_model=QueueStatusResponse,
    tags=["Состояние сервиса"],
    summary="Получить состояние очереди",
    description="Возвращает текущий размер очереди, её ёмкость и настройки пакетной обработки.",
    responses=SERVER_ERROR,
)
async def queue_status(request: Request):
    queue = request.app.state.inference_queue
    return QueueStatusResponse(
        queue_size=queue.qsize(),
        queue_max_size=queue.maxsize,
        max_batch_size=config.MAX_BATCH_SIZE,
        max_batch_wait_ms=config.MAX_BATCH_WAIT_MS,
        inference_timeout_seconds=config.INFERENCE_TIMEOUT_SECONDS,
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    tags=["Состояние сервиса"],
    summary="Получить информацию о модели",
    description="Возвращает название, версию, тип и готовность модели.",
    responses=SERVER_ERROR,
)
def model_info():
    return ModelInfoResponse(
        model_name=model.MODEL_NAME,
        model_version=model.MODEL_VERSION,
        model_type=model.MODEL_TYPE,
        status="ready" if model.MODEL_READY else "unavailable",
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Оценка риска"],
    summary="Оценить риск хозяйства",
    description=(
        "Проверяет входные данные и разрешённый регион, рассчитывает риск, "
        "формирует рекомендацию и сохраняет результат. "
        "Возвращает 200 OK: результат расчёта готов в рамках одного запроса."
    ),
    responses={
        **SERVER_ERROR,
        **VALIDATION_ERROR,
        **INFERENCE_ERRORS,
        400: {"model": ErrorResponse, "description": "Регион отсутствует в бизнес-справочнике"},
    },
)
async def predict(data: FarmRequest, request: Request):
    # FastAPI проверяет FarmRequest с помощью Pydantic до вызова этой функции.
    return await services.create_prediction(data, request.state.request_id)


@app.post(
    "/batch-predict",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Оценка риска"],
    summary="Оценить риск пакета хозяйств",
    description=(
        "Принимает от 1 до 8 хозяйств, выполняет один пакетный вызов модели "
        "и сохраняет все результаты одной транзакцией SQLite с общим request_id. "
        "Возвращает 200 после фиксации записей."
    ),
    responses={
        **SERVER_ERROR,
        **VALIDATION_ERROR,
        **INFERENCE_ERRORS,
        400: {"model": ErrorResponse, "description": "Регион одного из хозяйств отсутствует в справочнике"},
    },
)
async def batch_predict(batch: BatchRequest, request: Request):
    return await services.create_batch_prediction(batch, request.state.request_id)


@app.post(
    "/predict-queued",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Оценка риска"],
    summary="Оценить риск через очередь",
    description=(
        "Принимает одно хозяйство и ждёт результат в этом же HTTP-запросе. "
        "Worker объединяет отдельные запросы в пакет до 8 объектов или ждёт до 50 мс. "
        "Возвращает 200 после расчёта и сохранения прогноза в SQLite."
    ),
    responses={
        **SERVER_ERROR,
        **VALIDATION_ERROR,
        **INFERENCE_ERRORS,
        400: {"model": ErrorResponse, "description": "Регион отсутствует в справочнике"},
        429: {
            "model": ErrorResponse, "description": "Очередь заполнена",
            "headers": {"Retry-After": {"schema": {"type": "string"}, "description": "Повторить через 1 секунду"}},
        },
    },
)
async def predict_queued(data: FarmRequest, request: Request):
    if request.app.state.batch_worker.done():
        raise HTTPException(status_code=503, detail="Inference worker is unavailable")
    return await services.create_queued_prediction(
        data, request.state.request_id, request.app.state.inference_queue,
    )


@app.post(
    "/llm/chat",
    response_model=LLMChatResponse,
    tags=["LLM"],
    summary="Получить текстовый ответ LLM",
    description=(
        "Отправляет prompt во внешний LLM API и возвращает ответ, сведения о модели "
        "и длительность внешнего вызова. Прогноз риска не создаётся."
    ),
    responses={
        **SERVER_ERROR,
        **VALIDATION_ERROR,
        502: {"model": ErrorResponse, "description": "Ошибка или некорректный ответ провайдера LLM"},
        503: {"model": ErrorResponse, "description": "Не удалось связаться с LLM"},
        504: {"model": ErrorResponse, "description": "Превышено время ожидания LLM"},
    },
)
async def llm_chat(data: LLMChatRequest, request: Request):
    return await llm.call_llm(request.app.state.llm_client, data, request.state.request_id)


@app.post(
    "/predict-with-llm",
    response_model=PredictWithLLMResponse,
    tags=["LLM"],
    summary="Оценить риск и получить объяснение LLM",
    description=(
        "Рассчитывает риск и сохраняет прогноз в SQLite, затем передаёт LLM только "
        "балл, категорию, рекомендацию и версию модели для текстового объяснения. "
        "При ошибке LLM прогноз остаётся в БД и доступен по request_id из ошибки."
    ),
    responses={
        **SERVER_ERROR,
        **VALIDATION_ERROR,
        400: {"model": ErrorResponse, "description": "Регион отсутствует в справочнике"},
        502: {"model": ErrorResponse, "description": "Ошибка провайдера, неверный формат или несогласованное объяснение; прогноз сохранён"},
        503: {"model": ErrorResponse, "description": "Недоступность модели, хранилища или соединения с LLM"},
        504: {"model": ErrorResponse, "description": "Превышено время ожидания скоринга или LLM"},
    },
)
async def predict_with_llm(data: FarmRequest, request: Request):
    return await services.create_prediction_with_llm(
        data, request.state.request_id, request.app.state.llm_client,
    )


@app.get(
    "/predictions",
    response_model=list[PredictionResponse],
    tags=["Сохранённые прогнозы"],
    summary="Получить последние прогнозы",
    description=(
        "Возвращает прогнозы от новых к старым. "
        "Фильтрация по risk_level выполняется до ограничения количества результатов."
    ),
    responses={
        **SERVER_ERROR,
        **VALIDATION_ERROR,
        **STORAGE_ERROR,
        400: {"model": ErrorResponse, "description": "Неизвестная категория риска"},
    },
)
async def get_predictions(
    limit: int = Query(default=10, ge=1, le=100, description="Количество результатов: от 1 до 100"),
    risk_level: str | None = Query(default=None, description="Категория риска: low, medium или high"),
):
    return await services.list_predictions(limit, risk_level)


@app.get(
    "/requests/{request_id}/predictions",
    response_model=list[PredictionResponse],
    tags=["Сохранённые прогнозы"],
    summary="Получить все прогнозы одного запроса",
    description=(
        "Возвращает список прогнозов с указанным request_id. "
        "При отсутствии записей возвращает 404."
    ),
    responses={
        **SERVER_ERROR,
        **VALIDATION_ERROR,
        **STORAGE_ERROR,
        404: {"model": ErrorResponse, "description": "Прогнозы с таким ID не найдены"},
    },
)
async def get_request_predictions(request_id: str):
    return await services.find_request_predictions(request_id)


@app.get(
    "/predictions/by-weekday",
    response_model=PaginatedPredictionsResponse,
    tags=["Сохранённые прогнозы"],
    summary="Получить страницу прогнозов за день недели",
    description=(
        "Выбирает прогнозы по дню недели сохранения в UTC, от новых к старым. "
        "Возвращает data с результатами и meta со счётчиками и ссылками на страницы."
    ),
    responses={
        **SERVER_ERROR,
        **VALIDATION_ERROR,
        **STORAGE_ERROR,
        400: {"model": ErrorResponse, "description": "Неизвестный день недели"},
        404: {"model": ErrorResponse, "description": "Страница не существует"},
    },
)
async def get_predictions_by_weekday(
    request: Request,
    weekday: str = Query(
        ...,
        description="День недели: monday, tuesday, wednesday, thursday, friday, saturday, sunday",
    ),
    page: int = Query(default=1, ge=1, description="Номер страницы, начиная с 1"),
    per_page: int = Query(default=5, ge=1, le=100, description="Количество записей на странице: от 1 до 100"),
):
    weekday = weekday.lower()
    results, total_items, total_pages = await services.list_predictions_by_weekday(
        weekday, page, per_page,
    )
    next_page = None
    prev_page = None
    if page < total_pages:
        next_page = str(request.url.include_query_params(
            weekday=weekday, page=page + 1, per_page=per_page,
        ))
    if page > 1:
        prev_page = str(request.url.include_query_params(
            weekday=weekday, page=page - 1, per_page=per_page,
        ))
    return PaginatedPredictionsResponse(
        data=results,
        meta=PaginationMeta(
            weekday=weekday, current_page=page, per_page=per_page,
            total_pages=total_pages, total_items=total_items,
            next_page=next_page, prev_page=prev_page,
        ),
    )


@app.get(
    "/predictions/{request_id}",
    response_model=PredictionResponse,
    tags=["Сохранённые прогнозы"],
    summary="Получить прогноз по ID",
    description=(
        "Подставьте request_id, полученный в ответе POST /predict. "
        "Если с ID связано несколько прогнозов, возвращает 409; "
        "все результаты доступны через /requests/{request_id}/predictions."
    ),
    responses={
        **SERVER_ERROR,
        **VALIDATION_ERROR,
        **STORAGE_ERROR,
        404: {"model": ErrorResponse, "description": "Прогноз с таким ID не найден"},
        409: {"model": ErrorResponse, "description": "С ID связано несколько прогнозов"},
    },
)
async def get_prediction(request_id: str):
    return await services.find_prediction(request_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
