"""Точка входа FastAPI: HTTP endpoints, документация и журнал запросов."""

import logging
import time

from fastapi import FastAPI, Query, Request, status
from fastapi.responses import JSONResponse

import model
import services
from schemas import ErrorResponse, FarmRequest, HealthResponse, ModelInfoResponse, PredictionResponse


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Agro Scoring API",
    description=(
        "REST API для оценки риска сельскохозяйственных предприятий. "
        "Прогнозы хранятся в памяти до перезапуска приложения."
    ),
    version="1.0.0",
)

SERVER_ERROR = {500: {"model": ErrorResponse, "description": "Внутренняя ошибка сервиса"}}


@app.middleware("http")
async def log_request(request: Request, call_next):
    """Измеряет время обработки и добавляет его в заголовок ответа."""
    started = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - started
    response.headers["X-Process-Time"] = f"{elapsed:.6f}"
    logger.info("HTTP %s %s | status=%s | time=%.6fs", request.method, request.url.path, response.status_code, elapsed)
    return response


@app.exception_handler(Exception)
async def handle_internal_error(request: Request, exc: Exception):
    """Детали внутренней ошибки остаются в журнале, клиент получает HTTP 500."""
    logger.error("Internal error | %s %s", request.method, request.url.path, exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


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
        400: {"model": ErrorResponse, "description": "Регион отсутствует в бизнес-справочнике"},
        503: {"model": ErrorResponse, "description": "Модель временно недоступна"},
    },
)
def predict(request: FarmRequest):
    # FastAPI проверяет FarmRequest с помощью Pydantic до вызова этой функции.
    return services.create_prediction(request)


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
        400: {"model": ErrorResponse, "description": "Неизвестная категория риска"},
    },
)
def get_predictions(
    limit: int = Query(default=10, ge=1, le=100, description="Количество результатов: от 1 до 100"),
    risk_level: str | None = Query(default=None, description="Категория риска: low, medium или high"),
):
    return services.list_predictions(limit, risk_level)


@app.get(
    "/predictions/{request_id}",
    response_model=PredictionResponse,
    tags=["Сохранённые прогнозы"],
    summary="Получить прогноз по ID",
    description="Подставьте request_id, полученный в ответе POST /predict.",
    responses={
        **SERVER_ERROR,
        404: {"model": ErrorResponse, "description": "Прогноз с таким ID не найден"},
    },
)
def get_prediction(request_id: str):
    return services.find_prediction(request_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
