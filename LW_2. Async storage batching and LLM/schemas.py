"""Pydantic-схемы: структура запросов, ответов и ограничения значений."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from config import (
    LLM_DEFAULT_MAX_TOKENS,
    LLM_DEFAULT_TEMPERATURE,
    LLM_MAX_TOKENS_LIMIT,
    LLM_MODEL,
    MAX_BATCH_SIZE,
)


class FarmRequest(BaseModel):
    """Данные сельскохозяйственного предприятия, которые клиент передает в POST /predict"""

    farm_id: str = Field(min_length=1, description="Идентификатор хозяйства")
    region: str = Field(min_length=1, description="Регион: Krasnodar, Rostov или Stavropol")
    crop_type: str = Field(min_length=1, description="Основная сельскохозяйственная культура")
    area_ha: float = Field(gt=0, allow_inf_nan=False, description="Площадь посевов, га")
    temperature_avg: float = Field(ge=-60, le=60, allow_inf_nan=False, description="Средняя температура, °C")
    precipitation_mm: float = Field(ge=0, allow_inf_nan=False, description="Количество осадков, мм")
    payment_delay_days: int = Field(ge=0, description="Количество дней просрочки платежа")
    previous_defaults: int = Field(ge=0, description="Количество предыдущих дефолтов")
    debt: float = Field(ge=0, allow_inf_nan=False, description="Текущая задолженность")

    model_config = ConfigDict(json_schema_extra={
        "examples": [{
            
            "farm_id": "FARM-003",
            "region": "Krasnodar",
            "crop_type": "wheat",
            "area_ha": 2500,
            "temperature_avg": 24.3,
            "precipitation_mm": 320,
            "payment_delay_days": 45,
            "previous_defaults": 1,
            "debt": 6500000,
        }]
    })


class PredictionResponse(BaseModel):
    """Контракт успешного ответа для одного прогноза."""

    request_id: str
    farm_id: str
    risk_score: float = Field(ge=0, le=1, description="Оценка риска от 0 до 1")
    risk_level: Literal["low", "medium", "high"]
    recommendation: str
    model_version: str


class BatchRequest(BaseModel):
    """От одного до восьми хозяйств в одном HTTP-запросе."""

    items: list[FarmRequest] = Field(
        min_length=1, max_length=MAX_BATCH_SIZE,
        description="Хозяйства для пакетного расчёта, от 1 до 8 объектов",
    )


class BatchItemResponse(BaseModel):
    farm_id: str
    risk_score: float = Field(ge=0, le=1)
    risk_level: Literal["low", "medium", "high"]
    recommendation: str
    model_version: str


class BatchPredictionResponse(BaseModel):
    request_id: str
    batch_size: int = Field(ge=1, le=MAX_BATCH_SIZE)
    processing_time_ms: float = Field(ge=0)
    items: list[BatchItemResponse] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class PaginationMeta(BaseModel):
    """Параметры страницы, общее число результатов и ссылки перехода."""

    weekday: str
    current_page: int = Field(ge=1)
    per_page: int = Field(ge=1, le=100)
    total_pages: int = Field(ge=0)
    total_items: int = Field(ge=0)
    next_page: str | None
    prev_page: str | None


class PaginatedPredictionsResponse(BaseModel):
    """Одна страница прогнозов и сведения о полной выборке."""

    data: list[PredictionResponse]
    meta: PaginationMeta


class LLMChatRequest(BaseModel):
    """Промпт и параметры генерации текста."""

    prompt: str = Field(min_length=1, max_length=5000, description="Текст запроса к LLM")
    model: str = Field(default=LLM_MODEL, min_length=1, description="Модель провайдера")
    temperature: float = Field(
        default=LLM_DEFAULT_TEMPERATURE, ge=0, le=2, allow_inf_nan=False,
        description="Температура генерации от 0 до 2",
    )
    max_tokens: int = Field(
        default=LLM_DEFAULT_MAX_TOKENS, ge=1, le=LLM_MAX_TOKENS_LIMIT,
        description="Максимальное число токенов ответа, от 1 до 2000",
    )

    model_config = ConfigDict(json_schema_extra={"examples": [{
        "prompt": "Объясни в двух предложениях, зачем API нужен request_id.",
        "model": "auto",
        "temperature": 0.3,
        "max_tokens": 300,
    }]})


class LLMChatResponse(BaseModel):
    request_id: str
    requested_model: str
    returned_model: str | None = None
    answer: str = Field(min_length=1)
    reasoning: str | None = None
    finish_reason: str | None = None
    usage: dict | None = None
    latency_ms: float = Field(ge=0, allow_inf_nan=False)


class PredictWithLLMResponse(BaseModel):
    """Сохранённый прогноз и текстовое объяснение его результата."""

    prediction: PredictionResponse
    llm: LLMChatResponse


class HealthResponse(BaseModel):
    status: Literal["ok"]


class QueueStatusResponse(BaseModel):
    """Текущий размер очереди и настройки формирования пакетов."""

    queue_size: int = Field(ge=0)
    queue_max_size: int = Field(ge=1)
    max_batch_size: int = Field(ge=1)
    max_batch_wait_ms: int = Field(ge=0)
    inference_timeout_seconds: float = Field(gt=0)


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_type: str
    status: Literal["ready", "unavailable"]


class ErrorResponse(BaseModel):
    """Сообщение об ошибке сервиса и идентификатор HTTP-запроса."""

    detail: str
    request_id: str | None = None


class ValidationErrorResponse(BaseModel):
    """Список ошибок входных данных и идентификатор HTTP-запроса."""

    detail: list[dict]
    request_id: str | None = None
