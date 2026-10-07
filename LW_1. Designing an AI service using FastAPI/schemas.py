"""Pydantic-схемы: структура запросов, ответов и ограничения значений."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_type: str
    status: Literal["ready", "unavailable"]


class ErrorResponse(BaseModel):
    """Ошибки 400, 404, 500 и 503. Для 422 FastAPI использует свою схему."""

    detail: str
