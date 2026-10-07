"""Бизнес-правила, постобработка и координация расчёта с сохранением."""

import logging
import uuid

from fastapi import HTTPException, status

import model
import storage
from schemas import FarmRequest, PredictionResponse


logger = logging.getLogger(__name__)
ALLOWED_REGIONS = {"Krasnodar", "Rostov", "Stavropol"}
ALLOWED_RISK_LEVELS = {"low", "medium", "high"}


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


def create_prediction(request: FarmRequest) -> PredictionResponse:
    if not model.MODEL_READY:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is temporarily unavailable")

    # Тип region уже проверен. Теперь проверяем предметное правило: регион из справочника.
    if request.region not in ALLOWED_REGIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown region: {request.region}. Allowed regions: {sorted(ALLOWED_REGIONS)}",
        )

    logger.info("Prediction request received | farm_id=%s", request.farm_id)
    score = model.calculate_risk(request)
    level = get_risk_level(score)
    result = PredictionResponse(
        request_id=str(uuid.uuid4()),
        farm_id=request.farm_id,
        risk_score=score,
        risk_level=level,
        recommendation=get_recommendation(level),
        model_version=model.MODEL_VERSION,
    )
    storage.save_prediction(result)
    logger.info(
        "Prediction completed | request_id=%s | farm_id=%s | risk_score=%s | risk_level=%s | model_version=%s",
        result.request_id, result.farm_id, result.risk_score, result.risk_level, result.model_version,
    )
    return result


def find_prediction(request_id: str) -> PredictionResponse:
    result = storage.get_prediction(request_id)
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found")
    return result


def list_predictions(limit: int, risk_level: str | None) -> list[PredictionResponse]:
    if risk_level is not None and risk_level not in ALLOWED_RISK_LEVELS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="risk_level must be 'low', 'medium' or 'high'")

    results = storage.list_predictions()
    if risk_level is not None:
        results = [item for item in results if item.risk_level == risk_level]
    return results[:limit]
