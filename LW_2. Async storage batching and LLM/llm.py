"""Асинхронный вызов LLM и проверка ответа внешнего API."""

import asyncio
import json
import logging
import re
import time
from typing import Literal

import httpx
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field

import config
from schemas import LLMChatRequest, LLMChatResponse, PredictionResponse


logger = logging.getLogger(__name__)
SYSTEM_PROMPT = (
    "Ты помощник сервиса агроскоринга. Отвечай кратко и понятно на русском языке. "
    "Не изменяй переданные risk_score и risk_level. "
    "Не придумывай причины, которых нет во входных данных."
)
SCORING_SYSTEM_PROMPT = (
    "Ты формулируешь объяснение готового результата агроскоринга для оператора банка. "
    "Входное сообщение содержит единственный актуальный результат. Не рассчитывай риск заново. "
    "Верни только JSON с полями risk_score (число), risk_level (строка) и explanation (строка). "
    "risk_score и risk_level скопируй точно из входных данных. "
    "explanation: 2–4 предложения на русском языке, переданные балл, категория и смысл рекомендации. "
    "Не упоминай другие баллы или категории и не сравнивай с ними. "
    "Не называй балл калиброванной вероятностью дефолта. "
    "Исходные показатели хозяйства не переданы: не придумывай причины, финансовые значения "
    "или факторы риска. Итоговое решение принимается с учётом правил банка "
    "и при необходимости проверки оператором."
)


class ScoringExplanation(BaseModel):
    """Внутренний формат ответа LLM для проверки исходного результата."""

    risk_score: float = Field(ge=0, le=1, allow_inf_nan=False)
    risk_level: Literal["low", "medium", "high"]
    explanation: str = Field(min_length=1)
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, strict=True)


def build_scoring_explanation_prompt(prediction: PredictionResponse) -> str:
    """Короткое user-сообщение с результатом; инструкции передаются отдельно в system."""
    result = prediction.model_dump(include={
        "risk_score", "risk_level", "recommendation", "model_version",
    })
    return json.dumps(result, ensure_ascii=False)


def validate_scoring_explanation(
    response: LLMChatResponse, prediction: PredictionResponse,
) -> LLMChatResponse:
    """Проверяет повторённые поля и явные противоречия перед возвратом текста."""
    try:
        content = response.answer.strip()
        lines = content.splitlines()
        if lines[0].strip().lower() in {"```", "```json"} and lines[-1].strip() == "```":
            content = "\n".join(lines[1:-1])
        explanation = ScoringExplanation.model_validate(
            json.loads(content, parse_constant=reject_non_finite_json),
        )
        if explanation.risk_score != prediction.risk_score or explanation.risk_level != prediction.risk_level:
            logger.warning(
                "LLM scoring fields mismatch | request_id=%s | expected_score=%s | expected_level=%s "
                "| received_score=%s | received_level=%s",
                response.request_id, prediction.risk_score, prediction.risk_level,
                explanation.risk_score, explanation.risk_level,
            )
            raise ValueError("Repeated scoring fields do not match")
        text = explanation.explanation.casefold()
        levels = re.findall(r"\b(low|medium|high)\b", text)
        for adjective in re.findall(r"\b(низк\w*|средн\w*|высок\w*)\s+(?:уров\w*\s+)?риск\w*", text):
            levels.append("low" if adjective.startswith("низк") else "medium" if adjective.startswith("средн") else "high")
        if any(level != prediction.risk_level for level in levels):
            raise ValueError("Explanation mentions another risk level")
        scores = re.findall(
            r"(?:risk_score|балл(?:а|у|ом|е)?|оценк[а-я]*)(?:\s+риска)?\s*[:=—–-]?\s*(\d+(?:[.,]\d+)?)",
            text,
        )
        if any(float(score.replace(",", ".")) != prediction.risk_score for score in scores):
            raise ValueError("Explanation mentions another score")
    except (ValueError, TypeError) as exc:
        reason = str(exc) if type(exc) is ValueError else type(exc).__name__
        logger.warning(
            "LLM scoring explanation rejected | request_id=%s | reason=%s", response.request_id, reason,
        )
        raise HTTPException(status_code=502, detail="LLM explanation failed scoring consistency checks") from exc
    return response.model_copy(update={"answer": explanation.explanation})


def reject_non_finite_json(value: str):
    raise ValueError(f"Non-finite JSON constant: {value}")


async def call_llm(
    client: httpx.AsyncClient, data: LLMChatRequest, request_id: str,
    *, system_prompt: str = SYSTEM_PROMPT,
) -> LLMChatResponse:
    payload = {
        "model": data.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": data.prompt},
        ],
        "temperature": data.temperature,
        "max_tokens": data.max_tokens,
        "stream": False,
    }
    logger.info("LLM request started | request_id=%s | model=%s", request_id, data.model)
    started = time.perf_counter()
    try:
        # HTTPX limits individual network operations; this also limits the whole call.
        async with asyncio.timeout(config.LLM_TIMEOUT_SECONDS):
            response = await client.post(
                f"{config.LLM_BASE_URL.rstrip('/')}/chat/completions",
                headers={"Content-Type": "application/json", "Cache-Control": "no-cache, no-store"},
                json=payload,
            )
    except (TimeoutError, httpx.TimeoutException) as exc:
        logger.warning("LLM timeout | request_id=%s", request_id)
        raise HTTPException(status_code=504, detail="LLM request timeout") from exc
    except httpx.RequestError as exc:
        logger.warning("LLM connection error | request_id=%s | type=%s", request_id, type(exc).__name__)
        raise HTTPException(status_code=503, detail="LLM service is unavailable") from exc
    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    if response.status_code != 200:
        logger.warning("LLM upstream error | request_id=%s | status=%s", request_id, response.status_code)
        raise HTTPException(
            status_code=502,
            detail=f"LLM provider returned an error (HTTP {response.status_code})",
        )
    try:
        completion = response.json(parse_constant=reject_non_finite_json)
    except ValueError as exc:
        logger.warning("LLM invalid JSON | request_id=%s", request_id)
        raise HTTPException(status_code=502, detail="LLM provider returned invalid JSON") from exc

    try:
        if not isinstance(completion, dict):
            raise ValueError("Expected a JSON object")
        choices = completion.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ValueError("Expected at least one choice")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            raise ValueError("Expected a message object")
        answer = message.get("content")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Expected non-empty message content")
        result = LLMChatResponse(
            request_id=request_id, requested_model=data.model,
            returned_model=completion.get("model"), answer=answer,
            reasoning=message.get("reasoning_content"), finish_reason=choice.get("finish_reason"),
            usage=completion.get("usage"), latency_ms=latency_ms,
        )
    except ValueError as exc:
        logger.warning("LLM invalid response structure | request_id=%s", request_id)
        raise HTTPException(status_code=502, detail="LLM provider returned an invalid response") from exc
    logger.info(
        "LLM request completed | request_id=%s | model=%s | latency_ms=%s | finish_reason=%s | provider_route=%s",
        request_id, result.returned_model, result.latency_ms, result.finish_reason,
        response.headers.get("x-vireonix-route", "unknown"),
    )
    return result
