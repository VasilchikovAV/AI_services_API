"""Асинхронное сохранение и чтение прогнозов в SQLite."""

from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

from schemas import PredictionResponse


DATABASE_PATH = Path(__file__).resolve().parent / "agro_scoring.db"


@asynccontextmanager
async def get_db():
    """Открывает соединение на одну операцию и закрывает его при выходе."""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        # Настройка ожидания блокировки задаётся для каждого соединения.
        await db.execute("PRAGMA busy_timeout = 5000")
        yield db


async def init_db() -> None:
    """Создаёт файл базы и таблицу, сохраняя существующие записи."""
    async with get_db() as db:
        await db.execute("PRAGMA journal_mode = WAL")
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                farm_id TEXT NOT NULL,
                risk_score REAL NOT NULL,
                risk_level TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                model_version TEXT NOT NULL,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_request_id ON predictions(request_id)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_risk_level ON predictions(risk_level)"
        )
        await db.commit()


async def save_prediction(result: PredictionResponse, source: str) -> None:
    """Сохраняет результат; функция завершается только после COMMIT."""
    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO predictions (
                request_id, farm_id, risk_score, risk_level,
                recommendation, model_version, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.request_id, result.farm_id, result.risk_score,
                result.risk_level, result.recommendation, result.model_version, source,
            ),
        )
        await db.commit()


async def save_predictions_batch(results: list[PredictionResponse]) -> None:
    """Сохраняет весь пакет одной транзакцией: executemany и один COMMIT."""
    rows = [
        (
            result.request_id, result.farm_id, result.risk_score,
            result.risk_level, result.recommendation, result.model_version, "batch",
        )
        for result in results
    ]
    async with get_db() as db:
        await db.execute("BEGIN")
        await db.executemany(
            """
            INSERT INTO predictions (
                request_id, farm_id, risk_score, risk_level,
                recommendation, model_version, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        await db.commit()


async def get_predictions_by_request_id(request_id: str) -> list[PredictionResponse]:
    """Читает все результаты с указанным ID HTTP-запроса."""
    async with get_db() as db:
        async with db.execute(
            """
            SELECT request_id, farm_id, risk_score, risk_level,
                   recommendation, model_version
            FROM predictions
            WHERE request_id = ?
            ORDER BY id
            """,
            (request_id,),
        ) as cursor:
            rows = await cursor.fetchall()
    return [PredictionResponse.model_validate(dict(row)) for row in rows]


async def get_predictions_by_weekday_page(
    weekday_number: str, page: int, per_page: int,
) -> tuple[list[PredictionResponse], int]:
    """Читает страницу и общее число записей за день недели из одного снимка БД."""
    offset = (page - 1) * per_page
    async with get_db() as db:
        # Оба SELECT видят одни данные, даже если параллельно добавляются прогнозы.
        await db.execute("BEGIN")
        async with db.execute(
            "SELECT COUNT(*) FROM predictions WHERE strftime('%w', created_at) = ?",
            (weekday_number,),
        ) as cursor:
            total_items = (await cursor.fetchone())[0]
        rows = []
        if offset < total_items:
            async with db.execute(
                """
                SELECT request_id, farm_id, risk_score, risk_level,
                       recommendation, model_version
                FROM predictions
                WHERE strftime('%w', created_at) = ?
                ORDER BY id DESC
                LIMIT ? OFFSET ?
                """,
                (weekday_number, per_page, offset),
            ) as cursor:
                rows = await cursor.fetchall()
        await db.commit()
    return [PredictionResponse.model_validate(dict(row)) for row in rows], total_items


async def list_predictions(limit: int, risk_level: str | None) -> list[PredictionResponse]:
    """Фильтрует историю до LIMIT; новые записи возвращаются первыми."""
    async with get_db() as db:
        if risk_level is None:
            query = """
                SELECT request_id, farm_id, risk_score, risk_level,
                       recommendation, model_version
                FROM predictions
                ORDER BY id DESC
                LIMIT ?
            """
            parameters = (limit,)
        else:
            query = """
                SELECT request_id, farm_id, risk_score, risk_level,
                       recommendation, model_version
                FROM predictions
                WHERE risk_level = ?
                ORDER BY id DESC
                LIMIT ?
            """
            parameters = (risk_level, limit)
        async with db.execute(query, parameters) as cursor:
            rows = await cursor.fetchall()
    return [PredictionResponse.model_validate(dict(row)) for row in rows]
