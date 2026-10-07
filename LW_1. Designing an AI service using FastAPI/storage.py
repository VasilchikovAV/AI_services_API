"""Временное хранилище одного процесса. После перезапуска оно очищается."""

from threading import Lock

from schemas import PredictionResponse


predictions: dict[str, PredictionResponse] = {}
_lock = Lock()


def save_prediction(result: PredictionResponse) -> None:
    # Блокировка защищает словарь при одновременных HTTP-запросах.
    with _lock:
        predictions[result.request_id] = result


def get_prediction(request_id: str) -> PredictionResponse | None:
    with _lock:
        return predictions.get(request_id)


def list_predictions() -> list[PredictionResponse]:
    with _lock:
        # Порядок добавления записей в dict позволяет получить новые записи первыми.
        return list(reversed(predictions.values()))
