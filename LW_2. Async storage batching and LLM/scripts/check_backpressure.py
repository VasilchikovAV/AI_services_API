"""Check HTTP 429 using a separate local server and a temporary SQLite database."""

import argparse
import json
import logging
import socket
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from check_dynamic_batch import EXAMPLES, PROJECT_ROOT, request_json, save_report, send_one


PROFILE = {
    "QUEUE_MAX_SIZE": 2,
    "MAX_BATCH_SIZE": 1,
    "MAX_BATCH_WAIT_MS": 0,
    "INFERENCE_TIMEOUT_SECONDS": 5.0,
    "MODEL_FIXED_OVERHEAD_SECONDS": 1.0,
    "MODEL_PER_ITEM_SECONDS": 0.003,
}
REQUEST_COUNT = 8


@contextmanager
def isolated_server(port, log_path, *, profile=None):
    """Change settings only in this process; stop the server before deleting its DB."""
    sys.path.insert(0, str(PROJECT_ROOT))
    import config
    import uvicorn

    for name, value in (PROFILE if profile is None else profile).items():
        setattr(config, name, value)
    logging.basicConfig(
        filename=log_path, filemode="w", encoding="utf-8", level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s", force=True,
    )
    import main
    import storage

    temp_root = Path(tempfile.gettempdir()).resolve()
    with tempfile.TemporaryDirectory(prefix="agro-backpressure-", dir=temp_root) as directory:
        database_directory = Path(directory).resolve()
        if database_directory.parent != temp_root:
            raise RuntimeError("Unexpected temporary directory location")
        storage.DATABASE_PATH = database_directory / "predictions.db"
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            listener.bind(("127.0.0.1", port))
            listener.setblocking(False)
            actual_port = listener.getsockname()[1]
            server = uvicorn.Server(uvicorn.Config(
                main.app, host="127.0.0.1", port=actual_port,
                loop="asyncio", http="h11", lifespan="on",
                log_config=None, access_log=False, timeout_graceful_shutdown=5,
            ))
            thread = threading.Thread(
                target=server.run, kwargs={"sockets": [listener]}, daemon=True,
            )
            thread.start()
            try:
                deadline = time.monotonic() + 15
                while not server.started:
                    if not thread.is_alive() or time.monotonic() >= deadline:
                        raise RuntimeError(f"Server did not start; see {log_path}")
                    time.sleep(0.05)
                yield f"http://127.0.0.1:{actual_port}"
            finally:
                server.should_exit = True
                thread.join(timeout=10)
                if thread.is_alive():
                    server.force_exit = True
                    thread.join(timeout=5)
                if thread.is_alive():
                    raise RuntimeError("The isolated server did not stop")


def check_overload(base_url, report):
    examples = [
        (level, score, json.loads(
            (PROJECT_ROOT / "examples" / f"{level}.json").read_text(encoding="utf-8-sig")
        ))
        for level, score in EXAMPLES
    ]
    report["base_url"] = base_url
    report["queue_before"] = request_json(f"{base_url}/queue-status")
    if report["queue_before"]["status"] != 200:
        raise RuntimeError("Cannot read the isolated server queue status")
    print("Server:", base_url)
    print("Queue capacity: 2; batch size: 1; model delay: 1 s; timeout: 5 s")
    barrier = threading.Barrier(REQUEST_COUNT)
    with ThreadPoolExecutor(max_workers=REQUEST_COUNT) as executor:
        futures = [
            executor.submit(
                send_one, base_url, report["run_id"], index,
                examples[index % len(examples)], barrier,
            )
            for index in range(REQUEST_COUNT)
        ]
        results = [future.result() for future in futures]

    for result in results:
        if result.get("status") == 429:
            body = result.get("body")
            body = body if isinstance(body, dict) else {}
            result["checks"] = {
                "http_429": True,
                "valid_json": result.get("error") is None and bool(body),
                "header_id": result["headers"].get("x-request-id") == result["request_id"],
                "body_id": body.get("request_id") == result["request_id"],
                "retry_after": result["headers"].get("retry-after") == "1",
                "detail": body.get("detail") == "Inference queue is full. Try again later.",
            }
        lookup = request_json(f"{base_url}/requests/{result['request_id']}/predictions")
        result["stored_result"] = lookup
        checks = result.setdefault("checks", {})
        if result.get("status") == 200:
            checks["saved_once"] = lookup["status"] == 200 and lookup["body"] == [result["body"]]
        elif result.get("status") == 429:
            checks["not_saved"] = lookup["status"] == 404
        else:
            checks["expected_status"] = False
        result["ok"] = all(checks.values())

    report["results"] = results
    report["accepted_count"] = sum(result.get("status") == 200 for result in results)
    report["rejected_count"] = sum(result.get("status") == 429 for result in results)
    report["recovery"] = send_one(
        base_url, report["run_id"], REQUEST_COUNT, examples[0], threading.Barrier(1),
    )
    report["queue_after"] = request_json(f"{base_url}/queue-status")
    report["checks"] = {
        "at_least_one_200": report["accepted_count"] > 0,
        "at_least_one_429": report["rejected_count"] > 0,
        "all_responses": all(result["ok"] for result in results),
        "recovery_200": report["recovery"]["ok"],
        "queue_empty": report["queue_after"]["status"] == 200
        and report["queue_after"]["body"].get("queue_size") == 0,
    }
    report["all_ok"] = all(report["checks"].values())


def print_results(report):
    print("HTTP  Retry-After  Result  Request ID")
    for result in report.get("results", []):
        print(
            f"{str(result.get('status', '-')):<5} "
            f"{result.get('headers', {}).get('retry-after', '-'):<12} "
            f"{'PASS' if result['ok'] else 'FAIL':<6} {result['request_id']}"
        )
        if not result["ok"]:
            failed = [name for name, passed in result.get("checks", {}).items() if not passed]
            print("      Failed checks:", ", ".join(failed), result.get("error") or "")
    rejected = next((r for r in report.get("results", []) if r.get("status") == 429), None)
    if rejected is not None:
        print("\nHTTP 429 example:")
        print("X-Request-ID:", rejected["headers"].get("x-request-id"))
        print("Retry-After:", rejected["headers"].get("retry-after"))
        print(json.dumps(rejected["body"], ensure_ascii=False, indent=2))
    if "recovery" in report:
        print("\nRecovery request:", report["recovery"].get("status"))
        print("Accepted/rejected:", report["accepted_count"], "/", report["rejected_count"])
    for name, passed in report.get("checks", {}).items():
        if not passed:
            print("Failed:", name)
    print("Backpressure checks:", "PASS" if report["all_ok"] else "FAIL")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8001, help="Isolated server port; 0 selects a free port")
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT.parent / "report" / "backpressure_result.json",
        help="JSON report path (a server log is saved beside it)",
    )
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.with_suffix(".server.log")
    report = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": f"overload-{uuid.uuid4().hex[:12]}",
        "profile": PROFILE,
        "request_count": REQUEST_COUNT,
        "server_log": str(log_path),
        "all_ok": False,
    }
    print("Run:", report["run_id"])
    try:
        with isolated_server(args.port, log_path) as base_url:
            check_overload(base_url, report)
        report["server_stopped"] = True
        print("Isolated server stopped; temporary database removed.")
    except KeyboardInterrupt:
        report["all_ok"] = False
        report["error"] = "Interrupted by user"
        print(report["error"])
    except Exception as error:
        report["all_ok"] = False
        report["error"] = str(error)
        logging.exception("Backpressure check failed")
        print("Error:", error)
    print_results(report)
    save_report(output, report)
    logging.shutdown()
    print("Server log:", log_path)
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
