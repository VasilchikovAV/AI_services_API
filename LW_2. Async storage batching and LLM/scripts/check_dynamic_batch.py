"""Send eight separate queued requests together and check their responses."""

import argparse
import json
import math
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUEST_COUNT = 8
HTTP_TIMEOUT_SECONDS = 10
EXAMPLES = (("low", 0.1), ("medium", 0.4), ("high", 0.9))


def request_json(url, *, payload=None, request_id=None):
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    if request_id is not None:
        headers["X-Request-ID"] = request_id
    request = Request(url, data=data, headers=headers)
    started = time.perf_counter()
    result = {"status": None, "headers": {}, "body": None, "error": None}
    try:
        try:
            response = urlopen(request, timeout=HTTP_TIMEOUT_SECONDS)
        except HTTPError as error:
            response = error
        with response:
            result["status"] = response.code
            result["headers"] = {
                key.lower(): value for key, value in response.headers.items()
            }
            raw = response.read().decode("utf-8")
        try:
            result["body"] = json.loads(raw)
        except json.JSONDecodeError:
            result["error"] = "Response is not JSON"
            result["body"] = raw
    except (URLError, OSError, UnicodeError) as error:
        result["error"] = str(error)
    result["client_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return result


def send_one(base_url, run_id, index, example, barrier):
    level, score, payload = example
    payload = dict(payload)
    payload["farm_id"] = f"FARM-DYN-{index + 1:03d}"
    request_id = f"{run_id}-{index + 1:02d}"
    try:
        barrier.wait(timeout=HTTP_TIMEOUT_SECONDS)
    except threading.BrokenBarrierError:
        return {
            "request_id": request_id,
            "farm_id": payload["farm_id"],
            "ok": False,
            "error": "Could not synchronize client threads",
        }
    response = request_json(
        f"{base_url}/predict-queued", payload=payload, request_id=request_id
    )
    body = response["body"] if isinstance(response["body"], dict) else {}
    actual_score = body.get("risk_score")
    checks = {
        "http_200": response["status"] == 200,
        "valid_json": response["error"] is None and bool(body),
        "header_id": response["headers"].get("x-request-id") == request_id,
        "body_id": body.get("request_id") == request_id,
        "farm_id": body.get("farm_id") == payload["farm_id"],
        "risk_score": isinstance(actual_score, (int, float))
        and math.isclose(actual_score, score, abs_tol=1e-9),
        "risk_level": body.get("risk_level") == level,
    }
    return {
        "request_id": request_id,
        "farm_id": payload["farm_id"],
        "expected_score": score,
        "expected_level": level,
        **response,
        "checks": checks,
        "ok": all(checks.values()),
    }


def save_report(path, report):
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Saved: {path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    base_url = args.url.rstrip("/")
    examples = []
    for level, score in EXAMPLES:
        payload = json.loads(
            (PROJECT_ROOT / "examples" / f"{level}.json").read_text(encoding="utf-8-sig")
        )
        examples.append((level, score, payload))

    run_id = f"dyn-{uuid.uuid4().hex[:12]}"
    report = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "endpoint": f"{base_url}/predict-queued",
        "request_count": REQUEST_COUNT,
        "queue_before": request_json(f"{base_url}/queue-status"),
        "results": [],
        "all_ok": False,
    }
    print(f"Run: {run_id}")
    before = report["queue_before"]
    if before["status"] != 200 or not isinstance(before["body"], dict):
        print(f"Cannot read /queue-status: HTTP {before['status']}; {before['error'] or before['body']}")
        save_report(args.output, report)
        return 1
    print("Queue before:", json.dumps(before["body"], ensure_ascii=False))

    barrier = threading.Barrier(REQUEST_COUNT)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=REQUEST_COUNT) as executor:
        futures = [
            executor.submit(
                send_one, base_url, run_id, index, examples[index % len(examples)], barrier
            )
            for index in range(REQUEST_COUNT)
        ]
        results = [future.result() for future in futures]
    report["total_client_ms"] = round((time.perf_counter() - started) * 1000, 3)
    report["results"] = results
    report["queue_after"] = request_json(f"{base_url}/queue-status")
    report["all_ok"] = all(result["ok"] for result in results)

    print("HTTP  Farm          Score  Level   Client ms  Result  Request ID")
    for result in results:
        body = result.get("body")
        body = body if isinstance(body, dict) else {}
        print(
            f"{str(result.get('status', '-')):<5} {result['farm_id']:<13} "
            f"{str(body.get('risk_score', '-')):<6} {str(body.get('risk_level', '-')):<7} "
            f"{result.get('client_ms', 0):>9.3f}  "
            f"{'PASS' if result['ok'] else 'FAIL':<6} {result['request_id']}"
        )
        if not result["ok"]:
            failed = [name for name, passed in result.get("checks", {}).items() if not passed]
            print("      Failed checks:", ", ".join(failed) or result.get("error"))
            if result.get("error"):
                print("      Error:", result["error"])
    print("Queue after:", json.dumps(report["queue_after"]["body"], ensure_ascii=False))
    print(f"HTTP checks: {'PASS' if report['all_ok'] else 'FAIL'} ({sum(r['ok'] for r in results)}/{REQUEST_COUNT})")
    print(f"Find 'Dynamic batch formed | size=...' with IDs starting '{run_id}' in the server log.")
    print("Successful HTTP responses alone do not confirm the sizes of model batches.")
    save_report(args.output, report)
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
