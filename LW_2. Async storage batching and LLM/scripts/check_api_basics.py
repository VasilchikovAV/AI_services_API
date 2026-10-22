"""Check basic API responses and record single-call timings without using LLM."""

import argparse
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path

from check_dynamic_batch import PROJECT_ROOT, request_json, save_report


def body_object(response):
    return response["body"] if isinstance(response["body"], dict) else {}


def add_result(report, number, method, route, response, expected_status, request_id, checks):
    try:
        server_ms = float(response["headers"].get("x-process-time", "nan")) * 1000
        if not math.isfinite(server_ms) or server_ms < 0:
            server_ms = None
    except ValueError:
        server_ms = None
    result = {
        "case_number": number, "method": method, "route": route,
        "request_id": request_id, "expected_status": expected_status,
        **response, "server_ms": server_ms,
        "checks": {
            "http_status": response["status"] == expected_status,
            "json_object": response["error"] is None and isinstance(response["body"], dict),
            "header_id": response["headers"].get("x-request-id") == request_id,
            "time_header": server_ms is not None,
            **checks,
        },
    }
    result["ok"] = all(result["checks"].values())
    report["results"].append(result)
    return result


def check(base_url, report):
    run_id = report["run_id"]
    health_id = f"{run_id}-health"
    response = request_json(f"{base_url}/health", request_id=health_id)
    add_result(report, 1, "GET", "/health", response, 200, health_id,
               {"status_ok": body_object(response).get("status") == "ok"})

    payload = json.loads((PROJECT_ROOT / "examples" / "high.json").read_text(encoding="utf-8-sig"))
    payload["farm_id"] = "FARM-BASELINE-HIGH"
    prediction_id = f"{run_id}-predict"
    prediction = request_json(f"{base_url}/predict", payload=payload, request_id=prediction_id)
    data = body_object(prediction)
    add_result(report, 2, "POST", "/predict", prediction, 200, prediction_id, {
        "body_id": data.get("request_id") == prediction_id,
        "farm_id": data.get("farm_id") == payload["farm_id"],
        "risk_score": data.get("risk_score") == .9,
        "risk_level": data.get("risk_level") == "high",
    })

    region_id = f"{run_id}-region"
    response = request_json(
        f"{base_url}/predict", payload=dict(payload, region="UnknownRegion"), request_id=region_id,
    )
    rejected = add_result(report, 4, "POST", "/predict (unknown region)", response, 400, region_id, {
        "body_id": body_object(response).get("request_id") == region_id,
        "detail": isinstance(body_object(response).get("detail"), str),
    })
    rejected_history = request_json(f"{base_url}/requests/{region_id}/predictions")
    rejected["stored_result"] = rejected_history
    rejected["checks"]["not_saved"] = rejected_history["status"] == 404
    rejected["ok"] = all(rejected["checks"].values())

    read_id = f"{run_id}-read"
    response = request_json(f"{base_url}/predictions/{prediction_id}", request_id=read_id)
    add_result(report, 5, "GET", "/predictions/{saved_id}", response, 200, read_id, {
        "saved_prediction": response["body"] == prediction["body"] and prediction["status"] == 200,
        "saved_id": body_object(response).get("request_id") == prediction_id,
    })

    missing_id = f"{run_id}-missing"
    response = request_json(f"{base_url}/predictions/not-found-{uuid.uuid4().hex}", request_id=missing_id)
    add_result(report, 6, "GET", "/predictions/{unknown_id}", response, 404, missing_id, {
        "body_id": body_object(response).get("request_id") == missing_id,
        "detail": body_object(response).get("detail") == "Prediction not found",
    })
    report["all_ok"] = all(result["ok"] for result in report["results"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    base_url = args.url.rstrip("/")
    report = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": f"baseline-{uuid.uuid4().hex[:12]}",
        "base_url": base_url, "timing_note": "One observation per route; not a statistical benchmark",
        "results": [], "all_ok": False,
    }
    print("Run:", report["run_id"])
    check(base_url, report)
    print("Case  HTTP  Server ms  Client ms  Result  Route")
    for result in report["results"]:
        timing = "-" if result["server_ms"] is None else f"{result['server_ms']:.3f}"
        print(
            f"{result['case_number']:<5} {str(result['status']):<5} {timing:>9} "
            f"{result['client_ms']:>10.3f}  {'PASS' if result['ok'] else 'FAIL':<6} "
            f"{result['method']} {result['route']}"
        )
        if not result["ok"]:
            print("      Failed:", ", ".join(k for k, v in result["checks"].items() if not v))
            if result.get("error"):
                print("      Error:", result["error"])
    for result in report["results"]:
        if result["expected_status"] in (400, 404):
            print(f"\nHTTP {result['expected_status']} example; X-Request-ID: {result['headers'].get('x-request-id')}")
            print(json.dumps(result["body"], ensure_ascii=False, indent=2))
    print("\nBasic API checks:", "PASS" if report["all_ok"] else "FAIL")
    print("One prediction was submitted; no LLM calls were made. Timings are individual observations.")
    save_report(args.output, report)
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
