"""Check inference HTTP 504, absence of saved results, and recovery on an isolated server."""

import argparse
import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from check_backpressure import isolated_server
from check_dynamic_batch import PROJECT_ROOT, request_json, save_report, send_one


PROFILE = {
    "QUEUE_MAX_SIZE": 100,
    "MAX_BATCH_SIZE": 8,
    "MAX_BATCH_WAIT_MS": 50,
    "INFERENCE_TIMEOUT_SECONDS": 0.001,
    "MODEL_FIXED_OVERHEAD_SECONDS": 0.020,
    "MODEL_PER_ITEM_SECONDS": 0.003,
}


def check_timeouts(base_url, report):
    payload = json.loads((PROJECT_ROOT / "examples" / "high.json").read_text(encoding="utf-8-sig"))
    batch = json.loads((PROJECT_ROOT / "examples" / "batch.json").read_text(encoding="utf-8-sig"))
    report["base_url"] = base_url
    report["queue_before"] = request_json(f"{base_url}/queue-status")
    print("Server:", base_url)
    print("Inference timeout: 0.001 s; model duration: at least 0.023 s")
    cases = (("single", "/predict", payload), ("batch", "/batch-predict", batch),
             ("queued", "/predict-queued", payload))
    report["results"] = []
    for name, route, data in cases:
        request_id = f"{report['run_id']}-{name}"
        response = request_json(f"{base_url}{route}", payload=data, request_id=request_id)
        body = response["body"] if isinstance(response["body"], dict) else {}
        checks = {
            "http_504": response["status"] == 504,
            "valid_json": response["error"] is None and bool(body),
            "header_id": response["headers"].get("x-request-id") == request_id,
            "body_id": body.get("request_id") == request_id,
            "detail": body.get("detail") == "Inference timeout",
        }
        report["results"].append({
            "route": route, "request_id": request_id, **response, "checks": checks, "ok": False,
        })

    # Restore the normal limit only in this process and exercise the same worker.
    import config
    config.INFERENCE_TIMEOUT_SECONDS = 2.0
    report["recovery"] = send_one(
        base_url, report["run_id"], 0, ("high", 0.9, payload), threading.Barrier(1),
    )
    # Check history after recovery so a delayed write cannot pass unnoticed.
    for result in report["results"]:
        lookup = request_json(f"{base_url}/requests/{result['request_id']}/predictions")
        result["stored_result"] = lookup
        result["checks"]["not_saved"] = lookup["status"] == 404
        result["ok"] = all(result["checks"].values())
    recovery = report["recovery"]
    recovered_record = request_json(f"{base_url}/requests/{recovery['request_id']}/predictions")
    recovery["stored_result"] = recovered_record
    report["queue_after"] = request_json(f"{base_url}/queue-status")
    state = report["queue_after"]["body"]
    state = state if isinstance(state, dict) else {}
    report["checks"] = {
        "all_timeouts": all(result["ok"] for result in report["results"]),
        "recovery_200": recovery["ok"],
        "recovery_saved_once": recovered_record["status"] == 200
        and recovered_record["body"] == [recovery["body"]],
        "queue_empty": report["queue_after"]["status"] == 200 and state.get("queue_size") == 0,
        "normal_timeout_restored": state.get("inference_timeout_seconds") == 2.0,
    }
    report["all_ok"] = all(report["checks"].values())


def print_results(report):
    print("HTTP  Route                Not saved  Result  Request ID")
    for result in report.get("results", []):
        print(
            f"{str(result['status']):<5} {result['route']:<20} "
            f"{str(result['checks'].get('not_saved', False)):<10} "
            f"{'PASS' if result['ok'] else 'FAIL':<6} {result['request_id']}"
        )
        if not result["ok"]:
            print("      Failed:", ", ".join(k for k, v in result["checks"].items() if not v))
    if report.get("results"):
        sample = report["results"][-1]
        print("\nHTTP 504 example:")
        print("X-Request-ID:", sample["headers"].get("x-request-id"))
        print(json.dumps(sample["body"], ensure_ascii=False, indent=2))
    if "recovery" in report:
        print("\nRecovery request:", report["recovery"]["status"])
    for name, passed in report.get("checks", {}).items():
        if not passed:
            print("Failed:", name)
    print("Inference timeout checks:", "PASS" if report["all_ok"] else "FAIL")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8001, help="Isolated server port; 0 selects a free port")
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT.parent / "report" / "inference_timeout_result.json",
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
        "run_id": f"timeout-{uuid.uuid4().hex[:12]}",
        "profile": PROFILE,
        "server_log": str(log_path),
        "all_ok": False,
    }
    print("Run:", report["run_id"])
    try:
        with isolated_server(args.port, log_path, profile=PROFILE) as base_url:
            check_timeouts(base_url, report)
        report["server_stopped"] = True
        print("Isolated server stopped; temporary database removed.")
    except KeyboardInterrupt:
        report["all_ok"] = False
        report["error"] = "Interrupted by user"
        print(report["error"])
    except Exception as error:
        report["all_ok"] = False
        report["error"] = str(error)
        logging.exception("Inference timeout check failed")
        print("Error:", error)
    print_results(report)
    save_report(output, report)
    logging.shutdown()
    print("Server log:", log_path)
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
