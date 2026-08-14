"""Build deterministic frontend fixtures from the live forecast contract."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from demo.forecast_api import (
    forecast_response,
    model_health_response,
    monitoring_cases_response,
    replay_forecast_request,
    validation_summary_response,
)


FIXTURE_SCHEMA_VERSION = "osler.forecast-fixtures.v1"
DEFAULT_FIXTURE_PATH = Path(__file__).resolve().parent / "forecast_api_fixtures.json"


def build_forecast_api_fixtures() -> dict[str, Any]:
    cases_response = monitoring_cases_response()
    case = cases_response["cases"][0]
    first_request = replay_forecast_request(case, 1)
    second_request = replay_forecast_request(case, 2)
    unsupported_request = replay_forecast_request(case, 2)
    unsupported_request["requested_cells"] = [
        {"target": "lactate", "horizon_hours": 24}
    ]
    return {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "generated_from_live_contract": True,
        "get_monitoring_cases": {
            "method": "GET",
            "path": "/api/monitoring/cases",
            "status": 200,
            "response": cases_response,
        },
        "post_forecast_first_prefix": {
            "method": "POST",
            "path": "/api/forecast",
            "status": 200,
            "request": first_request,
            "response": forecast_response(first_request),
        },
        "post_forecast_second_prefix": {
            "method": "POST",
            "path": "/api/forecast",
            "status": 200,
            "request": second_request,
            "response": forecast_response(second_request),
        },
        "get_validation_summary": {
            "method": "GET",
            "path": "/api/forecast/validation-summary",
            "status": 200,
            "response": validation_summary_response(),
        },
        "get_model_health_unhealthy": {
            "method": "GET",
            "path": "/api/health/models",
            "status": 503,
            "fixture_only": True,
            "response": _unhealthy_health_example(),
        },
        "post_forecast_unsupported": {
            "method": "POST",
            "path": "/api/forecast",
            "status": 200,
            "request": unsupported_request,
            "response": forecast_response(unsupported_request),
        },
    }


def _unhealthy_health_example() -> dict[str, Any]:
    health = deepcopy(model_health_response())
    health["status"] = "degraded"
    health["ready"] = False
    health["loaded_artifacts"] = max(0, health["expected_artifacts"] - 1)
    if health["checks"]:
        health["checks"][-1] = {
            **health["checks"][-1],
            "status": "error",
            "error": "FixtureOnlyError: strict artifact verification failed",
        }
    return health


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_FIXTURE_PATH)
    args = parser.parse_args()
    rendered = json.dumps(
        build_forecast_api_fixtures(), indent=2, ensure_ascii=False
    ) + "\n"
    if args.write:
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
