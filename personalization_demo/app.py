"""FastAPI app for the Osler personalized belief demo."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from demo.forecast_api import (
    capabilities_response,
    forecast_response,
    model_health_response,
    validation_summary_response,
)
from personalization_demo.runtime import sample_payload


APP_DIR = Path(__file__).resolve().parent
INDEX_PATH = APP_DIR / "static" / "index.html"

app = FastAPI(
    title="Osler Personalized Belief Demo",
    version="0.1.0",
    description=(
        "Research demo for validated Osler predict-update belief features. "
        "No clinical, causal, counterfactual, or treatment claims."
    ),
)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    if not INDEX_PATH.exists():
        raise HTTPException(status_code=500, detail="demo index.html is missing")
    return INDEX_PATH.read_text(encoding="utf-8")


@app.get("/api/forecast/capabilities")
def get_capabilities() -> dict[str, Any]:
    return capabilities_response()


@app.get("/api/forecast/validation-summary")
def get_validation_summary() -> dict[str, Any]:
    return validation_summary_response()


@app.get("/api/health/models")
def get_model_health() -> JSONResponse:
    health = model_health_response()
    return JSONResponse(health, status_code=200 if health["ready"] else 503)


@app.get("/api/sample")
def get_sample() -> dict[str, Any]:
    return sample_payload()


@app.post("/api/forecast")
def post_forecast(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return forecast_response(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

