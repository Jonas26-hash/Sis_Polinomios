"""Aplicación ASGI compatible con Vercel para NUMÉRICA."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from analysis import COEFFICIENTS, DEFAULT_POINTS, DEFAULT_TOLERANCE, run_analysis
from muller import MullerError
from polynomial import evaluate_polynomial, synthetic_division_steps
from theory import coefficient_signs, descartes_possibilities


PUBLIC_DIR = Path(__file__).resolve().parent / "public"
app = FastAPI(title="NUMÉRICA — Raíces de Polinomios", docs_url="/api/docs")


class AnalysisRequest(BaseModel):
    z0: float = DEFAULT_POINTS[0]
    z1: float = DEFAULT_POINTS[1]
    z2: float = DEFAULT_POINTS[2]
    tolerance: float = Field(DEFAULT_TOLERANCE, gt=0, le=0.1)
    max_iterations: int = Field(50, ge=1, le=500)


def _complex(value: complex) -> dict[str, float]:
    return {"real": value.real, "imag": value.imag}


def _safe_float(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _iteration(row: dict[str, Any]) -> dict[str, Any]:
    complex_keys = (
        "z0", "z1", "z2", "f_z2", "h0", "h1", "delta0", "delta1", "a", "b", "c",
        "discriminant", "den1", "den2", "denominator", "dx", "z_new",
    )
    result = {key: _complex(complex(row[key])) for key in complex_keys}
    result.update({"iteration": int(row["iteration"]), "ea": _safe_float(float(row["ea"])),
                   "residual": _safe_float(float(row["residual"]))})
    return result


def _coefficients(values: list[complex]) -> list[dict[str, float]]:
    return [_complex(complex(value)) for value in values]


def serialize_analysis(data: dict[str, Any]) -> dict[str, Any]:
    first = data["first"]
    roots = data["roots"]
    stages = []
    for stage in data["stages"]:
        item: dict[str, Any] = {
            "degree": stage["degree"],
            "coefficients": _coefficients(stage["coefficients"]),
            "method": stage.get("method", "Müller + Horner"),
        }
        if "root" in stage:
            item.update({"root": _complex(stage["root"]), "quotient": _coefficients(stage["quotient"]),
                         "remainder": _complex(stage["remainder"]),
                         "iteration_count": len(stage["iterations"])})
        if "roots" in stage:
            item["roots"] = [_complex(root) for root in stage["roots"]]
        stages.append(item)

    horner_steps, quotient, remainder = synthetic_division_steps(COEFFICIENTS, first.root)
    return {
        "polynomial": COEFFICIENTS,
        "parameters": {"z0": first.iterations[0]["z0"].real, "z1": first.iterations[0]["z1"].real,
                       "z2": first.iterations[0]["z2"].real},
        "descartes": {
            "positive": {"coefficients": COEFFICIENTS, "signs": coefficient_signs(COEFFICIENTS),
                         "changes": data["positive_changes"],
                         "possibilities": descartes_possibilities(data["positive_changes"])},
            "negative": {"coefficients": [value.real for value in data["negative_coefficients"]],
                         "signs": coefficient_signs(data["negative_coefficients"]),
                         "changes": data["negative_changes"],
                         "possibilities": descartes_possibilities(data["negative_changes"])},
        },
        "lagrange": {"bound": data["bound"], "ratios": data["ratios"]},
        "first": {"root": _complex(first.root), "iterations": [_iteration(row) for row in first.iterations],
                  "iteration_count": len(first.iterations), "stop_reason": first.stop_reason},
        "deflation": stages,
        "horner": {"steps": [{"step": step["paso"], "coefficient": _complex(step["coeficiente"]),
                                "product": _complex(step["producto"]), "result": _complex(step["resultado"])}
                               for step in horner_steps],
                   "quotient": _coefficients(quotient), "remainder": _complex(remainder)},
        "roots": [{"index": index, "value": _complex(root), "real": root.real, "imag": root.imag,
                   "residual": abs(evaluate_polynomial(COEFFICIENTS, root)), "modulus": abs(root),
                   "inside": abs(root) < 1} for index, root in enumerate(roots, start=1)],
        "stable": data["stable"],
    }


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(PUBLIC_DIR / "index.html")


@app.get("/app.css", include_in_schema=False)
def stylesheet() -> FileResponse:
    return FileResponse(PUBLIC_DIR / "app.css", media_type="text/css")


@app.get("/app.js", include_in_schema=False)
def javascript() -> FileResponse:
    return FileResponse(PUBLIC_DIR / "app.js", media_type="application/javascript")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/analyze")
def analyze(request: AnalysisRequest) -> dict[str, Any]:
    try:
        data = run_analysis(request.z0, request.z1, request.z2, request.tolerance, request.max_iterations)
        response = serialize_analysis(data)
        response["tolerance"] = request.tolerance
        response["max_iterations"] = request.max_iterations
        return response
    except (ValueError, ZeroDivisionError, MullerError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
