"""Implementación manual del método de Müller y deflación sucesiva."""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Sequence

from polynomial import evaluate_polynomial, lagrange_bound, quadratic_roots, synthetic_division, trim_small_parts


class MullerError(RuntimeError):
    """Error numérico controlado del método de Müller."""


@dataclass
class MullerResult:
    root: complex
    iterations: list[dict[str, complex | float | int | str]]
    converged: bool
    stop_reason: str


def muller(
    coefficients: Sequence[complex],
    z0: complex,
    z1: complex,
    z2: complex,
    tolerance: float = 1e-5,
    max_iterations: int = 50,
    residual_tolerance: float = 1e-12,
) -> MullerResult:
    """Aproxima una raíz, manteniendo un registro completo de cada iteración."""
    if tolerance <= 0 or max_iterations < 1:
        raise ValueError("La tolerancia y el máximo de iteraciones deben ser positivos.")
    x0, x1, x2 = complex(z0), complex(z1), complex(z2)
    rows: list[dict[str, complex | float | int | str]] = []

    for iteration in range(1, max_iterations + 1):
        h0, h1 = x1 - x0, x2 - x1
        if abs(h0) == 0 or abs(h1) == 0 or abs(h0 + h1) == 0:
            raise MullerError("División por cero: los puntos de Müller son coincidentes o simétricos.")
        f0 = evaluate_polynomial(coefficients, x0)
        f1 = evaluate_polynomial(coefficients, x1)
        f2 = evaluate_polynomial(coefficients, x2)
        delta0 = (f1 - f0) / h0
        delta1 = (f2 - f1) / h1
        a = (delta1 - delta0) / (h1 + h0)
        b = a * h1 + delta1
        c = f2
        discriminant = cmath.sqrt(b * b - 4 * a * c)
        den1, den2 = b + discriminant, b - discriminant
        denominator = den1 if abs(den1) >= abs(den2) else den2
        if abs(denominator) == 0:
            raise MullerError("División por cero: ambos denominadores son nulos.")
        dx = -2 * c / denominator
        x3 = x2 + dx
        ea = abs((x3 - x2) / x3) if abs(x3) else math.inf
        residual = abs(evaluate_polynomial(coefficients, x3))
        rows.append(
            {
                "iteration": iteration, "z0": x0, "z1": x1, "z2": x2,
                "f_z2": f2, "h0": h0, "h1": h1, "delta0": delta0,
                "delta1": delta1, "a": a, "b": b, "c": c,
                "discriminant": discriminant, "den1": den1, "den2": den2,
                "denominator": denominator, "dx": dx, "z_new": x3,
                "ea": ea, "residual": residual,
            }
        )
        if residual <= residual_tolerance:
            return MullerResult(trim_small_parts(x3), rows, True, "residuo")
        if ea <= tolerance:
            return MullerResult(trim_small_parts(x3), rows, True, "tolerancia")
        if abs(dx) <= 1e-15 * max(1.0, abs(x3)):
            raise MullerError("El método se estancó sin satisfacer el criterio de parada.")
        x0, x1, x2 = x1, x2, x3

    return MullerResult(trim_small_parts(x2), rows, False, "máximo de iteraciones")


def _seed_sets(coefficients: Sequence[complex]) -> list[tuple[complex, complex, complex]]:
    radius, _ = lagrange_bound(coefficients)
    seeds = [(0j, 0.5 + 0j, 1 + 0j), (0j, -0.5 + 0j, -1 + 0j)]
    for angle in (0.0, math.pi / 3, 2 * math.pi / 3, math.pi, 4 * math.pi / 3, 5 * math.pi / 3):
        direction = cmath.exp(1j * angle)
        seeds.append((0j, 0.35 * radius * direction, 0.75 * radius * direction))
    return seeds


def find_root_for_deflation(
    coefficients: Sequence[complex], tolerance: float, max_iterations: int
) -> MullerResult:
    """Busca una raíz de un polinomio reducido con semillas deterministas."""
    last_error: Exception | None = None
    for seeds in _seed_sets(coefficients):
        try:
            result = muller(coefficients, *seeds, tolerance=tolerance, max_iterations=max_iterations)
            if result.converged and abs(evaluate_polynomial(coefficients, result.root)) <= max(1e-8, tolerance):
                return result
        except (MullerError, ZeroDivisionError) as error:
            last_error = error
    raise MullerError(f"No se encontró una raíz válida para deflactar. {last_error or ''}".strip())


def solve_all_roots(
    coefficients: Sequence[complex],
    first_result: MullerResult,
    tolerance: float = 1e-5,
    max_iterations: int = 50,
) -> tuple[list[complex], list[dict[str, object]]]:
    """Obtiene todas las raíces con Müller, deflación y fórmula cuadrática."""
    current = [complex(value) for value in coefficients]
    roots = [first_result.root]
    stages: list[dict[str, object]] = []
    quotient, remainder = synthetic_division(current, first_result.root)
    stages.append({"degree": len(current) - 1, "root": first_result.root, "coefficients": current,
                   "quotient": quotient, "remainder": remainder, "iterations": first_result.iterations})
    current = quotient

    while len(current) - 1 > 2:
        result = find_root_for_deflation(current, tolerance, max_iterations)
        root = result.root
        quotient, remainder = synthetic_division(current, root)
        roots.append(root)
        stages.append({"degree": len(current) - 1, "root": root, "coefficients": current,
                       "quotient": quotient, "remainder": remainder, "iterations": result.iterations})
        current = quotient

    if len(current) == 3:
        final_roots = quadratic_roots(current)
        roots.extend(trim_small_parts(root) for root in final_roots)
        stages.append({"degree": 2, "coefficients": current, "roots": final_roots, "method": "fórmula general"})
    elif len(current) == 2:
        roots.append(trim_small_parts(-current[1] / current[0]))
        stages.append({"degree": 1, "coefficients": current, "roots": roots[-1:], "method": "ecuación lineal"})
    return roots, stages
