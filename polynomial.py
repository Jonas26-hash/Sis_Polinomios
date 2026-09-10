"""Operaciones polinomiales sin solucionadores externos."""

from __future__ import annotations

import cmath
from typing import Iterable, Sequence


def _as_complex(coefficients: Iterable[complex]) -> list[complex]:
    values = [complex(value) for value in coefficients]
    if not values:
        raise ValueError("El polinomio debe contener al menos un coeficiente.")
    return values


def evaluate_polynomial(coefficients: Sequence[complex], z: complex) -> complex:
    """Evalúa un polinomio mediante el esquema de Horner."""
    values = _as_complex(coefficients)
    result = values[0]
    for coefficient in values[1:]:
        result = result * z + coefficient
    return result


def synthetic_division(
    coefficients: Sequence[complex], root: complex
) -> tuple[list[complex], complex]:
    """Divide P(z) entre (z-root) y devuelve cociente y residuo."""
    values = _as_complex(coefficients)
    if len(values) < 2:
        raise ValueError("Se requiere un polinomio de grado al menos uno.")
    quotient = [values[0]]
    for coefficient in values[1:-1]:
        quotient.append(coefficient + quotient[-1] * root)
    remainder = values[-1] + quotient[-1] * root
    return quotient, remainder


def synthetic_division_steps(
    coefficients: Sequence[complex], root: complex
) -> tuple[list[dict[str, complex]], list[complex], complex]:
    """Expone las operaciones de la división sintética para la UI."""
    values = _as_complex(coefficients)
    quotient = [values[0]]
    steps: list[dict[str, complex]] = []
    for index, coefficient in enumerate(values[1:], start=1):
        product = quotient[-1] * root
        result = coefficient + product
        steps.append(
            {"paso": index, "coeficiente": coefficient, "producto": product, "resultado": result}
        )
        if index < len(values) - 1:
            quotient.append(result)
    return steps, quotient, steps[-1]["resultado"]


def multiply_by_linear(coefficients: Sequence[complex], root: complex) -> list[complex]:
    """Multiplica Q(z) por (z-root), útil para auditar una deflación."""
    values = _as_complex(coefficients)
    result = [0j] * (len(values) + 1)
    for index, coefficient in enumerate(values):
        result[index] += coefficient
        result[index + 1] -= root * coefficient
    return result


def quadratic_roots(coefficients: Sequence[complex]) -> list[complex]:
    """Resuelve az²+bz+c=0 con la fórmula general y cmath.sqrt."""
    values = _as_complex(coefficients)
    if len(values) != 3:
        raise ValueError("Se requieren exactamente tres coeficientes.")
    a, b, c = values
    if abs(a) == 0:
        raise ZeroDivisionError("El coeficiente cuadrático no puede ser cero.")
    discriminant = cmath.sqrt(b * b - 4 * a * c)
    return [(-b + discriminant) / (2 * a), (-b - discriminant) / (2 * a)]


def lagrange_bound(coefficients: Sequence[complex]) -> tuple[float, list[float]]:
    """Calcula R = 1 + max(|a_k/a_n|), cota global Cauchy-Lagrange."""
    values = _as_complex(coefficients)
    leading = values[0]
    if abs(leading) == 0:
        raise ValueError("El coeficiente principal debe ser distinto de cero.")
    ratios = [abs(value / leading) for value in values[1:]]
    return 1.0 + max(ratios, default=0.0), ratios


def trim_small_parts(value: complex, tolerance: float = 1e-12) -> complex:
    real = 0.0 if abs(value.real) < tolerance else value.real
    imag = 0.0 if abs(value.imag) < tolerance else value.imag
    return complex(real, imag)
